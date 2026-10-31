"""
baselines.py
============
Simple models Track B has to beat in the Phase 7 backtest. Each one is fit only on
seasons before the test season, then predicts every test-season game before it is played.

Home: src/validation/baselines.py

    home_rate   the home team wins at the rate of all earlier seasons
    elo         classic Elo (win/loss only), settings tuned on earlier seasons
    rolling_xg  Phase 3 rolling 25-game xG strength -> logistic regression on earlier seasons
"""

import itertools

import numpy as np
import pandas as pd
from scipy import optimize

ELO_GRID = {
    "k": [4, 6, 8, 10, 12, 16],
    "home": [0, 20, 35, 50, 65],
    "carry": [0.5, 0.65, 0.8, 1.0],      # share of a team's rating kept over the off-season
}


def all_games(conn) -> pd.DataFrame:
    """Every regular-season game: date, season, teams, who won (OT/SO included)."""
    g = pd.read_sql("""
        SELECT id AS game_id, season, game_date, home_team_abbrev AS home, away_team_abbrev AS away,
               home_score, away_score
        FROM games WHERE game_type = 2 AND home_score IS NOT NULL
    """, conn)
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["season"] = g["season"].astype(str)
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)
    return g.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# 1. Home rate
# ---------------------------------------------------------------------------

def home_rate(games: pd.DataFrame, season: str) -> pd.DataFrame:
    """P(home) = home-win share of all games in earlier seasons."""
    past = games[games["season"] < season]
    rate = past["home_won"].mean()
    test = games[games["season"] == season][["game_id"]].copy()
    test["p_home_rate"] = rate
    return test


# ---------------------------------------------------------------------------
# 2. Elo
# ---------------------------------------------------------------------------

def run_elo(games: pd.DataFrame, k: float, home: float, carry: float) -> np.ndarray:
    """Pre-game P(home win) for every game, updating ratings after each one."""
    rating, season_of = {}, None
    p_out = np.empty(len(games))
    for i, (season, h, a, won) in enumerate(zip(games["season"], games["home"], games["away"], games["home_won"])):
        if season != season_of:                         # off-season: pull ratings toward 1500
            rating = {t: 1500 + carry * (r - 1500) for t, r in rating.items()}
            season_of = season
        rh, ra = rating.get(h, 1500.0), rating.get(a, 1500.0)
        p = 1.0 / (1.0 + 10 ** (-(rh + home - ra) / 400.0))
        p_out[i] = p
        rating[h] = rh + k * (won - p)
        rating[a] = ra - k * (won - p)
    return p_out


def elo(games: pd.DataFrame, season: str) -> tuple[pd.DataFrame, dict]:
    """Tune Elo on earlier seasons (first season = burn-in), then predict the test season."""
    seasons = sorted(games["season"].unique())
    tune_mask = (games["season"] < season) & (games["season"] > seasons[0])
    if not tune_mask.any():                              # only one earlier season: no burn-in
        tune_mask = games["season"] < season
    best, best_ll = None, np.inf
    for k, h, c in itertools.product(ELO_GRID["k"], ELO_GRID["home"], ELO_GRID["carry"]):
        p = run_elo(games[games["season"] <= season], k, h, c)
        ll = _log_loss(p[tune_mask[games["season"] <= season].to_numpy()],
                       games.loc[tune_mask, "home_won"].to_numpy())
        if ll < best_ll:
            best, best_ll = {"k": k, "home": h, "carry": c}, ll
    p = run_elo(games[games["season"] <= season], **best)
    sub = games[games["season"] <= season].assign(p_elo=p)
    out = sub[sub["season"] == season][["game_id", "p_elo"]]
    return out, {**best, "tuning_log_loss": best_ll}


# ---------------------------------------------------------------------------
# 3. Rolling xG (Phase 3) -> logistic regression
# ---------------------------------------------------------------------------

def rolling_xg_features(conn) -> pd.DataFrame:
    """Per game: home minus away 25-game xG differential per 60, going into the game."""
    from src.processing.team_strength import team_game_xg, rolling_strength
    rs = rolling_strength(team_game_xg(conn))
    rs["net60"] = rs["xgf60"] - rs["xga60"]
    h = rs[rs["is_home"] == 1][["game_id", "team", "net60", "games_in_window"]]
    a = rs[rs["is_home"] == 0][["game_id", "team", "net60", "games_in_window"]]
    f = h.merge(a, on="game_id", suffixes=("_home", "_away"))
    f["xg_diff"] = (f["net60_home"].fillna(0) - f["net60_away"].fillna(0))
    f["min_games"] = np.minimum(f["games_in_window_home"], f["games_in_window_away"])
    return f[["game_id", "xg_diff", "min_games"]]


def _fit_logistic(x, y):
    """Maximum-likelihood logistic regression p = 1 / (1 + exp(-(b0 + b1 x)))."""
    def nll(b):
        z = b[0] + b[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    return optimize.minimize(nll, x0=[0.0, 0.0], method="BFGS").x


def rolling_xg(conn, games: pd.DataFrame, season: str, min_games: int = 10) -> tuple[pd.DataFrame, dict]:
    """Fit on earlier-season games where both teams had min_games of history; predict the test season."""
    f = games[["game_id", "season", "home_won"]].merge(rolling_xg_features(conn), on="game_id")
    train = f[(f["season"] < season) & (f["min_games"] >= min_games)]
    b0, b1 = _fit_logistic(train["xg_diff"].to_numpy(), train["home_won"].to_numpy())
    test = f[f["season"] == season].copy()
    test["p_rolling_xg"] = 1.0 / (1.0 + np.exp(-(b0 + b1 * test["xg_diff"])))
    return test[["game_id", "p_rolling_xg"]], {"intercept": float(b0), "slope": float(b1),
                                                "train_games": int(len(train))}


def all_baselines(conn, season: str) -> tuple[pd.DataFrame, dict]:
    """Every baseline's P(home win) for every game of `season`, plus their fitted settings."""
    games = all_games(conn)
    hr = home_rate(games, season)
    el, elo_info = elo(games, season)
    rx, rx_info = rolling_xg(conn, games, season)
    test = games[games["season"] == season][["game_id", "game_date", "season", "home", "away", "home_won"]]
    out = test.merge(hr, on="game_id").merge(el, on="game_id").merge(rx, on="game_id", how="left")
    out["p_rolling_xg"] = out["p_rolling_xg"].fillna(out["p_home_rate"])     # no xG history: home rate
    info = {"home_rate": float(hr["p_home_rate"].iloc[0]), "elo": elo_info, "rolling_xg": rx_info}
    return out, info