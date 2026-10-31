"""
toy_mcmc.py
===========
Track A: one team pair, Poisson goals, Adaptive Metropolis.

Home: src/models/stochastic/toy_mcmc.py
"""

import numpy as np
import pandas as pd

PARAM_NAMES = ["d_A", "d_B"]
PRIOR_LO, PRIOR_HI = -1.0, 1.0      # uniform prior on log-rate differences
def league_baseline(tg: pd.DataFrame) -> tuple[float, float]:
    """League scoring level and home-ice edge, both on the log scale.

    Uses gf_xg (periods 1-4, empty-net goals excluded).
    Returns (alpha0, eta):
        alpha0  = log(away goals per game)
        eta     = log(home goals per game / away goals per game)
    """
    # 1. Split home and away rows (each game appears once in each)
    home = tg[tg["is_home"] == 1]
    away = tg[tg["is_home"] == 0]

    # 2. Mean goals per game, empty-net goals excluded
    home_rate = home["gf_xg"].mean()
    away_rate = away["gf_xg"].mean()

    # 3. League level: away teams get alpha0 alone
    alpha0 = np.log(away_rate)

    # 4. Home edge: home teams get alpha0 + eta
    eta = np.log(home_rate / away_rate)

    # 5. Plain floats, not numpy types
    return float(alpha0), float(eta)

def pair_games(tg: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    """Head-to-head games for one pair, one row per game from team_a's side.

    Columns: game_id, game_date, season, a_home, goals_a, goals_b
    """

    # 1. Rows where team_a played team_b (team_a's side only, so no double count)
    rows = tg[(tg["team"] == team_a) & (tg["opponent"] == team_b)]

    # 2. Keep the needed columns, then rename to the model's names
    df = rows[["game_id","game_date","season","is_home","gf_xg","ga_xg"]]
    df = df.rename(columns={
        "is_home": "a_home",
        "gf_xg": "goals_a",
        "ga_xg": "goals_b",
    })

    # 3. Chronological order, clean 0..n-1 index
    df["a_home"] = df["a_home"].astype(int)
    return df.sort_values("game_date").reset_index(drop=True)

def make_log_posterior(games: pd.DataFrame, alpha0: float, eta: float):
    """Build log p(d_A, d_B | games) up to a constant.

    Prior:      uniform on [PRIOR_LO, PRIOR_HI]^2 (log-rate differences)
    Likelihood: goals_a ~ Poisson(exp(alpha0 + eta*a_home     + d_A))
                goals_b ~ Poisson(exp(alpha0 + eta*(1-a_home) + d_B))

    Returns a function log_post(x) with x = [d_A, d_B].
    """
    # 1. Convert to numpy once, outside log_post, so each MCMC step is fast
    a_home  = games["a_home"].to_numpy(dtype=float)
    goals_a = games["goals_a"].to_numpy(dtype=float)
    goals_b = games["goals_b"].to_numpy(dtype=float)

    # 2. Fixed part of each log-rate (league level + home ice)
    base_a = alpha0 + eta * a_home
    base_b = alpha0 + eta * (1-a_home)

    def log_post(x: np.ndarray) -> float:
        # 1. Prior: zero probability outside the box
        if np.any(x < PRIOR_LO) or np.any(x > PRIOR_HI):
            return -np.inf

        # 2. Log-rates for every game
        log_lam_a = base_a + x[0]
        log_lam_b = base_b + x[1]

        # 3. Poisson log-likelihood, dropping log(y!) (constant in x)
        ll_a = np.sum(goals_a * log_lam_a - np.exp(log_lam_a))
        ll_b = np.sum(goals_b * log_lam_b - np.exp(log_lam_b))
        return float (ll_a + ll_b)

    return log_post


def simulate_pair_games(
        n_games: int,
        d_A: float,
        d_B: float,
        alpha0: float,
        eta: float,
        seed: int = 0,
) -> pd.DataFrame:
    """Fake head-to-head games with known d_A, d_B (for recovery tests).

    Same columns as pair_games, so the model can't tell real from simulated.
    """
    rng = np.random.default_rng(seed)

    # 1. Alternate home ice: A home in games 0, 2, 4, ...
    a_home = (np.arange(n_games) % 2 == 0).astype(int)

    # 2. True rates from the same formulas the likelihood uses
    lam_a = np.exp(alpha0 + eta * a_home + d_A)
    lam_b = np.exp(alpha0 + eta * (1 - a_home) + d_B)

    # 3. Poisson goal counts, one draw per game per team
    goals_a = rng.poisson(lam_a)
    goals_b = rng.poisson(lam_b)

    # 4. Same columns as pair_games (ids and dates are placeholders)
    return pd.DataFrame({
        "game_id": np.arange(n_games),
        "game_date": pd.date_range("2000-01-01", periods=n_games, freq="7D"),
        "season": 0,
        "a_home": a_home,
        "goals_a": goals_a,
        "goals_b":  goals_b,
    })