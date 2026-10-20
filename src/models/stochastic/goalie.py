"""
goalie.py
=========
Bayesian goalie rating: how many goals a goalie allows per expected goal.

Home: src/models/stochastic/goalie.py
Input comes from src/processing/goalie.py (goalie_game_xg).

Model (Gamma-Poisson, conjugate):
    goals allowed   GA ~ Poisson(theta * xGA)
    prior           theta ~ Gamma(k, k)          mean 1.0 = average goalie
    posterior       theta ~ Gamma(k + GA, k + xGA)

theta < 1 is better than average: 0.90 means 10% fewer goals than an average
goalie would allow on the same shots. k is the prior's strength, in expected
goals: each goalie starts as if he had already faced k xG with average results.

Season adjustment: league-wide GA / xGA drifts between seasons (about 1.03 in
2023-24 and 2024-25, 0.98 in 2025-26), which would make every goalie look
better or worse depending on the season. With season_adjust=True, each season's
xGA is scaled so league GA / xGA = 1.0 within that season, using only games
before the rating date, so a goalie is compared with his own season's league.
"""

import pandas as pd
from scipy import stats

DEFAULT_K = 800.0           # prior strength, in expected goals faced
DEFAULT_LOOKBACK_DAYS = 730  # about two seasons


def season_ratios(gg: pd.DataFrame) -> pd.Series:
    """League goals allowed per expected goal, by season (GA / xGA)."""
    totals = gg.groupby("season")[["ga", "xga"]].sum()
    return totals["ga"] / totals["xga"]


def goalie_rating_as_of(gg: pd.DataFrame, date, k: float = DEFAULT_K,
                        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                        season_adjust: bool = True) -> pd.DataFrame:
    """Every goalie's rating going into `date`: one row per goalie.

    Uses games strictly before `date` and no older than `lookback_days`.
    Goalies with no games in that window are left out. With season_adjust=True,
    xGA is rescaled per season to that season's league average (see module notes).

    Columns: goalie_id, goalie_name, team, games, shots, xga, ga, gsax,
             theta_mean, theta_var, theta_lo, theta_hi, last_game
    theta_lo / theta_hi bound a 90% credible interval.
    """
    # 1. Games in the window: before the date, within the lookback
    cutoff = pd.Timestamp(date)
    start = cutoff - pd.Timedelta(days=lookback_days)
    game_dates = pd.to_datetime(gg["game_date"])
    window = gg[(game_dates < cutoff) & (game_dates >= start)].sort_values("game_date")

    # 1b. Optional: rescale each season's xGA to that season's league average.
    #     Ratios come from games before the date only, so nothing leaks.
    if season_adjust and len(window):
        window = window.assign(xga=window["xga"] * window["season"].map(season_ratios(window)))

    # 2. Totals per goalie (team = his most recent team)
    out = (window.groupby("goalie_id")
                 .agg(goalie_name=("goalie_name", "last"),
                      team=("team", "last"),
                      games=("game_id", "nunique"),
                      shots=("shots", "sum"),
                      xga=("xga", "sum"),
                      ga=("ga", "sum"),
                      last_game=("game_date", "max"))
                 .reset_index())

    # 3. Posterior Gamma(a, b): a = k + goals allowed, b = k + expected goals
    a = k + out["ga"]
    b = k + out["xga"]

    # 4. Posterior mean and variance
    out["theta_mean"] = a / b
    out["theta_var"] = a / b ** 2

    # 5. 90% credible interval (SciPy's gamma takes scale = 1 / rate)
    out["theta_lo"] = stats.gamma.ppf(0.05, a, scale=1 / b)
    out["theta_hi"] = stats.gamma.ppf(0.95, a, scale=1 / b)

    # 6. Raw goals saved above expected, for reference
    out["gsax"] = out["xga"] - out["ga"]

    cols = ["goalie_id", "goalie_name", "team", "games", "shots", "xga", "ga", "gsax",
            "theta_mean", "theta_var", "theta_lo", "theta_hi", "last_game"]
    return out[cols].sort_values("theta_mean").reset_index(drop=True)