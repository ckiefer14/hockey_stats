"""
explain.py
==========
Why the model gives the home team the probability it does: a Shapley breakdown of P(home win)
into five parts.

Home: src/models/explain.py

The game model's scoring rates are a sum of separate pieces on the log scale:
    log lam_home = a0 + home_ice + (off_home - def_away) - goalie_away + rest_home + lineup_home
    log lam_away = a0            + (off_away - def_home) - goalie_home + rest_away + lineup_away
Each piece is 0 when neutral (average team, average goalie, normal rest, usual lineup), so
switching a factor "off" means setting its pieces to 0 for both teams. The base is a game between
equal teams with average goalies on neutral ice (close to 50%; the small rest is the late-game
model's own home edge in overtime / shootouts).

Factors (both teams' pieces together, so each number says which side the factor favours):
    home_ice   alpha_home
    ratings    skater ratings (offense and defense, both teams; includes finishing)
    goalies    starting goalies (Phase 3 rating)
    rest       rest, back-to-backs, travel, 3-in-4, goalie on zero rest
    lineup     tonight's lineups vs each team's usual one (v2.2)

Shapley value of a factor = its average effect on P(home win) over every order in which the five
could be switched on; the five values add up exactly to P(home win) - base. P(home win) is the
posterior mean of the exact per-draw probability, as in predict_day.
"""

from itertools import combinations
from math import factorial

import numpy as np

from src.models.game_model import outcome_probs
from src.models.stochastic.hierarchical_tv import context_shift

FACTORS = ["home_ice", "ratings", "goalies", "rest", "lineup"]
LABELS = {"home_ice": "Home ice", "ratings": "Team ratings", "goalies": "Goalies", "rest": "Rest & travel",
          "lineup": "Lineups"}
LIN_COLS = ("lin_home", "lin_away")


def _pieces(draws: dict, teams: list[str], home: str, away: str, gamma_home: float, gamma_away: float,
            ctx: dict) -> dict:
    """The log-rate pieces of each factor, (home, away), per draw where they vary by draw."""
    ti = {t: i for i, t in enumerate(teams)}
    h, a = ti[home], ti[away]
    n = len(draws["alpha0"])
    zero = np.zeros(n)
    rest_ctx = {k: [v] for k, v in ctx.items()}
    for c in LIN_COLS:
        if c in rest_ctx:
            rest_ctx[c] = [0.0]
    lin_ctx = {k: [0.0] for k in ctx}
    for c in LIN_COLS:
        lin_ctx[c] = [float(ctx.get(c, 0.0))]
    rh, ra = context_shift(draws, rest_ctx)
    lh, la = context_shift(draws, lin_ctx) if any(ctx.get(c) for c in LIN_COLS) else ([0.0], [0.0])
    return {
        "home_ice": (np.asarray(draws["alpha_home"], float), zero),
        "ratings": (draws["mu_off"][:, h] - draws["mu_def"][:, a], draws["mu_off"][:, a] - draws["mu_def"][:, h]),
        "goalies": (zero - gamma_away, zero - gamma_home),
        "rest": (zero + float(rh[0]), zero + float(ra[0])),
        "lineup": (zero + float(lh[0]), zero + float(la[0])),
    }


def win_prob_breakdown(draws: dict, teams: list[str], late: dict, home: str, away: str,
                       gamma_home: float, gamma_away: float, ctx: dict, max_draws: int | None = None,
                       seed: int = 0) -> dict:
    """{'base': P with every factor off, 'total': P with all on, 'home_ice': ..., 'ratings': ...}
    (the factor values are changes in P(home win), probability units; they sum to total - base).
    gamma_home / gamma_away: the starters' raw ratings (not the *_eff values, which already include rest).
    ctx: the game's context columns as logged by predict_day (rest, travel, g2b, lin_*)."""
    n = len(draws["alpha0"])
    idx = np.arange(n)
    if max_draws and n > max_draws:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_draws, replace=False))
    d = {k: (np.asarray(v)[idx] if hasattr(v, "__len__") and np.ndim(v) >= 1 and len(v) == n else v)
         for k, v in draws.items()}
    pcs = _pieces(d, teams, home, away, gamma_home, gamma_away, ctx)
    a0, rho = np.asarray(d["alpha0"], float), np.asarray(d["rho"], float)

    cache = {}

    def value(on: frozenset) -> float:
        if on not in cache:
            lh = a0 + sum(pcs[f][0] for f in on)
            la = a0 + sum(pcs[f][1] for f in on)
            cache[on] = float(np.mean(outcome_probs(np.exp(lh), np.exp(la), rho, late)["home_win"]))
        return cache[on]

    k = len(FACTORS)
    out = {"base": value(frozenset()), "total": value(frozenset(FACTORS))}
    for f in FACTORS:
        others = [g for g in FACTORS if g != f]
        phi = 0.0
        for r in range(k):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for S in combinations(others, r):
                S = frozenset(S)
                phi += w * (value(S | {f}) - value(S))
        out[f] = phi
    return out
