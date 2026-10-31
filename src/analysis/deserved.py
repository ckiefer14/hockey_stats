"""Deserve-to-win: replay a finished game's shots many times.

Each unblocked shot (goalie in net, periods 1-4, from ``shot_xg``) becomes a
goal with probability equal to its xG. A replay goes to the team with more
goals. A tied replay goes to overtime, where the home team wins with
probability equal to its share of the game's xG, the same rule the Phase 6
game model uses for who scores in OT (lambda_h / (lambda_h + lambda_a)).

Empty-net and shootout attempts are not in ``shot_xg`` and are not replayed, so
the result answers "who earned the win with the goalies in net".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def deserve_to_win(shots: pd.DataFrame, home: str, away: str,
                   n_sims: int = 10_000, seed: int = 42,
                   chunk: int = 2_000) -> dict:
    """Replay the shots ``n_sims`` times.

    ``shots`` needs columns ``team`` and ``xg``.

    Returns P(home win), P(tied after replayed shots), the xG totals, and the
    replayed goal distributions (``home_goals`` / ``away_goals`` arrays).
    """
    rng = np.random.default_rng(seed)
    xh = shots.loc[shots["team"] == home, "xg"].to_numpy(float)
    xa = shots.loc[shots["team"] == away, "xg"].to_numpy(float)

    gh = np.empty(n_sims, dtype=np.int32)
    ga = np.empty(n_sims, dtype=np.int32)
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        gh[start:start + m] = (rng.random((m, xh.size)) < xh).sum(axis=1)
        ga[start:start + m] = (rng.random((m, xa.size)) < xa).sum(axis=1)

    tot_h, tot_a = xh.sum(), xa.sum()
    share_h = tot_h / (tot_h + tot_a) if tot_h + tot_a > 0 else 0.5
    tied = gh == ga
    home_wins = (gh > ga) | (tied & (rng.random(n_sims) < share_h))

    return {
        "p_home": float(home_wins.mean()),
        "p_tied": float(tied.mean()),
        "xg_home": float(tot_h),
        "xg_away": float(tot_a),
        "home_goals": gh,
        "away_goals": ga,
        "n_sims": n_sims,
    }