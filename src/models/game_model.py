"""
game_model.py
=============
Exact game outcome probabilities from Track B posterior draws.

Home: src/models/game_model.py

For each posterior draw:
    1. Regulation  lam_home, lam_away from the ratings, home ice and tonight's goalies;
                   a (K+1) x (K+1) table of regulation scores from the tie-inflated
                   model (rho), the same model Track B was fit with.
    2. Overtime    if tied: P(OT goal) = 1 - exp(-ot_factor * (lam_home + lam_away) * 5/60),
                   home scores it with probability lam_home / (lam_home + lam_away).
    3. Shootout    if still tied: home wins with probability so_home (0.5).
    4. Empty net   final scores only: the regulation leader adds 0/1/2 empty-net goals
                   with the measured probabilities for its margin (never changes the winner).

Doing this per draw gives one P(home win) per draw. Their mean is the prediction;
their spread is the uncertainty that comes from not knowing the ratings exactly.
Game-to-game randomness is inside each P(win) already.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.models.stochastic.hierarchical import K_MAX
from src.processing.late_game import OT_MINUTES

TEAM_TERMS = ["mu_off", "mu_def"]
LEAGUE_TERMS = ["alpha0", "alpha_home", "rho"]


# ---------------------------------------------------------------------------
# Inputs: posterior draws and tonight's goalies
# ---------------------------------------------------------------------------

def load_track_b(path) -> tuple[dict, list[str]]:
    """Posterior draws of a saved Track B fit, pooled over chains, file closed.

    Returns (draws, teams): draws[name] is (n,) for league terms and (n, n_teams)
    for team terms; teams[i] is the abbreviation for index i.
    """
    import arviz as az
    idata = az.from_netcdf(path)
    idata = idata.load()
    if hasattr(idata, "close"):
        idata.close()
    post = idata.posterior
    draws = {}
    for name in LEAGUE_TERMS + TEAM_TERMS:
        x = post[name].values
        draws[name] = x.reshape(-1, *x.shape[2:])
    teams = str(post.attrs["teams"]).split(",")
    return draws, teams


def default_starter(conn, team: str, date, last_n: int = 10) -> int | None:
    """The goalie who started most of `team`'s last `last_n` games before `date`."""
    recent = pd.read_sql("""
        SELECT  g.player_id AS goalie_id, s.game_date
        FROM    goalie_game_stats g
        JOIN    schedule s ON s.game_id = g.game_id
        WHERE   g.starter = 1 AND g.team_abbreviation = ? AND s.game_date < ?
        ORDER BY s.game_date DESC
        LIMIT   ?
    """, conn, params=(team, str(pd.Timestamp(date).date()), last_n))
    if recent.empty:
        return None
    return int(recent["goalie_id"].mode().iloc[0])


def goalie_offset(gg: pd.DataFrame, goalie_id, date) -> float:
    """Fixed offset gamma = -log(theta) for one goalie, Phase 3 rating as of `date`.

    0.0 (league average) if the goalie is unknown or has no history.
    """
    if goalie_id is None:
        return 0.0
    from src.models.stochastic.goalie import goalie_rating_as_of
    r = goalie_rating_as_of(gg, date)
    row = r[r["goalie_id"] == goalie_id]
    return float(-np.log(row["theta_mean"].iloc[0])) if len(row) else 0.0


# ---------------------------------------------------------------------------
# Rates and exact probabilities
# ---------------------------------------------------------------------------

def matchup_rates(draws: dict, teams: list[str], home: str, away: str,
                  gamma_home: float = 0.0, gamma_away: float = 0.0,
                  neutral: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Regulation scoring rates per draw: (lam_home, lam_away), each shape (n_draws,).

    gamma_home / gamma_away: fixed offsets of the home / away starting goalies.
    neutral=True drops home ice.
    """
    h, a = teams.index(home), teams.index(away)
    home_ice = 0.0 if neutral else draws["alpha_home"]
    lam_home = np.exp(draws["alpha0"] + home_ice + draws["mu_off"][:, h] - draws["mu_def"][:, a] - gamma_away)
    lam_away = np.exp(draws["alpha0"] + draws["mu_off"][:, a] - draws["mu_def"][:, h] - gamma_home)
    return lam_home, lam_away


def regulation_table(lam_home, lam_away, rho, k_max: int = K_MAX) -> np.ndarray:
    """Tie-inflated regulation score probabilities, shape (n, K+1, K+1): [draw, home, away]."""
    lam_home, lam_away, rho = (np.atleast_1d(np.asarray(x, float)) for x in (lam_home, lam_away, rho))
    k = np.arange(k_max + 1)
    ph = stats.poisson.pmf(k[None, :], lam_home[:, None])            # (n, K+1)
    pa = stats.poisson.pmf(k[None, :], lam_away[:, None])
    table = ph[:, :, None] * pa[:, None, :]                            # independent Poisson
    T = np.einsum("nkk->n", table)                                     # P(tie) under plain Poisson
    table = table + rho[:, None, None] * np.eye(k_max + 1)[None] * table
    return table / (1.0 + rho * T)[:, None, None]


def overtime(lam_home, lam_away, late: dict) -> tuple[np.ndarray, np.ndarray]:
    """P(OT goal) and P(home wins OT) per draw (not yet conditioned on a tie)."""
    k = late["ot"]["ot_factor"]
    p_goal = 1.0 - np.exp(-k * (lam_home + lam_away) * OT_MINUTES / 60.0)
    p_home_first = lam_home / (lam_home + lam_away)
    return p_goal, p_goal * p_home_first


def outcome_probs(lam_home, lam_away, rho, late: dict) -> dict:
    """Exact outcome probabilities per draw. Every value is an array of shape (n_draws,).

    Keys: home_win, away_win, reg_home, reg_away, tie_reg, ot_home, ot_away, so_home, so_away
    (the last six are probabilities of the whole game ending that way).
    """
    table = regulation_table(lam_home, lam_away, rho)
    reg_home = np.tril(np.ones(table.shape[1:]), -1)[None] * table     # home > away
    reg_home = reg_home.sum(axis=(1, 2))
    tie = np.einsum("nkk->n", table)
    reg_away = table.sum(axis=(1, 2)) - reg_home - tie                # sums to 1 up to truncation

    p_ot_goal, p_ot_home = overtime(lam_home, lam_away, late)
    so_home = late["shootout"]["use_home_prob"]
    out = {
        "reg_home": reg_home,
        "reg_away": reg_away,
        "tie_reg": tie,
        "ot_home": tie * p_ot_home,
        "ot_away": tie * (p_ot_goal - p_ot_home),
        "so_home": tie * (1.0 - p_ot_goal) * so_home,
        "so_away": tie * (1.0 - p_ot_goal) * (1.0 - so_home),
    }
    out["home_win"] = out["reg_home"] + out["ot_home"] + out["so_home"]
    out["away_win"] = out["reg_away"] + out["ot_away"] + out["so_away"]
    return out


def final_score_table(lam_home, lam_away, rho, late: dict, empty_net: bool = True) -> np.ndarray:
    """Predictive distribution of the FINAL score, averaged over draws: [home, away].

    Includes empty-net goals (if empty_net) and the +1 the NHL credits the OT or
    shootout winner. Size (K+4, K+4).
    """
    table = regulation_table(lam_home, lam_away, rho)                  # (n, K+1, K+1)
    n, size = table.shape[0], table.shape[1]
    final = np.zeros((n, size + 3, size + 3))
    h_idx, a_idx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    # Decided in regulation: leader may add empty-net goals
    en = late["empty_net"]["by_margin"]
    for h in range(size):
        for a in range(size):
            if h == a:
                continue
            p = table[:, h, a]
            margin = str(min(abs(h - a), 3))
            probs = en[margin]["probs"] if empty_net else [1.0, 0.0, 0.0]
            for j, pj in enumerate(probs):
                if pj == 0:
                    continue
                if h > a:
                    final[:, h + j, a] += p * pj
                else:
                    final[:, h, a + j] += p * pj

    # Tied: OT or shootout winner gets +1
    p_ot_goal, p_ot_home = overtime(lam_home, lam_away, late)
    so_home = late["shootout"]["use_home_prob"]
    p_home_wins_tie = p_ot_home + (1.0 - p_ot_goal) * so_home
    for k in range(size):
        p = table[:, k, k]
        final[:, k + 1, k] += p * p_home_wins_tie
        final[:, k, k + 1] += p * (1.0 - p_home_wins_tie)
    return final.mean(axis=0)


# ---------------------------------------------------------------------------
# One call for a matchup
# ---------------------------------------------------------------------------

def predict_matchup(draws: dict, teams: list[str], late: dict, home: str, away: str,
                    gamma_home: float = 0.0, gamma_away: float = 0.0,
                    level: float = 0.90) -> dict:
    """Everything Phase 6 reports for one game, from exact per-draw probabilities.

    Returns a dict with:
        probs      per-draw outcome probabilities (arrays)
        summary    mean and `level` interval of the main probabilities
        final      final-score table (with empty-net goals), and its
                   total-goals and margin distributions
        rates      mean expected regulation goals for each side
    """
    lam_home, lam_away = matchup_rates(draws, teams, home, away, gamma_home, gamma_away)
    probs = outcome_probs(lam_home, lam_away, draws["rho"], late)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2

    summary = {}
    for key in ["home_win", "reg_home", "reg_away", "tie_reg", "ot_home", "ot_away", "so_home", "so_away"]:
        x = probs[key]
        summary[key] = {"mean": float(x.mean()), "lo": float(np.quantile(x, lo_q)), "hi": float(np.quantile(x, hi_q))}

    final = final_score_table(lam_home, lam_away, draws["rho"], late)
    size = final.shape[0]
    totals = np.bincount((np.add.outer(np.arange(size), np.arange(size))).ravel(), weights=final.ravel())
    margins_idx = np.subtract.outer(np.arange(size), np.arange(size)).ravel()
    margin_vals = np.arange(-(size - 1), size)
    margins = np.array([final.ravel()[margins_idx == m].sum() for m in margin_vals])

    return {
        "home": home, "away": away,
        "gamma_home": gamma_home, "gamma_away": gamma_away,
        "probs": probs,
        "summary": summary,
        "rates": {"home": float(lam_home.mean()), "away": float(lam_away.mean())},
        "final": {"table": final, "totals": totals,
                  "margins": pd.Series(margins, index=margin_vals)},
    }