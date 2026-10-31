"""
montecarlo.py
=============
Simulates full games from Track B posterior draws: regulation (tie-inflated),
overtime, shootout and late empty-net goals. One row = one simulated game.

Home: src/models/stochastic/montecarlo.py

game_model.py gives the same outcome probabilities exactly; this module draws actual
games. It is needed when outcomes must be combined across games (standings, playoff
odds, parlays), and it cross-checks game_model.

Parameter uncertainty: each simulated game uses one posterior draw (`draw_idx`).
For a season simulation, give every game in the same simulated season the SAME
draw index, so a team that is strong in one draw is strong all season.
"""

import numpy as np
import pandas as pd

from src.models.stochastic.hierarchical import sample_scores
from src.processing.late_game import OT_MINUTES

DECIDED = np.array(["REG", "OT", "SO"])


def simulate_games(draws: dict, late: dict, home_idx, away_idx,
                   gamma_home=0.0, gamma_away=0.0, draw_idx=None,
                   rng=None, empty_net: bool = True) -> pd.DataFrame:
    """Simulate games. All array arguments have one entry per simulated game.

    draws:      Track B posterior draws (game_model.load_track_b)
    late:       late-game parameters (late_game.load_params)
    home_idx, away_idx:  team indices
    gamma_home, gamma_away:  fixed goalie offsets of the home / away starters
    draw_idx:   which posterior draw each game uses (default: random)

    Columns: draw, reg_home, reg_away, decided ("REG"/"OT"/"SO"), home_win,
             en_home, en_away (empty-net goals), final_home, final_away
             (official final score: empty-net goals and the +1 for OT/SO winner)
    """
    rng = np.random.default_rng() if rng is None else rng
    home_idx = np.asarray(home_idx)
    away_idx = np.asarray(away_idx)
    n = home_idx.size
    gamma_home = np.broadcast_to(np.asarray(gamma_home, float), (n,))
    gamma_away = np.broadcast_to(np.asarray(gamma_away, float), (n,))
    n_draws = draws["alpha0"].size
    if draw_idx is None:
        draw_idx = rng.integers(0, n_draws, n)
    draw_idx = np.asarray(draw_idx)

    # 1. Rates for each simulated game, from its posterior draw
    a0, ah, rho = draws["alpha0"][draw_idx], draws["alpha_home"][draw_idx], draws["rho"][draw_idx]
    off, dfn = draws["mu_off"], draws["mu_def"]
    lam_home = np.exp(a0 + ah + off[draw_idx, home_idx] - dfn[draw_idx, away_idx] - gamma_away)
    lam_away = np.exp(a0 + off[draw_idx, away_idx] - dfn[draw_idx, home_idx] - gamma_home)

    # 2. Regulation: tie-inflated scores (same model Track B was fit with)
    reg_home, reg_away = sample_scores(lam_home, lam_away, rho, rng)
    tied = reg_home == reg_away

    # 3. Overtime: sudden death at ot_factor x the regulation rates for 5 minutes
    k = late["ot"]["ot_factor"]
    p_ot_goal = 1.0 - np.exp(-k * (lam_home + lam_away) * OT_MINUTES / 60.0)
    ot_goal = tied & (rng.random(n) < p_ot_goal)
    ot_home = ot_goal & (rng.random(n) < lam_home / (lam_home + lam_away))

    # 4. Shootout for games still tied
    shootout = tied & ~ot_goal
    so_home = shootout & (rng.random(n) < late["shootout"]["use_home_prob"])

    home_win = (reg_home > reg_away) | ot_home | so_home
    decided = np.where(~tied, 0, np.where(ot_goal, 1, 2))

    # 5. Empty-net goals for the regulation leader (final score only)
    en_home = np.zeros(n, dtype=int)
    en_away = np.zeros(n, dtype=int)
    if empty_net:
        margin = np.minimum(np.abs(reg_home - reg_away), 3)
        en_goals = np.zeros(n, dtype=int)
        u = rng.random(n)
        for m, row in late["empty_net"]["by_margin"].items():
            sel = margin == int(m)
            cum = np.cumsum(row["probs"])
            en_goals[sel] = np.searchsorted(cum, u[sel], side="right").clip(max=len(cum) - 1)
        en_home = np.where(reg_home > reg_away, en_goals, 0)
        en_away = np.where(reg_away > reg_home, en_goals, 0)

    # 6. Official final score: regulation + empty-net + 1 for the OT/SO winner
    extra_home = (ot_home | so_home).astype(int)
    extra_away = (tied & ~(ot_home | so_home)).astype(int)
    return pd.DataFrame({
        "draw": draw_idx,
        "reg_home": reg_home, "reg_away": reg_away,
        "decided": DECIDED[decided],
        "home_win": home_win.astype(int),
        "en_home": en_home, "en_away": en_away,
        "final_home": reg_home + en_home + extra_home,
        "final_away": reg_away + en_away + extra_away,
    })


def simulate_matchup(draws: dict, teams: list[str], late: dict, home: str, away: str,
                     gamma_home: float = 0.0, gamma_away: float = 0.0,
                     n_sims: int = 100_000, seed: int = 0, empty_net: bool = True) -> pd.DataFrame:
    """Simulate one matchup n_sims times, spreading the sims evenly over posterior draws."""
    rng = np.random.default_rng(seed)
    n_draws = draws["alpha0"].size
    draw_idx = np.resize(rng.permutation(n_draws), n_sims)       # every draw used about equally
    h, a = teams.index(home), teams.index(away)
    return simulate_games(draws, late, np.full(n_sims, h), np.full(n_sims, a),
                          gamma_home, gamma_away, draw_idx, rng, empty_net)


def summarize_sims(sims: pd.DataFrame) -> dict:
    """Shares of each way a simulated game ended, plus mean final goals."""
    reg = sims["decided"] == "REG"
    home_ahead = sims["reg_home"] > sims["reg_away"]
    return {
        "n_sims": int(len(sims)),
        "home_win": float(sims["home_win"].mean()),
        "reg_home": float((reg & home_ahead).mean()),
        "reg_away": float((reg & ~home_ahead).mean()),
        "tie_reg": float((~reg).mean()),
        "ot_home": float(((sims["decided"] == "OT") & (sims["home_win"] == 1)).mean()),
        "ot_away": float(((sims["decided"] == "OT") & (sims["home_win"] == 0)).mean()),
        "so_home": float(((sims["decided"] == "SO") & (sims["home_win"] == 1)).mean()),
        "so_away": float(((sims["decided"] == "SO") & (sims["home_win"] == 0)).mean()),
        "mean_final_total": float((sims["final_home"] + sims["final_away"]).mean()),
        "en_share_of_goals": float((sims["en_home"] + sims["en_away"]).sum()
                                   / (sims["reg_home"] + sims["reg_away"] + sims["en_home"] + sims["en_away"]).sum()),
    }


# ---------------------------------------------------------------------------
# Score-state simulator: regulation played in time slices
# ---------------------------------------------------------------------------

REG_SECONDS = 3 * 1200
STATE_ORDER = ["trailing 2+", "trailing 1", "tied", "tied late", "leading 1", "leading 2+"]


def _state_factor(diff, t, ss):
    """Normal-play scoring factor for a team `diff` goals ahead at second t (arrays)."""
    f = ss["state_factor"]
    tied = np.where(t >= REG_SECONDS - ss["tied_late_seconds"], f["tied late"], f["tied"])
    return np.select([diff <= -2, diff == -1, diff == 0, diff == 1],
                     [f["trailing 2+"], f["trailing 1"], tied, f["leading 1"]], f["leading 2+"])


def simulate_regulation_score_state(lam_home, lam_away, ss: dict, rng, scale: float | None = None,
                                    dt: float = 10.0) -> dict:
    """Regulation in dt-second slices with score effects and pulled goalies.

    lam_home, lam_away: Track B regulation rates (goals per 60 min), one per simulated game.
    ss: score_state.json (state factors, pull hazard, 6-on-5 factor, empty-net rates, scale).
    Returns reg_home, reg_away (goals with a goalie in net) and en_home, en_away (empty-net goals).
    """
    scale = ss.get("scale", 1.0) if scale is None else scale
    n = lam_home.size
    rate = {1: lam_home * scale / REG_SECONDS, 0: lam_away * scale / REG_SECONDS}     # goals per second
    goals = {1: np.zeros(n, int), 0: np.zeros(n, int)}
    en = {1: np.zeros(n, int), 0: np.zeros(n, int)}
    pulled = {1: np.zeros(n, bool), 0: np.zeros(n, bool)}

    window, bin_s = ss["window_seconds"], ss["bin_seconds"]
    n_bins = window // bin_s
    hazard = np.array([ss["pull_hazard"][str(m)] for m in (1, 2, 3)])            # (3, n_bins), index 0 = last bin
    six = np.array([ss["pulled"][str(m)]["trailer_factor"] for m in (1, 2, 3)])
    en_rate = np.array([ss["pulled"][str(m)]["leader_en_per_min"] for m in (1, 2, 3)]) / 60.0

    for t in np.arange(0.0, REG_SECONDS, dt):
        score_diff = (goals[1] + en[1]) - (goals[0] + en[0])                     # home - away, real score
        in_window = t >= REG_SECONDS - window
        b = int(min(n_bins - 1, (REG_SECONDS - t) // bin_s)) if in_window else 0

        # 1. Trailing teams may pull their goalie in the last minutes
        if in_window:
            for team, d in ((1, score_diff), (0, -score_diff)):
                trailing = (d < 0) & ~pulled[team]
                m = np.clip(-d, 1, 3) - 1
                p_pull = 1.0 - np.exp(-hazard[m, b] * dt)
                pulled[team] |= trailing & (rng.random(n) < p_pull)

        # 2. Scoring rates this slice
        p_goal = {}
        for team, d in ((1, score_diff), (0, -score_diff)):
            opp = 1 - team
            m_own = np.clip(-d, 1, 3) - 1          # margin index if this team trails
            m_lead = np.clip(d, 1, 3) - 1          # margin index if this team leads
            r = np.where(pulled[team], rate[team] * six[m_own],                        # 6-on-5
                np.where(pulled[opp], en_rate[m_lead],                                  # empty net
                         rate[team] * _state_factor(d, t, ss)))                         # normal play
            p_goal[team] = 1.0 - np.exp(-r * dt)

        # 3. Goals (both can score in the same slice; very rare at 10 s)
        scored = {team: rng.random(n) < p_goal[team] for team in (1, 0)}
        for team in (1, 0):
            empty = pulled[1 - team] & scored[team]
            en[team] += empty
            goals[team] += scored[team] & ~empty
        any_goal = scored[1] | scored[0]
        pulled[1] &= ~any_goal                     # goalies back in after a goal
        pulled[0] &= ~any_goal

    return {"reg_home": goals[1], "reg_away": goals[0], "en_home": en[1], "en_away": en[0]}


def simulate_games_score_state(draws: dict, late: dict, ss: dict, home_idx, away_idx,
                               gamma_home=0.0, gamma_away=0.0, draw_idx=None, rng=None,
                               scale: float | None = None, dt: float = 10.0) -> pd.DataFrame:
    """Like simulate_games, but regulation comes from the score-state simulator.

    No tie inflation: extra ties come from score effects ("tied late" play and
    pulled-goalie equalizers). Overtime and shootout as in simulate_games.
    Same columns as simulate_games.
    """
    rng = np.random.default_rng() if rng is None else rng
    home_idx, away_idx = np.asarray(home_idx), np.asarray(away_idx)
    n = home_idx.size
    gamma_home = np.broadcast_to(np.asarray(gamma_home, float), (n,))
    gamma_away = np.broadcast_to(np.asarray(gamma_away, float), (n,))
    if draw_idx is None:
        draw_idx = rng.integers(0, draws["alpha0"].size, n)
    draw_idx = np.asarray(draw_idx)

    a0, ah = draws["alpha0"][draw_idx], draws["alpha_home"][draw_idx]
    off, dfn = draws["mu_off"], draws["mu_def"]
    lam_home = np.exp(a0 + ah + off[draw_idx, home_idx] - dfn[draw_idx, away_idx] - gamma_away)
    lam_away = np.exp(a0 + off[draw_idx, away_idx] - dfn[draw_idx, home_idx] - gamma_home)

    reg = simulate_regulation_score_state(lam_home, lam_away, ss, rng, scale, dt)
    total_home = reg["reg_home"] + reg["en_home"]
    total_away = reg["reg_away"] + reg["en_away"]
    tied = total_home == total_away

    k = late["ot"]["ot_factor"]
    p_ot_goal = 1.0 - np.exp(-k * (lam_home + lam_away) * OT_MINUTES / 60.0)
    ot_goal = tied & (rng.random(n) < p_ot_goal)
    ot_home = ot_goal & (rng.random(n) < lam_home / (lam_home + lam_away))
    shootout = tied & ~ot_goal
    so_home = shootout & (rng.random(n) < late["shootout"]["use_home_prob"])

    home_win = (total_home > total_away) | ot_home | so_home
    decided = np.where(~tied, 0, np.where(ot_goal, 1, 2))
    return pd.DataFrame({
        "draw": draw_idx,
        "reg_home": reg["reg_home"], "reg_away": reg["reg_away"],
        "decided": DECIDED[decided],
        "home_win": home_win.astype(int),
        "en_home": reg["en_home"], "en_away": reg["en_away"],
        "final_home": total_home + (ot_home | so_home),
        "final_away": total_away + (tied & ~(ot_home | so_home)),
    })


def calibrate_scale(draws: dict, late: dict, ss: dict, gt: pd.DataFrame,
                    reps: int = 10, seed: int = 0, iterations: int = 3) -> float:
    """Scale factor so simulated goals-with-goalie-in-net match Track B's rates on the fit games.

    Track B's rates are average regulation goals without empty-netters over whole games,
    all score states included. The state factors and 6-on-5 boost shift that average a
    little; one overall scale puts it back.
    """
    rng = np.random.default_rng(seed)
    hi, ai = np.tile(gt["home_idx"].to_numpy(), reps), np.tile(gt["away_idx"].to_numpy(), reps)
    gh, ga = np.tile(gt["gamma_home"].to_numpy(), reps), np.tile(gt["gamma_away"].to_numpy(), reps)
    draw_idx = rng.integers(0, draws["alpha0"].size, hi.size)
    a0, ah = draws["alpha0"][draw_idx], draws["alpha_home"][draw_idx]
    lam_h = np.exp(a0 + ah + draws["mu_off"][draw_idx, hi] - draws["mu_def"][draw_idx, ai] - ga)
    lam_a = np.exp(a0 + draws["mu_off"][draw_idx, ai] - draws["mu_def"][draw_idx, hi] - gh)
    target = (lam_h + lam_a).mean()

    scale = 1.0
    for _ in range(iterations):
        reg = simulate_regulation_score_state(lam_h, lam_a, ss, rng, scale)
        simulated = (reg["reg_home"] + reg["reg_away"]).mean()
        scale *= target / simulated
    return float(scale)


def simulate_matchup_score_state(draws: dict, teams: list[str], late: dict, ss: dict, home: str, away: str,
                                 gamma_home: float = 0.0, gamma_away: float = 0.0,
                                 n_sims: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """One matchup simulated n_sims times with the score-state simulator, spread evenly over draws."""
    rng = np.random.default_rng(seed)
    draw_idx = np.resize(rng.permutation(draws["alpha0"].size), n_sims)
    h, a = teams.index(home), teams.index(away)
    return simulate_games_score_state(draws, late, ss, np.full(n_sims, h), np.full(n_sims, a),
                                      gamma_home, gamma_away, draw_idx, rng)