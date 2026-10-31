"""
sensitivity.py
==============
Phase 8, analysis A: which inputs make one game's prediction different from another's?

Home: src/analysis/sensitivity.py

The "truth model" is the exact game model (game_model.outcome_probs): microseconds per
game, so it is evaluated directly for the PCE design and for the Saltelli check.

Inputs (10), each uniform with the same mean and spread as its real distribution
(variance-matched: mean +/- sqrt(3) * sd, so the uniform has the real sd):

  Between games (what differs from matchup to matchup)
    off_home, off_away       team offense ratings     spread of the 32 teams' posterior means
    def_home, def_away       team defense ratings     same
    goalie_home, goalie_away starting goalie offset   spread of starters' gamma over the fit window
  League terms (the same in every game; varied over their posterior uncertainty)
    alpha0                   league scoring level     posterior mean +/- sqrt(3) sd
    alpha_home               home ice                 same
    rho                      tie inflation            same
    ot_factor                3-on-3 OT scoring factor late_game.json 90% range -> sd

Sobol' indices depend on how much each input varies, so these ranges are the main
modeling choice. Variance matching keeps every input at its real spread.

QoIs: P(home win), P(tied after regulation), expected regulation goals (both teams).
"""

import numpy as np

from src.models.game_model import outcome_probs, regulation_table

INPUT_NAMES = ["off_home", "def_home", "off_away", "def_away", "goalie_home", "goalie_away",
               "alpha0", "alpha_home", "rho", "ot_factor"]
GROUPS = {
    "team ratings":      ["off_home", "def_home", "off_away", "def_away"],
    "starting goalies":  ["goalie_home", "goalie_away"],
    "league terms":      ["alpha0", "alpha_home", "rho", "ot_factor"],
}
QOI_LABELS = {"p_home_win": "P(home win)", "p_tie_reg": "P(tied after regulation)",
              "exp_reg_goals": "Expected regulation goals"}
SQRT3 = np.sqrt(3.0)
Z90 = 1.6449
CHUNK = 20_000                 # games per outcome_probs call (memory: 16 x 16 table each)


def _uniform(mean, sd):
    return ('linear', float(mean - SQRT3 * sd), float(mean + SQRT3 * sd))


def input_specs(draws: dict, gt, late: dict) -> tuple[dict, dict]:
    """Uniform ranges for the 10 inputs from the active fit, its games and late_game.json.

    Returns (param_specs, info): param_specs {name: ('linear', lo, hi)} in INPUT_NAMES order;
    info {name: {"mean", "sd", "source"}} for the README / notebook table.
    """
    off_means = draws["mu_off"].mean(axis=0)            # one posterior mean per team
    def_means = draws["mu_def"].mean(axis=0)
    gam = np.concatenate([gt["gamma_home"].to_numpy(), gt["gamma_away"].to_numpy()])
    lo90, hi90 = late["ot"]["ot_factor_90"]
    info = {
        "off_home":    (off_means.mean(), off_means.std(), "sd of the teams' offense ratings (posterior means)"),
        "off_away":    (off_means.mean(), off_means.std(), "same"),
        "def_home":    (def_means.mean(), def_means.std(), "sd of the teams' defense ratings (posterior means)"),
        "def_away":    (def_means.mean(), def_means.std(), "same"),
        "goalie_home": (gam.mean(), gam.std(), "sd of the starting goalies' offsets over the fit window"),
        "goalie_away": (gam.mean(), gam.std(), "same"),
        "alpha0":      (draws["alpha0"].mean(), draws["alpha0"].std(), "posterior mean and sd"),
        "alpha_home":  (draws["alpha_home"].mean(), draws["alpha_home"].std(), "posterior mean and sd"),
        "rho":         (draws["rho"].mean(), draws["rho"].std(), "posterior mean and sd"),
        "ot_factor":   (late["ot"]["ot_factor"], (hi90 - lo90) / (2 * Z90), "late_game.json 90% range / 3.29"),
    }
    specs = {n: _uniform(info[n][0], info[n][1]) for n in INPUT_NAMES}
    if specs["rho"][1] <= -1.0:
        specs["rho"] = ('linear', -0.99, specs["rho"][2])
    if specs["ot_factor"][1] <= 0.0:
        specs["ot_factor"] = ('linear', 0.01, specs["ot_factor"][2])
    return specs, {n: {"mean": float(m), "sd": float(s), "source": src} for n, (m, s, src) in info.items()}


def game_qoi(theta: dict, late: dict) -> dict:
    """Exact game model for arrays of inputs. theta: {input name: (N,) array}."""
    th = {k: np.atleast_1d(np.asarray(v, float)) for k, v in theta.items()}
    n = th["off_home"].size
    out = {"p_home_win": np.empty(n), "p_tie_reg": np.empty(n), "exp_reg_goals": np.empty(n)}
    for s in range(0, n, CHUNK):
        sl = slice(s, s + CHUNK)
        lam_h = np.exp(th["alpha0"][sl] + th["alpha_home"][sl] + th["off_home"][sl]
                       - th["def_away"][sl] - th["goalie_away"][sl])
        lam_a = np.exp(th["alpha0"][sl] + th["off_away"][sl] - th["def_home"][sl] - th["goalie_home"][sl])
        rho = th["rho"][sl]
        late_k = {**late, "ot": {**late["ot"], "ot_factor": th["ot_factor"][sl]}}
        pr = outcome_probs(lam_h, lam_a, rho, late_k)
        table = regulation_table(lam_h, lam_a, rho)
        k = np.arange(table.shape[1])
        out["p_home_win"][sl] = pr["home_win"]
        out["p_tie_reg"][sl] = pr["tie_reg"]
        out["exp_reg_goals"][sl] = (table.sum(axis=2) @ k) + (table.sum(axis=1) @ k)
    return out


def truth_predictor(param_specs: dict, late: dict, qoi: str):
    """xi (N, d) -> QoI values from the exact game model (for the Saltelli check)."""
    from src.models.stochastic.pce import xi_to_theta
    def predict(Xi):
        return game_qoi(xi_to_theta(Xi, param_specs), late)[qoi]
    return predict


def sd_contributions(S, variance) -> np.ndarray:
    """First-order Sobol' index as 'how much this input alone moves the QoI' (sd units)."""
    return np.sqrt(np.clip(np.asarray(S), 0, None) * variance)