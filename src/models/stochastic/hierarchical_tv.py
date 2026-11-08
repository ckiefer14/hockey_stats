"""
hierarchical_tv.py
==================
Track B v2.0: v1.1 (ratings from xG + goals) with ratings that change over time.

Home: src/models/stochastic/hierarchical_tv.py

Time is cut into steps of `step_days` days (7 = weekly; 2 is about one step per game).
Only steps that contain games exist; the gap between two steps sets how far things can
move between them. A gap of more than SEASON_GAP_DAYS is an off-season.

  Team shot-quality ratings (random walk, one per team, offense and defense):
      mu[t, 0] = sigma * z0[t]
      mu[t, k] = mu[t, k-1] + s_k * eps[t, k],   eps summing to zero over teams each step
      s_k = tau * sqrt(gap_days / 7)  in season  (tau = typical change per week)
      s_k = tau_summer               across an off-season
  League levels (random walks, same rule, their own tau's):
      beta0[k]   league xG per team-game (log)
      delta0[k]  league goals per xG (log): finishing / goaltending trends
  Per game (step k), exactly as v1.1:
      log nu_home  = beta0[k] + beta_home + mu_off[h, k] - mu_def[a, k]
      xG_home ~ Gamma(mean nu_home, shape kappa)
      log lam_home = log nu_home + delta0[k] + delta_home + phi_off[h] - phi_def[a] - gamma_away
      goals ~ tie-inflated Poisson(lam_home, lam_away; rho)
  Finishing effects phi, home ice, kappa and rho stay fixed over time.

The zero-sum steps make league-wide changes go to beta0 / delta0 and team changes to mu:
without them, a league-wide rise in scoring would move all 32 ratings together.

Ratings for prediction: after sampling, `add_current_ratings` steps every draw forward from
the last step to the window end (adding the drift for that gap, an off-season step if it
crosses one) and saves the result under the v1 names (alpha0, alpha_home, mu_off, mu_def),
so game_model, the simulators, predict_day, the cards and the backtest work unchanged.
The weekly paths are kept as mu_off_path / mu_def_path / beta0_path / delta0_path.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.stochastic.hierarchical import K_MAX, sample_scores
from src.models.stochastic.hierarchical_xg import (
    BETA0_PRIOR, HOME_PRIOR_SD, DELTA_PRIOR_SD, SIGMA_PRIOR_SD, SIGMA_PHI_PRIOR_SD,
    LOG_KAPPA_PRIOR, TIE_PRIOR_SD, XG_COLS, GOAL_COLS, XG_FLOOR,
)

STEP_DAYS = 7
SEASON_GAP_DAYS = 45          # a gap between game steps longer than this is an off-season

# Random-walk step priors (half-normal sd). Per week in season; per off-season otherwise.
TAU_TEAM_PRIOR = 0.02         # team ratings: a few hundredths a month at most
TAU_TEAM_SUMMER_PRIOR = 0.10  # off-season: roster turnover can move a team ~0.1
TAU_LEAGUE_PRIOR = 0.01       # league xG level / conversion, per week
TAU_LEAGUE_SUMMER_PRIOR = 0.05
REST_PRIOR_SD = 0.1           # back-to-back effects on the log scoring rate (v2.0d)
TAU_HOME_SUMMER_PRIOR = 0.05  # season-to-season change in home ice (v2.0d, home_drift)
REST_VARS = ["rest_off_home", "rest_def_home", "rest_off_away", "rest_def_away"]
# v2.0e: back-to-back effects on goals per xG (finishing of the tired team, its goalie)
REST_GOAL_VARS = ["rest_fin_home", "rest_gk_home", "rest_fin_away", "rest_gk_away"]
# v2.0f: schedule context on shot quality, same for home and away teams (schedule_context.py):
# trav = overnight travel (1000s of km), tz = time zones crossed, dense = 3rd game in 4 nights
TRAVEL_FEATURES = ["trav", "tz", "dense"]
TRAVEL_VARS = [f"{f}_{k}" for f in TRAVEL_FEATURES for k in ("off", "def")]
# v2.0g: a starting goalie on zero rest (also started yesterday) lets in more of the same
# shots: one effect on the opponent's goals per xG, shared by home and road goalies
GOALIE_REST_VARS = ["gk_b2b"]
# v2.2: lineup strength vs the team's usual lineup (lineup_context.py), a known log xG-rate
# shift lin_home / lin_away scaled by a learned kappa (1 = take the player ratings at face value)
LINEUP_VARS = ["kappa_lineup"]
KAPPA_PRIOR = (1.0, 0.5)
GK_B2B_PRIOR_SD = 0.15
ALL_REST_VARS = REST_VARS + REST_GOAL_VARS + TRAVEL_VARS + GOALIE_REST_VARS + LINEUP_VARS


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------

def time_grid(dates, step_days: int = STEP_DAYS, season_gap: int = SEASON_GAP_DAYS) -> dict:
    """Step index for each game and the time (days) between consecutive steps.

    Steps are bins of `step_days` days counted from the first game; only bins with games
    are kept. The time between two steps is the distance between their bin starts (7 for
    consecutive weeks, 14 if a week without games sits between them); more than
    `season_gap` days means an off-season.

    Returns {"step": (n_games,) int, "start": (K,) bin start dates, "mid": (K,) bin
             middles, "gap_days": (K,) days from the previous step (0 for the first),
             "summer": (K,) bool, "step_days": step_days}.
    """
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    d0 = d.min()
    bins = ((d - d0).dt.days // step_days).to_numpy()
    used = np.unique(bins)
    step = np.searchsorted(used, bins)
    gap = np.zeros(len(used))
    gap[1:] = np.diff(used) * step_days
    start = d0 + pd.to_timedelta(used * step_days, unit="D")
    return {"step": step, "start": start, "mid": start + pd.Timedelta(days=step_days / 2),
            "gap_days": gap, "summer": gap > season_gap, "step_days": step_days}


def _scales(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per step: sqrt(gap / 7) for in-season steps (0 for summer steps and step 0),
    and a 0/1 indicator for summer steps."""
    g = grid["gap_days"].astype(float)
    in_season = (~grid["summer"]) & (np.arange(len(g)) > 0)
    return np.where(in_season, np.sqrt(g / 7.0), 0.0), grid["summer"].astype(float)


def _walk(name: str, start, tau, tau_summer, scale_in, is_summer, shape_extra=None, zero_sum=False):
    """Random walk with non-centered steps. Returns (K,) or (K, n) tensor."""
    K = len(scale_in)
    if shape_extra is None:
        eps = pm.Normal(f"{name}_eps", 0.0, 1.0, shape=K)
        s = tau * scale_in + tau_summer * is_summer
        return start + pt.cumsum(s * eps)
    if zero_sum:
        eps = pm.ZeroSumNormal(f"{name}_eps", sigma=1.0, shape=(K, shape_extra), n_zerosum_axes=1)
    else:
        eps = pm.Normal(f"{name}_eps", 0.0, 1.0, shape=(K, shape_extra))
    s = (tau * scale_in + tau_summer * is_summer)[:, None]
    return start[None, :] + pt.cumsum(s * eps, axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_tv(gt: pd.DataFrame, teams: list[str], step_days: int = STEP_DAYS,
                   rest: bool = False, home_drift: bool = False,
                   rest_goals: bool = False, travel: bool = False,
                   goalie_rest: bool = False, lineup: bool = False) -> tuple[pm.Model, dict]:
    """Track B v2.0 for the games in gt. Returns (model, grid).

    rest (v2.0d): back-to-back effects. A team on the second night of a back-to-back
        generates rest_off_* less shot quality (log scale) and allows rest_def_* more;
        separate values for a tired home team and a tired road team. Needs b2b_home /
        b2b_away in gt (build_game_table adds them).
    rest_goals (v2.0e): back-to-back effects on goals per xG. A tired team converts its
        chances rest_fin_* worse and its goalie (often a backup or tired starter) lets in
        rest_gk_* more of the same shots. These move goals without moving xG.
    travel (v2.0f): schedule context on shot quality. For each feature f in TRAVEL_FEATURES,
        a team with value x creates f_off * x less xG (log scale) and allows f_def * x more.
        Shared by home and road teams (the b2b terms already carry the side difference).
        Needs <f>_home / <f>_away in gt (build_game_table adds them).
    lineup (v2.2): log_nu += kappa_lineup * lin_<side> (lineup strength vs usual; needs
        lin_home / lin_away in gt, from build_game_table(..., lineup=True)).
    goalie_rest (v2.0g): gk_b2b, the log change in the opponent's goals per xG when the
        starting goalie also started yesterday. Needs g2b_home / g2b_away in gt.
    home_drift (v2.0d): home ice changes from season to season (a random walk that steps
        only across off-seasons) instead of one value for the whole window.
    """
    grid = time_grid(gt["game_date"], step_days)
    k = grid["step"]
    K, T = len(grid["gap_days"]), len(teams)
    scale_in, is_summer = _scales(grid)
    coords = {"team": teams, "step": np.arange(K), "game": gt["game_id"].to_numpy()}

    hi, ai = gt["home_idx"].to_numpy(), gt["away_idx"].to_numpy()
    g_home, g_away = gt["gamma_home"].to_numpy(), gt["gamma_away"].to_numpy()
    xg_h = np.maximum(gt[XG_COLS[0]].to_numpy(float), XG_FLOOR)
    xg_a = np.maximum(gt[XG_COLS[1]].to_numpy(float), XG_FLOOR)
    y_h, y_a = gt[GOAL_COLS[0]].to_numpy(), gt[GOAL_COLS[1]].to_numpy()
    tied = (y_h == y_a).astype(float)
    if rest or rest_goals:
        b2b_h, b2b_a = gt["b2b_home"].to_numpy(float), gt["b2b_away"].to_numpy(float)
    if goalie_rest:
        g2b_h, g2b_a = gt["g2b_home"].to_numpy(float), gt["g2b_away"].to_numpy(float)
    if lineup:
        lin_h, lin_a = gt["lin_home"].to_numpy(float), gt["lin_away"].to_numpy(float)
    if travel:
        ctx = {f"{f}_{sd}": gt[f"{f}_{sd}"].to_numpy(float) for f in TRAVEL_FEATURES for sd in ("home", "away")}

    with pm.Model(coords=coords) as model:
        # 1. Random-walk step sizes
        tau_off = pm.HalfNormal("tau_off", TAU_TEAM_PRIOR)
        tau_def = pm.HalfNormal("tau_def", TAU_TEAM_PRIOR)
        tau_off_s = pm.HalfNormal("tau_off_summer", TAU_TEAM_SUMMER_PRIOR)
        tau_def_s = pm.HalfNormal("tau_def_summer", TAU_TEAM_SUMMER_PRIOR)
        tau_b = pm.HalfNormal("tau_beta0", TAU_LEAGUE_PRIOR)
        tau_d = pm.HalfNormal("tau_delta0", TAU_LEAGUE_PRIOR)
        tau_b_s = pm.HalfNormal("tau_beta0_summer", TAU_LEAGUE_SUMMER_PRIOR)
        tau_d_s = pm.HalfNormal("tau_delta0_summer", TAU_LEAGUE_SUMMER_PRIOR)

        # 2. League levels over time
        beta0_init = pm.Normal("beta0_init", BETA0_PRIOR[0], BETA0_PRIOR[1])
        delta0_init = pm.Normal("delta0_init", 0.0, DELTA_PRIOR_SD)
        beta0 = pm.Deterministic("beta0_path", _walk("beta0", beta0_init, tau_b, tau_b_s, scale_in, is_summer),
                                 dims="step")
        delta0 = pm.Deterministic("delta0_path", _walk("delta0", delta0_init, tau_d, tau_d_s, scale_in, is_summer),
                                  dims="step")
        beta_home = pm.Normal("beta_home", 0.0, HOME_PRIOR_SD)
        if home_drift:                                   # steps only across off-seasons
            tau_h_s = pm.HalfNormal("tau_home_summer", TAU_HOME_SUMMER_PRIOR)
            eps_h = pm.Normal("home_eps", 0.0, 1.0, shape=K)
            home = pm.Deterministic("beta_home_path", beta_home + pt.cumsum(tau_h_s * is_summer * eps_h),
                                    dims="step")
        else:
            home = pt.zeros(K) + beta_home
        delta_home = pm.Normal("delta_home", 0.0, DELTA_PRIOR_SD)
        log_kappa = pm.Normal("log_kappa", LOG_KAPPA_PRIOR[0], LOG_KAPPA_PRIOR[1])
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))

        # 3. Team shot-quality ratings over time (zero-sum steps)
        sigma_off = pm.HalfNormal("sigma_off", SIGMA_PRIOR_SD)
        sigma_def = pm.HalfNormal("sigma_def", SIGMA_PRIOR_SD)
        z0_off = pm.ZeroSumNormal("z0_off", sigma=1.0, dims="team")
        z0_def = pm.ZeroSumNormal("z0_def", sigma=1.0, dims="team")
        mu_off = pm.Deterministic("xg_mu_off_path", _walk("off", sigma_off * z0_off, tau_off, tau_off_s,
                                                          scale_in, is_summer, T, zero_sum=True),
                                  dims=("step", "team"))
        mu_def = pm.Deterministic("xg_mu_def_path", _walk("def", sigma_def * z0_def, tau_def, tau_def_s,
                                                          scale_in, is_summer, T, zero_sum=True),
                                  dims=("step", "team"))

        # 4. Finishing effects (static, tightly pooled), as v1.1
        s_phi_off = pm.HalfNormal("sigma_phi_off", SIGMA_PHI_PRIOR_SD)
        s_phi_def = pm.HalfNormal("sigma_phi_def", SIGMA_PHI_PRIOR_SD)
        phi_off = pm.Deterministic("phi_off", s_phi_off * pm.ZeroSumNormal("zp_off", sigma=1.0, dims="team"),
                                   dims="team")
        phi_def = pm.Deterministic("phi_def", s_phi_def * pm.ZeroSumNormal("zp_def", sigma=1.0, dims="team"),
                                   dims="team")

        # 5. xG likelihood
        log_nu_h = beta0[k] + home[k] + mu_off[k, hi] - mu_def[k, ai]
        log_nu_a = beta0[k] + mu_off[k, ai] - mu_def[k, hi]
        if rest:                                         # tired team: less for, more against
            r_off_h = pm.Normal("rest_off_home", 0.0, REST_PRIOR_SD)
            r_def_h = pm.Normal("rest_def_home", 0.0, REST_PRIOR_SD)
            r_off_a = pm.Normal("rest_off_away", 0.0, REST_PRIOR_SD)
            r_def_a = pm.Normal("rest_def_away", 0.0, REST_PRIOR_SD)
            log_nu_h = log_nu_h - r_off_h * b2b_h + r_def_a * b2b_a
            log_nu_a = log_nu_a - r_off_a * b2b_a + r_def_h * b2b_h
        if lineup:                                       # who dressed tonight vs the usual lineup
            kap = pm.Normal("kappa_lineup", *KAPPA_PRIOR)
            log_nu_h = log_nu_h + kap * lin_h
            log_nu_a = log_nu_a + kap * lin_a
        if travel:                                       # travel, jet lag, crowded stretches
            for f in TRAVEL_FEATURES:
                e_off = pm.Normal(f"{f}_off", 0.0, REST_PRIOR_SD)
                e_def = pm.Normal(f"{f}_def", 0.0, REST_PRIOR_SD)
                log_nu_h = log_nu_h - e_off * ctx[f"{f}_home"] + e_def * ctx[f"{f}_away"]
                log_nu_a = log_nu_a - e_off * ctx[f"{f}_away"] + e_def * ctx[f"{f}_home"]

        def gamma_logp(x, log_nu):
            return (kappa * (log_kappa - log_nu) - pt.gammaln(kappa)
                    + (kappa - 1.0) * np.log(x) - kappa * x * pm.math.exp(-log_nu))
        pm.Potential("xg_loglik", pm.math.sum(gamma_logp(xg_h, log_nu_h) + gamma_logp(xg_a, log_nu_a)))

        # 6. Goals likelihood: tie-inflated Poisson, as v1.1
        log_lam_h = log_nu_h + delta0[k] + delta_home + phi_off[hi] - phi_def[ai] - g_away
        log_lam_a = log_nu_a + delta0[k] + phi_off[ai] - phi_def[hi] - g_home
        if rest_goals:                                   # tired team: finishes worse, goalie leaks
            f_h = pm.Normal("rest_fin_home", 0.0, REST_PRIOR_SD)
            k_h = pm.Normal("rest_gk_home", 0.0, REST_PRIOR_SD)
            f_a = pm.Normal("rest_fin_away", 0.0, REST_PRIOR_SD)
            k_a = pm.Normal("rest_gk_away", 0.0, REST_PRIOR_SD)
            log_lam_h = log_lam_h - f_h * b2b_h + k_a * b2b_a
            log_lam_a = log_lam_a - f_a * b2b_a + k_h * b2b_h
        if goalie_rest:                                  # tired goalie: more goals from the same xG
            gk = pm.Normal("gk_b2b", 0.0, GK_B2B_PRIOR_SD)
            log_lam_h = log_lam_h + gk * g2b_a
            log_lam_a = log_lam_a + gk * g2b_h
        lam_h, lam_a = pm.math.exp(log_lam_h), pm.math.exp(log_lam_a)
        tau = pm.Normal("tau", 0.0, TIE_PRIOR_SD)
        rho = pm.Deterministic("rho", pm.math.exp(tau) - 1.0)
        pois = (y_h * log_lam_h - lam_h - pt.gammaln(y_h + 1.0)
                + y_a * log_lam_a - lam_a - pt.gammaln(y_a + 1.0))
        kk = np.arange(K_MAX + 1)[:, None]
        lgk = pt.gammaln(kk + 1.0)
        T_tie = pm.math.exp(kk * log_lam_h[None, :] - lam_h[None, :] - lgk
                            + kk * log_lam_a[None, :] - lam_a[None, :] - lgk).sum(axis=0)
        pm.Potential("goal_loglik", pm.math.sum(pois + tau * tied - pm.math.log1p(rho * T_tie)))

        # 7. Home ice on the goal scale (as v1.1); with home_drift, the latest season's
        pm.Deterministic("alpha_home", home[-1] + delta_home)

    return model, grid


# Variables kept in the trace (the random-walk innovations are not needed after sampling,
# and dropping them keeps saved fits small)
KEEP_VARS = ["tau_off", "tau_def", "tau_off_summer", "tau_def_summer", "tau_beta0", "tau_delta0",
             "tau_beta0_summer", "tau_delta0_summer", "beta0_init", "delta0_init", "beta_home", "delta_home",
             "log_kappa", "kappa", "sigma_off", "sigma_def", "sigma_phi_off", "sigma_phi_def", "tau", "rho",
             "alpha_home", "phi_off", "phi_def", "beta0_path", "delta0_path", "xg_mu_off_path", "xg_mu_def_path",
             "tau_home_summer", "beta_home_path"] + ALL_REST_VARS


TARGET_ACCEPT_TV = 0.95      # random-walk steps near tau = 0 need smaller NUTS steps than v1.1 (0.9)


def sample_tv(model: pm.Model, seed: int = 42, progressbar: bool = True, **kw):
    """NUTS as hierarchical.sample (cores=1), storing only KEEP_VARS."""
    from src.models.stochastic.hierarchical import DRAWS, TUNE, CHAINS
    with model:
        return pm.sample(draws=kw.get("draws", DRAWS), tune=kw.get("tune", TUNE), chains=kw.get("chains", CHAINS),
                         cores=1, target_accept=kw.get("target_accept", TARGET_ACCEPT_TV), random_seed=seed,
                         progressbar=progressbar, var_names=[v for v in KEEP_VARS if v in model.named_vars])


def compact_paths(idata) -> None:
    """Store the rating paths as float32 before saving (halves the file; plenty of precision)."""
    post = idata.posterior
    for name in ["beta0_path", "delta0_path", "xg_mu_off_path", "xg_mu_def_path"]:
        da = post[name].astype("float32")
        try:
            post[name] = da
        except TypeError:
            idata["posterior"][name] = da


# ---------------------------------------------------------------------------
# Ratings at the window end (what predictions use)
# ---------------------------------------------------------------------------

def _set_posterior(idata, name, values, dims):
    """Add a variable to the posterior (works for ArviZ InferenceData and DataTree)."""
    import xarray as xr
    post = idata.posterior
    coords = {d: post.coords[d] for d in dims if d in post.coords}
    da = xr.DataArray(values, dims=dims, coords=coords)
    try:
        post[name] = da
    except TypeError:                                     # DataTree node
        idata["posterior"][name] = da


def add_current_ratings(idata, grid: dict, end, seed: int = 0) -> None:
    """Step every posterior draw from the last step to `end` and save goal-scale
    ratings under the v1 names: alpha0, mu_off, mu_def (alpha_home is already there).

    The gap is measured from the middle of the last step; an off-season step is used if
    it is longer than SEASON_GAP_DAYS (a pre-season fit), otherwise in-season drift.
    """
    rng = np.random.default_rng(seed)
    post = idata.posterior
    gap = max((pd.Timestamp(end).normalize() - pd.Timestamp(grid["mid"][-1])).days, 0)
    summer = gap > SEASON_GAP_DAYS

    def step(cur, tau_in, tau_summer, zero_sum=False):
        s = post[tau_summer].values if summer else post[tau_in].values * np.sqrt(gap / 7.0)
        noise = rng.standard_normal(cur.shape)
        if zero_sum:
            noise = noise - noise.mean(axis=-1, keepdims=True)
            s = s[..., None]
        return cur + s * noise

    beta0 = step(post["beta0_path"].values[:, :, -1], "tau_beta0", "tau_beta0_summer")
    delta0 = step(post["delta0_path"].values[:, :, -1], "tau_delta0", "tau_delta0_summer")
    off = step(post["xg_mu_off_path"].values[:, :, -1, :], "tau_off", "tau_off_summer", zero_sum=True)
    dfn = step(post["xg_mu_def_path"].values[:, :, -1, :], "tau_def", "tau_def_summer", zero_sum=True)

    if "beta_home_path" in post:                         # v2.0d home_drift: latest season's home ice
        home = post["beta_home_path"].values[:, :, -1]
        if summer:
            home = home + post["tau_home_summer"].values * rng.standard_normal(home.shape)
        _set_posterior(idata, "alpha_home", home + post["delta_home"].values, ("chain", "draw"))
    _set_posterior(idata, "alpha0", beta0 + delta0, ("chain", "draw"))
    _set_posterior(idata, "mu_off", off + post["phi_off"].values, ("chain", "draw", "team"))
    _set_posterior(idata, "mu_def", dfn + post["phi_def"].values, ("chain", "draw", "team"))
    idata.posterior.attrs.update({"ratings_as_of": str(pd.Timestamp(end).date()),
                                  "days_after_last_step": int(gap), "crosses_off_season": int(summer),
                                  "step_days": int(grid["step_days"]), "first_game": str(pd.Timestamp(grid["start"][0]).date())})


# ---------------------------------------------------------------------------
# Scoring rates of the fit games at the time they were played
# ---------------------------------------------------------------------------
# late_game (OT factor) and score_state (score effects, simulator scale) compare real games
# with each game's expected scoring rate. With time-varying ratings that rate must come from
# the ratings as they were on the game's date, not from today's.

PATH_VARS = ["beta0_path", "delta0_path", "xg_mu_off_path", "xg_mu_def_path", "phi_off", "phi_def"]


def is_time_varying(post_or_draws) -> bool:
    return "xg_mu_off_path" in post_or_draws


def add_paths(draws: dict, post) -> dict:
    """Add pooled path draws (and step_days) to a draws dict, if the fit has them."""
    if not is_time_varying(post):
        return draws
    for name in PATH_VARS + ["beta_home_path", "delta_home"] + ALL_REST_VARS:
        if name in post:
            x = post[name].values
            draws[name] = x.reshape(-1, *x.shape[2:])
    draws["step_days"] = int(post.attrs.get("step_days", STEP_DAYS))
    return draws


def _shift(m: dict, x: dict):
    """Log scoring-rate shifts (home, away) from the schedule context.

    m: effect values (floats or per-draw arrays) for the terms the fit has; x: context arrays
    b2b_home, b2b_away and, for v2.0f, <f>_home / <f>_away for f in TRAVEL_FEATURES.
    Shot-quality terms (v2.0d, v2.0f) and goals-per-xG terms (v2.0e) add on the log scale."""
    b2b_h, b2b_a = np.asarray(x["b2b_home"], float), np.asarray(x["b2b_away"], float)
    sh, sa = 0.0 * b2b_h, 0.0 * b2b_a
    if "rest_off_home" in m:
        sh = sh - m["rest_off_home"] * b2b_h + m["rest_def_away"] * b2b_a
        sa = sa - m["rest_off_away"] * b2b_a + m["rest_def_home"] * b2b_h
    if "rest_fin_home" in m:
        sh = sh - m["rest_fin_home"] * b2b_h + m["rest_gk_away"] * b2b_a
        sa = sa - m["rest_fin_away"] * b2b_a + m["rest_gk_home"] * b2b_h
    if "kappa_lineup" in m:
        sh = sh + m["kappa_lineup"] * np.asarray(x["lin_home"], float)
        sa = sa + m["kappa_lineup"] * np.asarray(x["lin_away"], float)
    if "gk_b2b" in m:
        sh = sh + m["gk_b2b"] * np.asarray(x["g2b_away"], float)
        sa = sa + m["gk_b2b"] * np.asarray(x["g2b_home"], float)
    if "trav_off" in m:
        for f in TRAVEL_FEATURES:
            fh, fa = np.asarray(x[f"{f}_home"], float), np.asarray(x[f"{f}_away"], float)
            sh = sh - m[f"{f}_off"] * fh + m[f"{f}_def"] * fa
            sa = sa - m[f"{f}_off"] * fa + m[f"{f}_def"] * fh
    return sh, sa


def context_shift(draws: dict, x) -> tuple:
    """Log-rate shifts (home, away) for games with schedule context x (a DataFrame or dict
    with b2b_home / b2b_away and, for v2.0f fits, the travel columns), posterior means.
    Zero for a fit without these terms. Prediction code adds them through the goalie
    offsets: gamma_away -= shift_home, gamma_home -= shift_away."""
    m = {n: float(np.mean(draws[n])) for n in ALL_REST_VARS if n in draws}
    if "trav_off" in m and not all(f"{f}_home" in x for f in TRAVEL_FEATURES):
        raise ValueError("this fit has travel terms (v2.0f); pass the schedule context columns "
                         "(src.processing.schedule_context)")
    if "kappa_lineup" in m and not ("lin_home" in x and "lin_away" in x):
        raise ValueError("this fit has a lineup term (v2.2); pass lin_home / lin_away")
    if "gk_b2b" in m and not ("g2b_home" in x and "g2b_away" in x):
        raise ValueError("this fit has a goalie-rest term (v2.0g); pass g2b_home / g2b_away")
    return _shift(m, x)


def rest_shift(draws: dict, b2b_home, b2b_away):
    """Log-rate shifts (home, away) from back-to-backs, posterior means (0 if the fit has no rest terms).

    Prediction code adds them through the goalie offsets: gamma_away -= shift_home,
    gamma_home -= shift_away, so every downstream function works unchanged.
    """
    b2b_home, b2b_away = np.asarray(b2b_home, float), np.asarray(b2b_away, float)
    m = {n: float(np.mean(draws[n])) for n in ALL_REST_VARS if n in draws}
    if "trav_off" in m or "gk_b2b" in m or "kappa_lineup" in m:
        raise ValueError("this fit has travel / goalie-rest terms: use context_shift with the full context")
    return _shift(m, {"b2b_home": b2b_home, "b2b_away": b2b_away})


def game_steps(gt: pd.DataFrame, step_days: int) -> np.ndarray:
    """Step index of each game: the same grid the fit used, provided gt is the fit's games."""
    return time_grid(gt["game_date"], step_days)["step"]


def game_rates(d: dict, gt: pd.DataFrame, draw_idx=None, reps: int = 1):
    """Regulation scoring rates (lam_home, lam_away) for the fit games, from the paths.

    d: draws dict with paths (add_paths) plus alpha_home. draw_idx: one posterior draw per
    game (length reps * len(gt), games tiled reps times); None = posterior means (plug-in).
    """
    k = np.tile(game_steps(gt, d["step_days"]), reps)
    hi, ai = np.tile(gt["home_idx"].to_numpy(), reps), np.tile(gt["away_idx"].to_numpy(), reps)
    gh, ga = np.tile(gt["gamma_home"].to_numpy(), reps), np.tile(gt["gamma_away"].to_numpy(), reps)
    x = {c: np.tile(gt[c].to_numpy(float), reps) if c in gt else np.zeros(len(k))
         for c in ["b2b_home", "b2b_away", "g2b_home", "g2b_away", "lin_home", "lin_away"]
         + [f"{f}_{sd}" for f in TRAVEL_FEATURES for sd in ("home", "away")]}
    if draw_idx is None:
        m = {n: d[n].mean(axis=0) for n in PATH_VARS + ["alpha_home", "beta_home_path", "delta_home"] + ALL_REST_VARS
             if n in d}
        lvl = m["beta0_path"][k] + m["delta0_path"][k]
        home = m["beta_home_path"][k] + m["delta_home"] if "beta_home_path" in m else m["alpha_home"]
        rh, ra = _shift(m, x)
        lam_h = np.exp(lvl + home + m["xg_mu_off_path"][k, hi] + m["phi_off"][hi]
                       - m["xg_mu_def_path"][k, ai] - m["phi_def"][ai] - ga + rh)
        lam_a = np.exp(lvl + m["xg_mu_off_path"][k, ai] + m["phi_off"][ai]
                       - m["xg_mu_def_path"][k, hi] - m["phi_def"][hi] - gh + ra)
        return lam_h, lam_a
    j = np.asarray(draw_idx)
    lvl = d["beta0_path"][j, k] + d["delta0_path"][j, k]
    home = d["beta_home_path"][j, k] + d["delta_home"][j] if "beta_home_path" in d else d["alpha_home"][j]
    rh, ra = _shift({n: d[n][j] for n in ALL_REST_VARS if n in d}, x)
    lam_h = np.exp(lvl + home + d["xg_mu_off_path"][j, k, hi] + d["phi_off"][j, hi]
                   - d["xg_mu_def_path"][j, k, ai] - d["phi_def"][j, ai] - ga + rh)
    lam_a = np.exp(lvl + d["xg_mu_off_path"][j, k, ai] + d["phi_off"][j, ai]
                   - d["xg_mu_def_path"][j, k, hi] - d["phi_def"][j, hi] - gh + ra)
    return lam_h, lam_a


# ---------------------------------------------------------------------------
# Synthetic league with known drifting ratings (recovery test)
# ---------------------------------------------------------------------------

def simulate_league_tv(n_teams: int = 16, seasons: int = 2, games_per_team: int = 60,
                       sigma: float = 0.08, tau_week: float = 0.015, tau_summer: float = 0.06,
                       tau_league_week: float = 0.004, tau_league_summer: float = 0.03,
                       beta0: float = np.log(2.6), beta_home: float = 0.05, delta0: float = 0.0,
                       kappa: float = 9.0, rho: float = 0.5, season_days: int = 180,
                       off_season_days: int = 150, seed: int = 0,
                       rest_effects: tuple = (0.0, 0.0, 0.0, 0.0), b2b_share: float = 0.15,
                       rest_goal_effects: tuple = (0.0, 0.0, 0.0, 0.0),
                       travel_effects: dict | None = None, gk_b2b: float = 0.0,
                       lineup_kappa: float = 0.0, lineup_sd: float = 0.0,
                       home_by_season: tuple | None = None):
    """Fake league with drifting team and league levels, in game-table format.

    Truth is daily random walks (tau's are per week, scaled by sqrt(days / 7)).
    Returns (gt, teams, truth) with truth paths by day.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    days = []
    for s in range(seasons):
        start = s * (season_days + off_season_days)
        days.extend(range(start, start + season_days))
    days = np.array(days)
    D = days.max() + 1

    def walk(dim, t_week, t_summer, init):
        x = np.zeros((D,) + dim)
        x[0] = init
        for d in range(1, D):
            summer_start = any(d == s * (season_days + off_season_days) for s in range(1, seasons))
            step = (t_summer if summer_start else 0.0) + (0.0 if summer_start else t_week / np.sqrt(7))
            if d in days or summer_start:
                inc = rng.standard_normal(dim) * step
                if dim:
                    inc -= inc.mean()
                x[d] = x[d - 1] + inc
            else:
                x[d] = x[d - 1]
        return x

    off = walk((n_teams,), tau_week, tau_summer, rng.normal(0, sigma, n_teams))
    dfn = walk((n_teams,), tau_week, tau_summer, rng.normal(0, sigma, n_teams))
    off -= off.mean(axis=1, keepdims=True); dfn -= dfn.mean(axis=1, keepdims=True)
    b0 = walk((), tau_league_week, tau_league_summer, beta0)
    d0 = walk((), tau_league_week, tau_league_summer, delta0)

    n_games = n_teams * games_per_team * seasons // 2
    gd = np.sort(rng.choice(days, n_games))
    home = rng.integers(0, n_teams, n_games)
    away = (home + rng.integers(1, n_teams, n_games)) % n_teams
    gh, ga = rng.normal(0, 0.03, n_games), rng.normal(0, 0.03, n_games)
    # back-to-backs (random, b2b_share of team-games) and season-specific home ice
    b2b_h, b2b_a = (rng.random(n_games) < b2b_share).astype(int), (rng.random(n_games) < b2b_share).astype(int)
    r_off_h, r_def_h, r_off_a, r_def_a = rest_effects
    season_of = gd // (season_days + off_season_days)
    bh = np.asarray(home_by_season)[season_of] if home_by_season is not None else beta_home
    nu_h = np.exp(b0[gd] + bh + off[gd, home] - dfn[gd, away] - r_off_h * b2b_h + r_def_a * b2b_a)
    nu_a = np.exp(b0[gd] + off[gd, away] - dfn[gd, home] - r_off_a * b2b_a + r_def_h * b2b_h)
    # v2.0f schedule context: overnight km (on back-to-backs), time zones, 3-in-4 (mostly with a b2b)
    ctx = {}
    for sd, b in (("home", b2b_h), ("away", b2b_a)):
        ctx[f"trav_{sd}"] = b * rng.uniform(0.2, 3.0, n_games)
        ctx[f"tz_{sd}"] = rng.integers(0, 4, n_games) * (rng.random(n_games) < 0.3)
        ctx[f"dense_{sd}"] = ((b == 1) & (rng.random(n_games) < 0.4)).astype(int)
    lin_h = rng.normal(0, lineup_sd, n_games) if lineup_sd > 0 else np.zeros(n_games)   # v2.2
    lin_a = rng.normal(0, lineup_sd, n_games) if lineup_sd > 0 else np.zeros(n_games)
    nu_h = nu_h * np.exp(lineup_kappa * lin_h)
    nu_a = nu_a * np.exp(lineup_kappa * lin_a)
    te = travel_effects or {}
    for f in TRAVEL_FEATURES:
        eo, ed = te.get(f"{f}_off", 0.0), te.get(f"{f}_def", 0.0)
        nu_h = nu_h * np.exp(-eo * ctx[f"{f}_home"] + ed * ctx[f"{f}_away"])
        nu_a = nu_a * np.exp(-eo * ctx[f"{f}_away"] + ed * ctx[f"{f}_home"])
    xg_h, xg_a = rng.gamma(kappa, nu_h / kappa), rng.gamma(kappa, nu_a / kappa)
    f_h, k_h, f_a, k_a = rest_goal_effects               # v2.0e: on goals per xG only
    # v2.0g: on about a third of back-to-backs the same goalie starts both nights
    g2b_h = ((b2b_h == 1) & (rng.random(n_games) < 0.35)).astype(int)
    g2b_a = ((b2b_a == 1) & (rng.random(n_games) < 0.35)).astype(int)
    lam_h = nu_h * np.exp(d0[gd] - ga - f_h * b2b_h + k_a * b2b_a + gk_b2b * g2b_a)
    lam_a = nu_a * np.exp(d0[gd] - gh - f_a * b2b_a + k_h * b2b_h + gk_b2b * g2b_h)
    y_h, y_a = sample_scores(lam_h, lam_a, rho, rng)

    date0 = pd.Timestamp("2000-10-01")
    gt = pd.DataFrame({
        "game_id": np.arange(n_games), "game_date": date0 + pd.to_timedelta(gd, unit="D"),
        "season": gd // (season_days + off_season_days),
        "home": [teams[i] for i in home], "away": [teams[i] for i in away],
        "home_idx": home, "away_idx": away,
        "home_goals": y_h, "away_goals": y_a, "home_goals_reg": y_h, "away_goals_reg": y_a,
        "home_xg_reg": xg_h, "away_xg_reg": xg_a,
        "home_goalie": 0, "away_goalie": 0, "gamma_home": gh, "gamma_away": ga,
        "b2b_home": b2b_h, "b2b_away": b2b_a, "g2b_home": g2b_h, "g2b_away": g2b_a, **ctx,
        "lin_home": lin_h, "lin_away": lin_a,
    })
    truth = {"off": off, "def": dfn, "beta0": b0, "delta0": d0, "date0": date0,
             "rest_effects": rest_effects, "rest_goal_effects": rest_goal_effects,
             "travel_effects": te, "gk_b2b": gk_b2b, "home_by_season": home_by_season,
             "tau_week": tau_week, "tau_summer": tau_summer}
    return gt, teams, truth