"""
hierarchical_xg.py
==================
Track B v1.1: team ratings learned mainly from expected goals (xG), with goals on top.

Home: src/models/stochastic/hierarchical_xg.py

Each game gives two observations per side (regulation, no empty-netters):

  Shot quality   log nu_home = beta0 + beta_home + mu_off[home] - mu_def[away]
                 xG_home ~ Gamma(mean nu_home, shape kappa)
  Goals          log lam_home = log nu_home + delta0 + delta_home
                                + phi_off[home] - phi_def[away] - gamma_away
                 (home goals, away goals) ~ tie-inflated Poisson(lam_home, lam_away; rho)

  mu_off, mu_def    shot-quality ratings (learned mostly from xG, the stable signal)
  phi_off, phi_def  finishing / team saves beyond xG and the goalie; tightly pooled, so
                    they stay near 0 unless the data clearly support them
  gamma             fixed goalie offsets (Phase 3), on goals only: goalies change
                    conversion, not shot quality

Phase 7 found the goals-only Track B (v1) lost to a simple rolling-xG model and was
over-confident out of sample; blending with xG fixed both. This puts xG inside Track B.

Goal-scale ratings for everything downstream. The model also saves
    alpha0     = beta0 + delta0              alpha_home = beta_home + delta_home
    mu_off_g   = mu_off + phi_off            mu_def_g   = mu_def + phi_def
under the v1 names (alpha0, alpha_home, mu_off, mu_def), so game_model, montecarlo,
late_game, score_state, the backtest and predict_game work unchanged. The xG-scale
ratings are saved as xg_mu_off / xg_mu_def.

Optional recency weighting: half_life_days gives each game the weight
0.5 ** (age / half_life) in the likelihood (age measured to `ref_date`).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.stochastic.hierarchical import K_MAX, sample_scores

# Priors
BETA0_PRIOR = (np.log(2.7), 0.5)      # league regulation xG per team-game, log scale
HOME_PRIOR_SD = 0.5
DELTA_PRIOR_SD = 0.2                  # league goals-per-xG on the log scale (near 0)
SIGMA_PRIOR_SD = 0.3                  # spread of shot-quality ratings
SIGMA_PHI_PRIOR_SD = 0.05             # spread of finishing effects: small by design
LOG_KAPPA_PRIOR = (np.log(9.0), 1.0)  # xG per game has CV about 1/3 -> shape about 9
TIE_PRIOR_SD = 0.5

XG_COLS = ("home_xg_reg", "away_xg_reg")
GOAL_COLS = ("home_goals_reg", "away_goals_reg")
XG_FLOOR = 0.05                       # Gamma needs xG > 0; a game below this is a data glitch


def game_weights(gt: pd.DataFrame, half_life_days: float | None, ref_date=None) -> np.ndarray:
    """Recency weights 0.5 ** (age / half_life); all ones if half_life_days is None."""
    if half_life_days is None:
        return np.ones(len(gt))
    ref = pd.Timestamp(ref_date) if ref_date is not None else gt["game_date"].max() + pd.Timedelta(days=1)
    age = (ref - pd.to_datetime(gt["game_date"])).dt.days.to_numpy(float)
    return 0.5 ** (age / half_life_days)


def build_model_xg(gt: pd.DataFrame, teams: list[str], half_life_days: float | None = None,
                   ref_date=None) -> pm.Model:
    """Track B v1.1 for the games in gt (needs xG and regulation-goal columns)."""
    coords = {"team": teams, "game": gt["game_id"].to_numpy()}
    w = game_weights(gt, half_life_days, ref_date)
    hi, ai = gt["home_idx"].to_numpy(), gt["away_idx"].to_numpy()
    g_home, g_away = gt["gamma_home"].to_numpy(), gt["gamma_away"].to_numpy()
    xg_h = np.maximum(gt[XG_COLS[0]].to_numpy(float), XG_FLOOR)
    xg_a = np.maximum(gt[XG_COLS[1]].to_numpy(float), XG_FLOOR)
    y_h, y_a = gt[GOAL_COLS[0]].to_numpy(), gt[GOAL_COLS[1]].to_numpy()
    tied = (y_h == y_a).astype(float)

    with pm.Model(coords=coords) as model:
        # 1. League terms
        beta0 = pm.Normal("beta0", mu=BETA0_PRIOR[0], sigma=BETA0_PRIOR[1])
        beta_home = pm.Normal("beta_home", mu=0.0, sigma=HOME_PRIOR_SD)
        delta0 = pm.Normal("delta0", mu=0.0, sigma=DELTA_PRIOR_SD)
        delta_home = pm.Normal("delta_home", mu=0.0, sigma=DELTA_PRIOR_SD)
        log_kappa = pm.Normal("log_kappa", mu=LOG_KAPPA_PRIOR[0], sigma=LOG_KAPPA_PRIOR[1])
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))

        # 2. Shot-quality ratings (non-centered)
        sigma_off = pm.HalfNormal("sigma_off", sigma=SIGMA_PRIOR_SD)
        sigma_def = pm.HalfNormal("sigma_def", sigma=SIGMA_PRIOR_SD)
        z_off = pm.Normal("z_off", 0.0, 1.0, dims="team")
        z_def = pm.Normal("z_def", 0.0, 1.0, dims="team")
        xg_mu_off = pm.Deterministic("xg_mu_off", sigma_off * z_off, dims="team")
        xg_mu_def = pm.Deterministic("xg_mu_def", sigma_def * z_def, dims="team")

        # 3. Finishing / team-save effects beyond xG (tightly pooled)
        sigma_phi_off = pm.HalfNormal("sigma_phi_off", sigma=SIGMA_PHI_PRIOR_SD)
        sigma_phi_def = pm.HalfNormal("sigma_phi_def", sigma=SIGMA_PHI_PRIOR_SD)
        zp_off = pm.Normal("zp_off", 0.0, 1.0, dims="team")
        zp_def = pm.Normal("zp_def", 0.0, 1.0, dims="team")
        phi_off = pm.Deterministic("phi_off", sigma_phi_off * zp_off, dims="team")
        phi_def = pm.Deterministic("phi_def", sigma_phi_def * zp_def, dims="team")

        # 4. Goal-scale ratings under the v1 names (used by everything downstream)
        pm.Deterministic("alpha0", beta0 + delta0)
        pm.Deterministic("alpha_home", beta_home + delta_home)
        pm.Deterministic("mu_off", xg_mu_off + phi_off, dims="team")
        pm.Deterministic("mu_def", xg_mu_def + phi_def, dims="team")

        # 5. xG likelihood: Gamma with mean nu and shape kappa (weighted)
        log_nu_h = beta0 + beta_home + xg_mu_off[hi] - xg_mu_def[ai]
        log_nu_a = beta0 + xg_mu_off[ai] - xg_mu_def[hi]

        def gamma_logp(x, log_nu):
            # Gamma(shape kappa, rate kappa / nu)
            return (kappa * (log_kappa - log_nu) - pt.gammaln(kappa)
                    + (kappa - 1.0) * np.log(x) - kappa * x * pm.math.exp(-log_nu))
        pm.Potential("xg_loglik", pm.math.sum(w * (gamma_logp(xg_h, log_nu_h) + gamma_logp(xg_a, log_nu_a))))

        # 6. Goals likelihood: tie-inflated Poisson on top of the xG rates (weighted)
        log_lam_h = log_nu_h + delta0 + delta_home + phi_off[hi] - phi_def[ai] - g_away
        log_lam_a = log_nu_a + delta0 + phi_off[ai] - phi_def[hi] - g_home
        lam_h, lam_a = pm.math.exp(log_lam_h), pm.math.exp(log_lam_a)
        tau = pm.Normal("tau", mu=0.0, sigma=TIE_PRIOR_SD)
        rho = pm.Deterministic("rho", pm.math.exp(tau) - 1.0)

        pois = (y_h * log_lam_h - lam_h - pt.gammaln(y_h + 1.0)
                + y_a * log_lam_a - lam_a - pt.gammaln(y_a + 1.0))
        k = np.arange(K_MAX + 1)[:, None]
        lgk = pt.gammaln(k + 1.0)
        log_ph = k * log_lam_h[None, :] - lam_h[None, :] - lgk
        log_pa = k * log_lam_a[None, :] - lam_a[None, :] - lgk
        T = pm.math.exp(log_ph + log_pa).sum(axis=0)
        tie_adj = tau * tied - pm.math.log1p(rho * T)
        pm.Potential("goal_loglik", pm.math.sum(w * (pois + tie_adj)))

    return model


def simulate_league_xg(n_teams: int = 32, games_per_pair: int = 3,
                       sigma_off: float = 0.08, sigma_def: float = 0.08,
                       sigma_phi: float = 0.02, beta0: float = np.log(2.6), beta_home: float = 0.05,
                       delta0: float = 0.03, delta_home: float = 0.0, kappa: float = 9.0,
                       rho: float = 0.5, gamma_sd: float = 0.03, seed: int = 0):
    """Fake league with known shot-quality and finishing effects, in game-table format."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    mo, md = rng.normal(0, sigma_off, n_teams), rng.normal(0, sigma_def, n_teams)
    po, pd_ = rng.normal(0, sigma_phi, n_teams), rng.normal(0, sigma_phi, n_teams)

    home, away = [], []
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            for g in range(games_per_pair):
                h, a = (i, j) if g % 2 == 0 else (j, i)
                home.append(h); away.append(a)
    order = rng.permutation(len(home))
    home, away = np.array(home)[order], np.array(away)[order]
    n = len(home)
    gh, ga = rng.normal(0, gamma_sd, n), rng.normal(0, gamma_sd, n)

    nu_h = np.exp(beta0 + beta_home + mo[home] - md[away])
    nu_a = np.exp(beta0 + mo[away] - md[home])
    xg_h = rng.gamma(kappa, nu_h / kappa)
    xg_a = rng.gamma(kappa, nu_a / kappa)
    lam_h = nu_h * np.exp(delta0 + delta_home + po[home] - pd_[away] - ga)
    lam_a = nu_a * np.exp(delta0 + po[away] - pd_[home] - gh)
    y_h, y_a = sample_scores(lam_h, lam_a, rho, rng)

    gt = pd.DataFrame({
        "game_id": np.arange(n),
        "game_date": pd.Timestamp("2000-10-01") + pd.to_timedelta(np.arange(n) // 8, unit="D"),
        "season": 0, "home": [teams[i] for i in home], "away": [teams[i] for i in away],
        "home_idx": home, "away_idx": away,
        "home_goals": y_h, "away_goals": y_a, "home_goals_reg": y_h, "away_goals_reg": y_a,
        "home_xg_reg": xg_h, "away_xg_reg": xg_a,
        "home_goalie": 0, "away_goalie": 0, "gamma_home": gh, "gamma_away": ga,
    })
    truth = {"beta0": beta0, "beta_home": beta_home, "delta0": delta0, "delta_home": delta_home,
             "kappa": kappa, "rho": rho, "sigma_off": sigma_off, "sigma_def": sigma_def,
             "xg_mu_off": mo, "xg_mu_def": md, "phi_off": po, "phi_def": pd_,
             "mu_off": mo + po, "mu_def": md + pd_}
    return gt, teams, truth