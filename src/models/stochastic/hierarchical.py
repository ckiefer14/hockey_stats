"""
hierarchical.py
===============
Track B: full-league hierarchical Poisson model with tie inflation (PyMC, NUTS).

Home: src/models/stochastic/hierarchical.py

Model (one row of the game table = one game):
    log lam_home = alpha0 + alpha_home + mu_off[home] - mu_def[away] - gamma_away
    log lam_away = alpha0              + mu_off[away] - mu_def[home] - gamma_home
    P(home = h, away = a) = Pois(h | lam_home) * Pois(a | lam_away)
                            * (1 + rho * [h == a]) / (1 + rho * T)
    T   = sum_k Pois(k | lam_home) * Pois(k | lam_away)   (tie chance under plain Poisson)
    rho = exp(tau) - 1, tau ~ N(0, 0.5)                    (rho = 0: independent Poisson)

    Real regulation scores end tied more often than two independent Poisson
    counts allow (23.7% vs about 17%, Phase 5 posterior predictive check);
    rho > 0 adds exactly that. tie_inflation=False gives plain Poisson.

    Goals are REGULATION goals (periods 1-3, no empty-netters). Overtime is
    sudden death, which a Poisson rate can't describe; Phase 6 simulates it separately.

    mu_off = sigma_off * z_off,  z_off ~ N(0, 1)   (non-centered; same for def)
    gamma = fixed goalie offsets from the game table (Phase 3 rating, not estimated)

Positive mu_off = scores more; positive mu_def = allows fewer;
positive gamma = better goalie (opponent scores fewer).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

FIT_SEASONS = ["20242025", "20252026"]

# Priors
ALPHA0_PRIOR = (np.log(3.0), 0.5)     # (mean, sd) of the league away scoring level
HOME_PRIOR_SD = 1.0                   # alpha_home ~ N(0, 1)
SIGMA_PRIOR_SD = 0.3                  # sigma_off, sigma_def ~ HalfNormal(0.3)
MATCHED_SIGMA = 1.0                   # fixed team spread in the matched (no-pooling) check
TIE_PRIOR_SD = 0.5                    # tau = log(1 + rho) ~ N(0, 0.5)

# Tie sums run over k = 0..K_MAX goals; P(one team > 15 in regulation) is negligible
K_MAX = 15

# Which game-table columns the model fits (regulation goals; see docstring)
GOAL_COLS = ("home_goals_reg", "away_goals_reg")

# Sampler defaults
DRAWS, TUNE, CHAINS = 1000, 1000, 4
TARGET_ACCEPT = 0.9


def build_model(gt: pd.DataFrame, teams: list[str],
                matched: bool = False, alpha0: float | None = None,
                alpha_home: float | None = None,
                goal_cols: tuple[str, str] = GOAL_COLS,
                tie_inflation: bool = True) -> pm.Model:
    """Track B model for the games in gt (output of build_game_table).

    goal_cols: (home, away) goal columns to fit. Default is regulation goals;
    the Track A cross-check passes ("home_goals", "away_goals") because Track A
    was fit on periods 1-4.

    matched=True is the Track A cross-check: no pooling (sigmas fixed at
    MATCHED_SIGMA), alpha0 and alpha_home fixed to the given values, goalie
    offsets ignored and no tie inflation (Track A has neither).

    tie_inflation=False gives the plain independent-Poisson model (the baseline).
    """
    if matched:
        tie_inflation = False
    if matched and (alpha0 is None or alpha_home is None):
        raise ValueError("matched=True needs alpha0 and alpha_home (Track A's fixed values).")

    coords = {"team": teams, "game": gt["game_id"].to_numpy()}

    with pm.Model(coords=coords) as model:
        # 1. Data arrays: which teams played, and the goalie offsets
        home_idx = pm.Data("home_idx", gt["home_idx"].to_numpy(), dims="game")
        away_idx = pm.Data("away_idx", gt["away_idx"].to_numpy(), dims="game")
        if matched:
            gamma_home = np.zeros(len(gt))
            gamma_away = np.zeros(len(gt))
        else:
            gamma_home = pm.Data("gamma_home", gt["gamma_home"].to_numpy(), dims="game")
            gamma_away = pm.Data("gamma_away", gt["gamma_away"].to_numpy(), dims="game")

        # 2. League terms: estimated, or fixed numbers in matched mode
        if matched:
            a0, a_home = alpha0, alpha_home
        else:
            a0 = pm.Normal("alpha0", mu=ALPHA0_PRIOR[0], sigma=ALPHA0_PRIOR[1])
            a_home = pm.Normal("alpha_home", mu=0.0, sigma=HOME_PRIOR_SD)

        # 3. How spread out teams are: estimated (pooling) or fixed (no pooling)
        if matched:
            sigma_off, sigma_def = MATCHED_SIGMA, MATCHED_SIGMA
        else:
            sigma_off = pm.HalfNormal("sigma_off", sigma=SIGMA_PRIOR_SD)
            sigma_def = pm.HalfNormal("sigma_def", sigma=SIGMA_PRIOR_SD)

        # 4. Team effects, non-centered: sample standard normals, then scale.
        #    Same model as mu ~ N(0, sigma^2), but avoids the "funnel" when sigma is small.
        z_off = pm.Normal("z_off", mu=0.0, sigma=1.0, dims="team")
        z_def = pm.Normal("z_def", mu=0.0, sigma=1.0, dims="team")
        mu_off = pm.Deterministic("mu_off", sigma_off * z_off, dims="team")
        mu_def = pm.Deterministic("mu_def", sigma_def * z_def, dims="team")

        # 5. Log scoring rate for each side of each game.
        #    The home team shoots on the AWAY goalie, so it faces gamma_away.
        log_lam_home = a0 + a_home + mu_off[home_idx] - mu_def[away_idx] - gamma_away
        log_lam_away = a0 + mu_off[away_idx] - mu_def[home_idx] - gamma_home

        # 6. Likelihood: observed goals (regulation by default, no empty-netters)
        pm.Poisson("home_goals", mu=pm.math.exp(log_lam_home),
                   observed=gt[goal_cols[0]].to_numpy(), dims="game")
        pm.Poisson("away_goals", mu=pm.math.exp(log_lam_away),
                   observed=gt[goal_cols[1]].to_numpy(), dims="game")

        # 7. Tie inflation: multiply tied scores by (1 + rho), then renormalize.
        #    The two Poisson terms above stay as they are; this adds the correction
        #    log(1 + rho * [h == a]) - log(1 + rho * T) for every game.
        if tie_inflation:
            tau = pm.Normal("tau", mu=0.0, sigma=TIE_PRIOR_SD)
            rho = pm.Deterministic("rho", pm.math.exp(tau) - 1.0)

            k = np.arange(K_MAX + 1)[:, None]                        # (K+1, 1)
            tied = (gt[goal_cols[0]].to_numpy() == gt[goal_cols[1]].to_numpy()).astype(float)

            # log Pois(k | lam) = k * log_lam - lam - log(k!)
            lgk = pt.gammaln(k + 1.0)
            log_ph = k * log_lam_home[None, :] - pm.math.exp(log_lam_home)[None, :] - lgk
            log_pa = k * log_lam_away[None, :] - pm.math.exp(log_lam_away)[None, :] - lgk
            T = pm.math.exp(log_ph + log_pa).sum(axis=0)              # (games,)

            pm.Potential("tie_adjust", pm.math.sum(tau * tied - pm.math.log1p(rho * T)))

    return model


def sample(model: pm.Model, seed: int = 42, draws: int = DRAWS, tune: int = TUNE,
           chains: int = CHAINS, target_accept: float = TARGET_ACCEPT, progressbar: bool = True):
    """Run NUTS. cores=1 runs chains one after another (safe on Windows)."""
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                         target_accept=target_accept, random_seed=seed,
                         progressbar=progressbar)


def sample_scores(lam_home, lam_away, rho, rng, k_max: int = K_MAX):
    """Draw (home, away) scores from the tie-inflated model. Works on arrays.

    rho >= 0: a mixture. With probability rho*T / (1 + rho*T) the game is forced to a
    tie, with the tied score k drawn from Pois(k|lam_home) * Pois(k|lam_away);
    otherwise both scores are independent Poisson. This reproduces the model exactly.
    rho < 0: independent draws, keeping a tie only with probability 1 + rho (redraw otherwise).
    """
    lam_home, lam_away, rho = np.broadcast_arrays(np.asarray(lam_home, float),
                                                  np.asarray(lam_away, float),
                                                  np.asarray(rho, float))
    shape = lam_home.shape
    h, a = rng.poisson(lam_home), rng.poisson(lam_away)

    # Positive rho: mixture with a "forced tie" component
    rho_pos = np.clip(rho, 0.0, None)
    k = np.arange(k_max + 1).reshape(-1, *([1] * len(shape)))
    pk = stats.poisson.pmf(k, lam_home) * stats.poisson.pmf(k, lam_away)    # (K+1, ...)
    T = pk.sum(axis=0)
    force_tie = rng.random(shape) < rho_pos * T / (1.0 + rho_pos * T)
    cdf = np.cumsum(pk / T, axis=0)
    k_tie = np.minimum((rng.random(shape) > cdf).sum(axis=0), k_max)
    h = np.where(force_tie, k_tie, h)
    a = np.where(force_tie, k_tie, a)

    # Negative rho: thin out ties by redrawing
    redo = (rho < 0) & (h == a) & (rng.random(shape) > 1.0 + rho)
    while redo.any():
        h[redo], a[redo] = rng.poisson(lam_home[redo]), rng.poisson(lam_away[redo])
        redo = redo & (h == a) & (rng.random(shape) > 1.0 + rho)
    return h, a


def simulate_league(n_teams: int = 32, games_per_pair: int = 3,
                    sigma_off: float = 0.12, sigma_def: float = 0.12,
                    alpha0: float = 1.03, alpha_home: float = 0.07,
                    gamma_sd: float = 0.03, rho: float = 0.5, seed: int = 0) -> tuple[pd.DataFrame, list[str], dict]:
    """Fake league with known ratings, in the same format as build_game_table.

    Every pair of teams plays games_per_pair times, home ice alternating.
    Goalie offsets are random known values (the model treats them as fixed).
    Scores come from the tie-inflated model with the given rho (0 = plain Poisson).
    Returns (gt, teams, truth) where truth holds the true parameter values.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]

    # 1. True team effects
    mu_off = rng.normal(0.0, sigma_off, n_teams)
    mu_def = rng.normal(0.0, sigma_def, n_teams)

    # 2. Schedule: every pair, alternating home
    home, away = [], []
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            for g in range(games_per_pair):
                h, a = (i, j) if g % 2 == 0 else (j, i)
                home.append(h)
                away.append(a)
    home, away = np.array(home), np.array(away)
    order = rng.permutation(len(home))            # shuffle so dates are mixed
    home, away = home[order], away[order]
    n = len(home)

    # 3. Known goalie offsets, then goals from the same formulas as the model
    gamma_home = rng.normal(0.0, gamma_sd, n)
    gamma_away = rng.normal(0.0, gamma_sd, n)
    lam_home = np.exp(alpha0 + alpha_home + mu_off[home] - mu_def[away] - gamma_away)
    lam_away = np.exp(alpha0 + mu_off[away] - mu_def[home] - gamma_home)

    home_goals, away_goals = sample_scores(lam_home, lam_away, rho, rng)

    gt = pd.DataFrame({
        "game_id": np.arange(n),
        "game_date": pd.Timestamp("2000-10-01") + pd.to_timedelta(np.arange(n) // 8, unit="D"),
        "season": 0,
        "home": [teams[i] for i in home],
        "away": [teams[i] for i in away],
        "home_idx": home,
        "away_idx": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_goals_reg": 0,                       # filled just below (same draws)
        "away_goals_reg": 0,
        "home_goalie": 0,
        "away_goalie": 0,
        "gamma_home": gamma_home,
        "gamma_away": gamma_away,
    })
    gt["home_goals_reg"] = gt["home_goals"]       # no overtime in the fake league
    gt["away_goals_reg"] = gt["away_goals"]
    truth = {"alpha0": alpha0, "alpha_home": alpha_home, "rho": rho,
             "sigma_off": sigma_off, "sigma_def": sigma_def,
             "mu_off": mu_off, "mu_def": mu_def}
    return gt, teams, truth


def team_ratings(idata, teams: list[str], level: float = 0.90) -> pd.DataFrame:
    """Per-team posterior summary, sorted best to worst by net rating.

    off / def / net are on the log-rate scale: +0.10 is about 10% more goals
    scored (off) or fewer allowed (def). net = off + def.
    """
    post = idata.posterior
    off = post["mu_off"].values.reshape(-1, len(teams))     # (chains*draws, teams)
    dfn = post["mu_def"].values.reshape(-1, len(teams))
    net = off + dfn
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2

    out = pd.DataFrame({"team": teams})
    for name, s in [("off", off), ("def", dfn), ("net", net)]:
        out[f"{name}_mean"] = s.mean(axis=0)
        out[f"{name}_lo"] = np.quantile(s, lo, axis=0)
        out[f"{name}_hi"] = np.quantile(s, hi, axis=0)
    return out.sort_values("net_mean", ascending=False).reset_index(drop=True)