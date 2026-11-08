"""
xg_layer.py
===========
v2.1: a hierarchical layer on top of the LightGBM xG model. LightGBM's out-of-sample
prediction (shot_xg, leave-one-season-out) enters as a fixed offset; only the random
effects are estimated:

    logit p_i = logit(xg_i) + beta[shooter] - gamma[goalie] + delta[rink] + eps[season x type]
    goal_i ~ Bernoulli(p_i)

    beta   shooter finishing talent        ~ N(0, sigma_beta^2)
    gamma  goalie skill (+ = better)       ~ N(0, sigma_gamma^2)
    delta  rink recording effect           ~ N(0, sigma_delta^2)
    eps    season x shot-type conversion   ~ N(0, sigma_eps^2)   (redirect drift)

Shooters and goalies with few shots are shrunk toward 0 (league average relative to
LightGBM); high-volume ones keep their own estimate.

Fitting (fit_laplace): the Laplace approximation used for mixed-effects logistic models
(as lme4::glmer). For given sigmas, Newton steps find the posterior mode of the effects on a
sparse design (560k shots x ~1,750 effects, seconds); the sigmas maximize the Laplace
marginal likelihood times their half-normal priors. Each effect gets a Gaussian posterior
N(mode, H^-1), which with this many shots is very close to the exact posterior. Plain NUTS
on 560k Bernoulli rows takes many hours; build_xg_layer (PyMC) is kept to check the Laplace
fit against NUTS on a subsample (compare_with_nuts).

Home: src/models/stochastic/xg_layer.py
(Not to be confused with hierarchical_xg.py, Track B v1.1, the game-level model.)

Rink = the building: the home team's abbreviation as the schedule gives it (ARI and UTA are
different rinks), with neutral-site games (a venue that is not the home team's usual one
that season) as one shared "NEUTRAL" rink.

Used by src/pipelines/fit_xg_layer.py (fit, held-out test, save). Shots in a held-out
season with a shooter / goalie / rink / season x type unseen in training get 0 for that term.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

XG_FLOOR = 1e-4                 # LightGBM can output tiny xG; keep the logit finite
SIGMA_PRIORS = {"beta": 0.3, "gamma": 0.3, "delta": 0.2, "eps": 0.2}   # half-normal sds (logit scale)
SHOT_GROUPS = {"wrist": "wrist-snap", "snap": "wrist-snap", "tip-in": "redirect", "deflected": "redirect",
               "slap": "slap", "backhand": "backhand", "wrap-around": "wrap-around"}
TERMS = ["beta", "gamma", "delta", "eps"]
TERM_COLS = {"beta": "shooter", "gamma": "goalie", "delta": "rink", "eps": "season_type"}
TERM_NAMES = {"beta": "shooter finishing", "gamma": "goalie skill", "delta": "rink", "eps": "season x shot type"}
TARGET_ACCEPT = 0.9


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_shots(conn, seasons: list[str]) -> pd.DataFrame:
    """Every shot scored in shot_xg for the seasons: offset, goal flag and the four group keys.

    Columns: game_id, event_id, season, game_date, xg, goal, shooter, goalie, rink, shot_group,
             season_type, logit_xg
    """
    seasons = [int(s) for s in seasons]
    q = f"""
        SELECT  x.game_id, x.event_id, x.season, x.xg, s.game_date,
                CASE WHEN se.event_type = 'goal' THEN 1 ELSE 0 END AS goal,
                se.shooting_player_id AS shooter, se.goalie_id AS goalie, se.shot_type,
                s.home_team_abbrev AS site
        FROM    shot_xg x
        JOIN    shot_events se ON se.game_id = x.game_id AND se.event_id = x.event_id
        JOIN    schedule s     ON s.game_id = x.game_id
        WHERE   x.season IN ({",".join("?" * len(seasons))}) AND x.xg IS NOT NULL
    """
    df = pd.read_sql(q, conn, params=seasons)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(str)
    df["shot_group"] = df["shot_type"].map(SHOT_GROUPS).fillna("other")
    df["season_type"] = df["season"] + ":" + df["shot_group"]
    df["rink"] = df["site"]
    neutral = _neutral_games(conn, df)
    df.loc[df["game_id"].isin(neutral), "rink"] = "NEUTRAL"
    df["shooter"] = df["shooter"].fillna(-1).astype(int)
    df["goalie"] = df["goalie"].fillna(-1).astype(int)       # -1: no goalie recorded (rare)
    p = df["xg"].clip(XG_FLOOR, 1 - XG_FLOOR)
    df["logit_xg"] = np.log(p / (1 - p))
    return df.drop(columns=["shot_type", "site"]).sort_values(["game_date", "game_id", "event_id"]).reset_index(drop=True)


def _neutral_games(conn, shots: pd.DataFrame) -> set:
    """Games at a venue that is not the home team's usual one that season (from games.venue)."""
    try:
        v = pd.read_sql("SELECT id AS game_id, venue FROM games", conn)
    except Exception:
        return set()
    g = shots.drop_duplicates("game_id")[["game_id", "season", "site"]].merge(v, on="game_id").dropna(subset=["venue"])
    if g.empty:
        return set()
    usual = g.groupby(["season", "site"])["venue"].agg(lambda x: x.mode().iloc[0]).rename("usual")
    g = g.join(usual, on=["season", "site"])
    return set(g.loc[g["venue"] != g["usual"], "game_id"])


def index_groups(df: pd.DataFrame, levels: dict | None = None) -> tuple[dict, dict]:
    """Integer codes for each term's group key. levels: the training levels (for held-out
    data: unseen levels get -1). Returns (codes, levels)."""
    if levels is None:
        levels = {t: np.sort(df[c].unique()) for t, c in TERM_COLS.items()}
    codes = {}
    for t, c in TERM_COLS.items():
        lookup = pd.Series(np.arange(len(levels[t])), index=levels[t])
        codes[t] = df[c].map(lookup).fillna(-1).astype(int).to_numpy()
    return codes, levels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_xg_layer(df: pd.DataFrame, terms: list[str] = TERMS) -> tuple[pm.Model, dict]:
    """PyMC model for the shots in df. Returns (model, levels). terms: which effects to include."""
    codes, levels = index_groups(df)
    coords = {TERM_COLS[t]: [str(x) for x in levels[t]] for t in terms}
    y = df["goal"].to_numpy().astype("int8")
    off = df["logit_xg"].to_numpy()
    with pm.Model(coords=coords) as model:
        eta = pt.as_tensor_variable(off)
        for t in terms:
            sigma = pm.HalfNormal(f"sigma_{t}", SIGMA_PRIORS[t])
            z = pm.Normal(f"z_{t}", 0.0, 1.0, dims=TERM_COLS[t])
            eff = pm.Deterministic(t, sigma * z, dims=TERM_COLS[t])
            sign = -1.0 if t == "gamma" else 1.0          # better goalie = fewer goals
            eta = eta + sign * eff[codes[t]]
        pm.Bernoulli("goal", logit_p=eta, observed=y)
    return model, levels


def sample_layer(model: pm.Model, draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 42,
                 progressbar: bool = True):
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=1, target_accept=TARGET_ACCEPT,
                         random_seed=seed, progressbar=progressbar, idata_kwargs={"log_likelihood": False})


# ---------------------------------------------------------------------------
# Laplace fit (the production fitter)
# ---------------------------------------------------------------------------

def _design(df: pd.DataFrame, terms: list[str]):
    """Sparse design Z (shots x all effects, gamma columns signed -1) and the block slices."""
    codes, levels = index_groups(df)
    blocks, cols, start = {}, [], 0
    n = len(df)
    rows, cidx, vals = [], [], []
    for t in terms:
        k = len(levels[t])
        blocks[t] = slice(start, start + k)
        ok = codes[t] >= 0
        rows.append(np.nonzero(ok)[0]); cidx.append(codes[t][ok] + start)
        vals.append(np.full(ok.sum(), -1.0 if t == "gamma" else 1.0))
        start += k
    Z = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cidx))), shape=(n, start))
    return Z, blocks, levels


def _mode(Z, ZT, y, off, prec, u0, tol=1e-8, max_iter=50):
    """Posterior mode of the effects for fixed prior precisions (Newton). Returns (u, H factor, loglik, H)."""
    u = u0.copy()
    for _ in range(max_iter):
        eta = off + Z @ u
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = ZT @ (y - p) - prec * u
        W = p * (1 - p)
        H = (ZT @ sp.diags(W) @ Z).toarray()
        H[np.diag_indices_from(H)] += prec
        cf = cho_factor(H, lower=True)
        step = cho_solve(cf, grad)
        u = u + step
        if np.max(np.abs(step)) < tol:
            break
    eta = off + Z @ u
    loglik = float(np.sum(y * eta - np.logaddexp(0, eta)))
    p = 1.0 / (1.0 + np.exp(-eta))
    H = (ZT @ sp.diags(p * (1 - p)) @ Z).toarray()
    H[np.diag_indices_from(H)] += prec
    return u, cho_factor(H, lower=True), loglik


def fit_laplace(df: pd.DataFrame, terms: list[str] = TERMS, verbose: bool = True,
                sigma0: dict | None = None, intervals: bool = True) -> dict:
    """Fit the layer by Laplace approximation (empirical-Bayes sigmas at their posterior mode).

    Returns {"sigma": {t: value}, "sigma_ci": {t: (lo, hi)} (approx. 90%, from the curvature of
    the marginal likelihood in log sigma), "effects": {t: DataFrame(level, mean, sd, shots, goals, xg)},
    "levels", "blocks", "cov_gamma" (posterior covariance of the goalie effects, for draws),
    "log_marginal", "n_shots"}.
    sigma0: starting sigmas (e.g. the previous weekly fit's), for speed. intervals=False skips
    the sigma intervals (4-8 extra evaluations) when only the effects are needed.
    """
    Z, blocks, levels = _design(df, terms)
    ZT = Z.T.tocsr()
    y = df["goal"].to_numpy(float)
    off = df["logit_xg"].to_numpy(float)
    K = Z.shape[1]
    state = {"u": np.zeros(K)}

    def neg_obj(log_s):
        sig = {t: float(np.exp(v)) for t, v in zip(terms, log_s)}
        prec = np.concatenate([np.full(blocks[t].stop - blocks[t].start, 1 / sig[t] ** 2) for t in terms])
        u, cf, ll = _mode(Z, ZT, y, off, prec, state["u"])
        state["u"] = u
        logdet_H = 2 * np.sum(np.log(np.diag(cf[0])))
        logdet_D = -np.sum(np.log(prec))
        lm = ll - 0.5 * np.sum(prec * u ** 2) - 0.5 * logdet_D - 0.5 * logdet_H
        # half-normal prior on each sigma, plus the log-sigma Jacobian (optimizing in log sigma)
        lp = sum(-0.5 * (sig[t] / SIGMA_PRIORS[t]) ** 2 + np.log(sig[t]) for t in terms)
        return -(lm + lp)

    x0 = np.log([(sigma0 or {}).get(t, 0.1) for t in terms])
    bounds = [(np.log(1e-3), np.log(2.0))] * len(terms)
    res = minimize(neg_obj, x0, method="L-BFGS-B", bounds=bounds, options={"eps": 1e-4, "maxiter": 200})
    log_s = res.x
    # approx. 90% intervals for sigma from the numerical curvature in log sigma
    h = 0.05
    ci = {}
    f0 = neg_obj(log_s)
    for i, t in enumerate(terms if intervals else []):
        e = np.zeros(len(terms)); e[i] = h
        curv = (neg_obj(log_s + e) - 2 * f0 + neg_obj(log_s - e)) / h ** 2
        se = 1 / np.sqrt(curv) if curv > 0 else np.inf
        ci[t] = (float(np.exp(log_s[i] - 1.645 * se)), float(np.exp(log_s[i] + 1.645 * se)))
    for t in terms:
        ci.setdefault(t, (float("nan"), float("nan")))
    sig = {t: float(np.exp(v)) for t, v in zip(terms, log_s)}
    prec = np.concatenate([np.full(blocks[t].stop - blocks[t].start, 1 / sig[t] ** 2) for t in terms])
    u, cf, ll = _mode(Z, ZT, y, off, prec, state["u"])
    Hinv = cho_solve(cf, np.eye(K))
    sd = np.sqrt(np.diag(Hinv))
    codes, _ = index_groups(df, levels)
    effects = {}
    for t in terms:
        b = blocks[t]
        k = b.stop - b.start
        cnt = np.bincount(codes[t][codes[t] >= 0], minlength=k)
        gls = np.bincount(codes[t][codes[t] >= 0], weights=y[codes[t] >= 0], minlength=k)
        xgs = np.bincount(codes[t][codes[t] >= 0], weights=df["xg"].to_numpy()[codes[t] >= 0], minlength=k)
        effects[t] = pd.DataFrame({"level": levels[t], "mean": u[b], "sd": sd[b], "shots": cnt,
                                   "goals": gls, "xg": xgs})
    out = {"sigma": sig, "sigma_ci": ci, "effects": effects, "levels": levels, "blocks": blocks,
           "cov_gamma": Hinv[blocks["gamma"], blocks["gamma"]] if "gamma" in blocks else None,
           "log_marginal": float(-f0), "n_shots": len(df), "converged": bool(res.success), "terms": terms}
    if verbose and intervals:
        print(f"Laplace fit: {len(df):,} shots, {K:,} effects; " + ", ".join(
            f"sigma_{t} {sig[t]:.3f} [{ci[t][0]:.3f}, {ci[t][1]:.3f}]" for t in terms)
            + ("" if res.success else f"  (optimizer: {res.message})"))
    return out


def effect_draws(fit: dict, term: str, n: int, seed: int = 0) -> np.ndarray:
    """n joint posterior draws of one term's effects (n x levels). Exact block covariance for
    gamma (saved), independent normals otherwise."""
    rng = np.random.default_rng(seed)
    e = fit["effects"][term]
    if term == "gamma" and fit.get("cov_gamma") is not None:
        return rng.multivariate_normal(e["mean"].to_numpy(), fit["cov_gamma"], size=n, method="cholesky")
    return e["mean"].to_numpy() + e["sd"].to_numpy() * rng.standard_normal((n, len(e)))


# ---------------------------------------------------------------------------
# Effects and prediction
# ---------------------------------------------------------------------------

def effect_means(idata, levels: dict) -> dict:
    """Posterior mean and sd of each term's effects: {term: DataFrame(level, mean, sd)}."""
    post = idata.posterior
    out = {}
    for t in TERMS:
        if t not in post:
            continue
        x = post[t].values.reshape(-1, post[t].shape[-1])
        out[t] = pd.DataFrame({"level": levels[t], "mean": x.mean(axis=0), "sd": x.std(axis=0)})
    return out


def adjusted_logit(df: pd.DataFrame, effects: dict, levels: dict, terms: list[str] = TERMS,
                   carry_eps: bool = True) -> np.ndarray:
    """logit(xg) plus the posterior-mean effects, for any shots (unseen groups get 0).

    carry_eps: for a season x type not in training (a new season), use the latest training
    season's value for that shot type (conversion drift tends to persist), else 0.
    """
    codes, _ = index_groups(df, levels)
    eta = df["logit_xg"].to_numpy().copy()
    for t in terms:
        if t not in effects:
            continue
        m = effects[t]["mean"].to_numpy()
        val = np.where(codes[t] >= 0, m[np.clip(codes[t], 0, None)], 0.0)
        if t == "eps" and carry_eps:
            last = _latest_eps(effects["eps"])
            unseen = codes[t] < 0
            val[unseen] = df.loc[unseen, "shot_group"].map(last).fillna(0.0).to_numpy()
        eta += -val if t == "gamma" else val
    return eta


def _latest_eps(eps: pd.DataFrame) -> dict:
    """{shot group: eps of the latest season in training}."""
    e = eps.assign(season=eps["level"].str.split(":").str[0], group=eps["level"].str.split(":").str[1])
    latest = e["season"].max()
    return e[e["season"] == latest].set_index("group")["mean"].to_dict()


def shot_log_loss(eta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-shot log loss from logits."""
    return np.logaddexp(0, eta) - y * eta


# ---------------------------------------------------------------------------
# Synthetic shots (recovery test)
# ---------------------------------------------------------------------------

def simulate_goals(df: pd.DataFrame, truth_sigmas: dict, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Replace df's goals with draws from the layer model with known effects on top of the
    real offsets. Returns (df with new 'goal', true effects {term: Series by level})."""
    rng = np.random.default_rng(seed)
    codes, levels = index_groups(df)
    eta = df["logit_xg"].to_numpy().copy()
    truth = {}
    for t in TERMS:
        s = truth_sigmas.get(t, 0.0)
        eff = rng.normal(0.0, s, len(levels[t])) if s > 0 else np.zeros(len(levels[t]))
        truth[t] = pd.Series(eff, index=levels[t])
        eta += (-eff if t == "gamma" else eff)[codes[t]]
    out = df.copy()
    out["goal"] = (rng.random(len(df)) < 1 / (1 + np.exp(-eta))).astype(int)
    return out, truth