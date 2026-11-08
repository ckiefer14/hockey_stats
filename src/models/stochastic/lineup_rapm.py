"""
lineup_rapm.py
==============
v2.2: Bayesian RAPM. The same 5v5 stint regression as the ridge reference
(src/models/deterministic/rapm.py), with the prior spreads learned from the data and an exact
posterior:

    xGF/60_i = intercept + covariates + sum_offense O_p + sum_defense D_q [+ team-season terms] + e_i
    e_i ~ N(0, sigma^2 / minutes_i)
    O_p ~ N(0, tau_off[F or D]^2),  D_q ~ N(0, tau_def[F or D]^2)
    team-season system effects (optional): T_off ~ N(0, tau_team_off^2), T_def ~ N(0, tau_team_def^2)

Home: src/models/stochastic/lineup_rapm.py

With a Gaussian likelihood and Gaussian priors the posterior of all effects is exactly
Gaussian given the spreads: N(m, H^-1) with H = X'WX / sigma^2 + Lambda, m = H^-1 X'Wy / sigma^2.
The spreads (sigma and the taus) are set at the mode of the exact marginal likelihood times
half-normal priors (empirical Bayes; their own uncertainty is small with ~1M stints and is
not propagated). So no MCMC is needed and the fit takes about a minute.

Compared with ridge: ridge uses one penalty lambda for every player (lambda = sigma^2 / tau^2
with one tau); here forwards and defensemen, offense and defense each get their own tau, and
every rating comes with a posterior sd and correlations (linemates who always play together
get large, negatively correlated uncertainty instead of an arbitrary split).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from src.models.deterministic import rapm

FLAT_PREC = 1e-6                 # intercept and covariates: effectively flat
OTHER_TAU = 0.5                  # the shared "other player" columns: a loose fixed prior
TAU_PRIOR_SD = 0.5               # half-normal hyperprior on each tau (xG/60)


def design(st: pd.DataFrame, positions: pd.Series, min_toi: float = 100.0, teams: bool = False):
    """rapm.design plus (optionally) team-season offense/defense columns, and the prior group
    of every column. positions: player_id -> 'C'/'L'/'R'/'D' (unknown = forward)."""
    X, y, w, groups, info = rapm.design(st, min_toi=min_toi)
    P = info["P"]
    is_d = pd.Series(info["players"]).map(positions).eq("D").to_numpy()
    grp = (["D_off" if d else "F_off" for d in is_d] + ["other_off"]
           + ["D_def" if d else "F_def" for d in is_d] + ["other_def"]
           + ["flat"] * (1 + len(rapm.COVARIATES)))
    names = list(info["names"])
    if teams:
        n_st = len(st)
        off_team = np.concatenate([st["home_team"] + ":" + st["season"].astype(str),
                                   st["away_team"] + ":" + st["season"].astype(str)])
        def_team = np.concatenate([st["away_team"] + ":" + st["season"].astype(str),
                                   st["home_team"] + ":" + st["season"].astype(str)])
        levels = np.unique(np.concatenate([off_team, def_team]))
        k = pd.Series(np.arange(len(levels)), index=levels)
        n = X.shape[0]
        T = len(levels)
        Xt = sp.csr_matrix((np.ones(2 * n), (np.concatenate([np.arange(n), np.arange(n)]),
                                             np.concatenate([k.reindex(off_team).to_numpy(),
                                                             T + k.reindex(def_team).to_numpy()]))), shape=(n, 2 * T))
        X = sp.hstack([X, Xt]).tocsr()
        grp += ["team_off"] * T + ["team_def"] * T
        names += [f"Toff_{l}" for l in levels] + [f"Tdef_{l}" for l in levels]
        info["team_levels"] = levels
        assert n == 2 * n_st
    info["groups"] = np.array(grp)
    info["names"] = names
    return X, y, w, groups, info


def free_groups(info) -> list[str]:
    g = set(info["groups"]) - {"flat", "other_off", "other_def"}
    order = ["F_off", "D_off", "F_def", "D_def", "team_off", "team_def"]
    return [x for x in order if x in g]


def _prec(theta: dict, groups: np.ndarray) -> np.ndarray:
    lam = np.full(len(groups), FLAT_PREC)
    for g, tau in theta.items():
        if g == "sigma" or g.startswith("_"):         # "_converged" etc. are flags, not spreads
            continue
        lam[groups == g] = 1.0 / float(tau) ** 2
    lam[groups == "other_off"] = lam[groups == "other_def"] = 1.0 / OTHER_TAU ** 2
    return lam


def posterior(A, c, theta: dict, groups):
    """Posterior mean and Cholesky factor of H for given spreads."""
    s2 = theta["sigma"] ** 2
    lam = _prec(theta, groups)
    H = A / s2
    H[np.diag_indices_from(H)] += lam
    cf = cho_factor(H, lower=True)
    m = cho_solve(cf, c / s2)
    return m, cf, lam


def log_marginal(A, c, yy, n, slw, theta: dict, groups) -> float:
    """Exact log marginal likelihood of the weighted Gaussian model (constants dropped)."""
    s2 = theta["sigma"] ** 2
    m, cf, lam = posterior(A, c, theta, groups)
    logdet_H = 2 * np.sum(np.log(np.diag(cf[0])))
    quad = yy / s2 - (c / s2) @ m
    return float(-0.5 * (n * np.log(s2) - slw - np.sum(np.log(lam)) + logdet_H + quad))


def fit_spreads(A, c, yy, n, slw, info, x0: dict | None = None, verbose=False) -> dict:
    """Spreads at the mode of marginal likelihood x half-normal priors (optimized in log)."""
    free = free_groups(info)
    keys = ["sigma"] + free
    start = {"sigma": 10.0}
    start.update({g: 0.15 for g in free})
    if x0:
        start.update({k: v for k, v in x0.items() if k in keys})
    groups = info["groups"]

    def neg(v):
        th = dict(zip(keys, np.exp(v)))
        lp = sum(-0.5 * (th[g] / TAU_PRIOR_SD) ** 2 + np.log(th[g]) for g in free)
        return -(log_marginal(A, c, yy, n, slw, th, groups) + lp)

    v0 = np.log([start[k] for k in keys])
    bounds = [(np.log(0.5), np.log(100.0))] + [(np.log(1e-3), np.log(2.0))] * len(free)
    res = minimize(neg, v0, method="L-BFGS-B", bounds=bounds, options={"eps": 1e-4, "maxiter": 300})
    th = dict(zip(keys, map(float, np.exp(res.x))))
    th["_converged"] = bool(res.success)
    if verbose:
        print("  spreads: " + ", ".join(f"{k} {v:.3f}" for k, v in th.items() if not k.startswith("_")))
    return th


def gram_parts(X, y, w, fold=None, k=None):
    if fold is not None:
        idx = np.nonzero(fold == k)[0]
        X, y, w = X[idx], y[idx], w[idx]
    A, c, yy, sw = rapm.gram(X, y, w)
    return A, c, yy, len(y), float(np.sum(np.log(w)))


def cross_validate(X, y, w, games, info, ridge_lams, folds=5, seed=0) -> pd.DataFrame:
    """Held-out weighted MSE on the same game folds for: controls only, ridge (best lambda chosen
    on the held-out error, as in fit_rapm, so slightly flattering to ridge), and the Bayesian
    model with its spreads re-fit inside every training fold (honest)."""
    rng = np.random.default_rng(seed)
    gl = np.unique(games)
    fold = pd.Series(rng.integers(0, folds, len(gl)), index=gl).reindex(games).to_numpy()
    parts = [gram_parts(X, y, w, fold, k) for k in range(folds)]
    tot = [sum(p[i] for p in parts) for i in range(5)]
    pen = np.isin(info["groups"], ["F_off", "D_off", "F_def", "D_def", "other_off", "other_def", "team_off", "team_def"])
    flat = ~pen
    res = {"controls only": 0.0, "Bayes": 0.0}
    ridge = {lam: 0.0 for lam in ridge_lams}
    wsum = 0.0
    th = None
    for k in range(folds):
        A_te, c_te, yy_te, n_te, _ = parts[k]
        A_tr, c_tr, yy_tr, n_tr, slw_tr = (tot[0] - A_te, tot[1] - c_te, tot[2] - yy_te, tot[3] - n_te,
                                           tot[4] - parts[k][4])
        wsum += float(A_te[info["c_int"], info["c_int"]])           # sum of weights (intercept column)
        b = np.zeros(len(c_tr))
        b[flat] = rapm.solve(A_tr[np.ix_(flat, flat)], c_tr[flat], 0.0, np.zeros(flat.sum(), bool))
        res["controls only"] += rapm.weighted_sse(b, A_te, c_te, yy_te)
        for lam in ridge_lams:
            b = rapm.solve(A_tr, c_tr, float(lam), pen)
            ridge[lam] += rapm.weighted_sse(b, A_te, c_te, yy_te)
        th = fit_spreads(A_tr, c_tr, yy_tr, n_tr, slw_tr, info, x0=th)
        m, _, _ = posterior(A_tr, c_tr, th, info["groups"])
        res["Bayes"] += rapm.weighted_sse(m, A_te, c_te, yy_te)
    best = min(ridge, key=ridge.get)
    out = [{"model": "controls only", "heldout_mse": res["controls only"] / wsum},
           {"model": f"ridge (lambda {best:,.0f})", "heldout_mse": ridge[best] / wsum},
           {"model": "Bayes", "heldout_mse": res["Bayes"] / wsum}]
    return pd.DataFrame(out)


def fit(X, y, w, info, verbose=True) -> dict:
    A, c, yy, n, slw = gram_parts(X, y, w)
    th = fit_spreads(A, c, yy, n, slw, info, verbose=verbose)
    m, cf, _ = posterior(A, c, th, info["groups"])
    cov = cho_solve(cf, np.eye(len(m)))
    return {"theta": th, "mean": m, "cov": cov, "info": info}


def player_table(f: dict) -> pd.DataFrame:
    info, m, cov = f["info"], f["mean"], f["cov"]
    P = info["P"]
    io, idf = np.arange(P), np.arange(P + 1, 2 * P + 1)
    var_o, var_d = cov[io, io], cov[idf, idf]
    cov_od = cov[io, idf]
    return pd.DataFrame({"player_id": info["players"], "toi_5v5": info["minutes"],
                         "off": m[io], "off_sd": np.sqrt(var_o), "def": -m[idf], "def_sd": np.sqrt(var_d),
                         "net": m[io] - m[idf], "net_sd": np.sqrt(var_o + var_d - 2 * cov_od),
                         "group": np.where(info["groups"][io] == "D_off", "D", "F")})


def team_table(f: dict) -> pd.DataFrame:
    info, m, cov = f["info"], f["mean"], f["cov"]
    if "team_levels" not in info:
        return pd.DataFrame()
    T = len(info["team_levels"])
    base = len(m) - 2 * T
    io, idf = base + np.arange(T), base + T + np.arange(T)
    return pd.DataFrame({"team_season": info["team_levels"], "off": m[io], "off_sd": np.sqrt(cov[io, io]),
                         "def": -m[idf], "def_sd": np.sqrt(cov[idf, idf]), "net": m[io] - m[idf]})