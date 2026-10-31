"""
pce.py
======
Polynomial chaos expansion (PCE) surrogate and Sobol' indices, ported from the
AEM 8290 soil-column UQ pipeline (driver section 4-5).

Home: src/models/stochastic/pce.py

Same method as the soil project:
  * inputs mapped to xi in [-1, 1]^d (uniform inputs -> Legendre polynomials)
  * total-order truncation: every multi-index alpha with |alpha| <= P
  * coefficients by least squares on a Latin hypercube design (non-intrusive regression)
  * accuracy: leave-one-out Q^2 (closed form from the hat matrix) and a hold-out test
  * Sobol' indices straight from the coefficients:
        Var = sum_{alpha != 0} c_alpha^2 E[Psi_alpha^2],  E[P_n^2] = 1 / (2n + 1)
  * Saltelli / Jansen Monte Carlo estimator as an independent check

Changes from the soil version (behavior is the same):
  * total_order_indices builds the multi-indices recursively. The soil version looped
    over all (P+1)^d combinations, which is about 10 million for d = 10, P = 4; same set,
    same order (by total degree).
  * build_pce_surrogate accepts a vectorized QoI function (dict of arrays in, dict of
    arrays out), since the hockey game model evaluates thousands of inputs at once.
    vectorized=False keeps the soil behavior (one dict of floats per call).
  * sobol_groups: first-order and total indices for groups of inputs.
  * ishigami: standard test function with known Sobol' indices (known-answer check).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Input mapping: physical value <-> xi in [-1, 1]
# param_specs: {name: (kind, lo, hi)}, kind 'linear' (uniform) or 'log' (log-uniform)
# ---------------------------------------------------------------------------

def theta_to_xi(theta_dict, param_specs):
    xi = np.zeros(len(param_specs))
    for i, (name, (kind, lo, hi)) in enumerate(param_specs.items()):
        if kind == 'linear':
            xi[i] = 2.0 * (theta_dict[name] - lo) / (hi - lo) - 1.0
        elif kind == 'log':
            ll, lh = np.log(lo), np.log(hi)
            xi[i] = 2.0 * (np.log(theta_dict[name]) - ll) / (lh - ll) - 1.0
    return xi


def xi_to_theta(xi, param_specs):
    """xi: (d,) or (N, d). Returns {name: float or (N,) array}."""
    xi = np.asarray(xi, float)
    theta = {}
    for i, (name, (kind, lo, hi)) in enumerate(param_specs.items()):
        u = 0.5 * (xi[..., i] + 1.0)
        if kind == 'linear':
            theta[name] = lo + u * (hi - lo)
        elif kind == 'log':
            ll, lh = np.log(lo), np.log(hi)
            theta[name] = np.exp(ll + u * (lh - ll))
    return theta


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def total_order_indices(d, P):
    """All multi-indices alpha in N^d with |alpha| <= P, ordered by total degree."""
    def compositions(total, parts):
        # every way to write `total` as an ordered sum of `parts` non-negative integers,
        # in the same lexicographic order as itertools.product
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest
    indices = []
    for total in range(P + 1):
        indices.extend(sorted(compositions(total, d)))
    return np.array(indices, dtype=int)


def legendre_eval(n, xi):
    if n == 0: return np.ones_like(np.asarray(xi, dtype=float))
    if n == 1: return np.asarray(xi, dtype=float)
    P_nm1 = np.ones_like(np.asarray(xi, dtype=float))
    P_n   = np.asarray(xi, dtype=float)
    for k in range(1, n):
        P_np1 = ((2*k + 1) * xi * P_n - k * P_nm1) / (k + 1)
        P_nm1, P_n = P_n, P_np1
    return P_n


def legendre_norm_sq(n):
    """E[P_n(xi)^2] for xi uniform on [-1, 1]."""
    return 1.0 / (2 * n + 1)


def build_design_matrix(Xi, multi_indices):
    N, d = Xi.shape
    P_max = int(multi_indices.max())
    # one Legendre evaluation per (input, degree), reused across terms
    L = np.stack([np.stack([legendre_eval(n, Xi[:, j]) for n in range(P_max + 1)], axis=1)
                  for j in range(d)], axis=0)                      # (d, N, P_max + 1)
    Phi = np.ones((N, multi_indices.shape[0]))
    for j in range(d):
        Phi *= L[j][:, multi_indices[:, j]]
    return Phi


# ---------------------------------------------------------------------------
# Fit, predict, accuracy
# ---------------------------------------------------------------------------

def lhs_samples(N, d, seed=0):
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return 2.0 * sampler.random(N) - 1.0


def fit_pce(Xi_train, Y_train, multi_indices):
    Phi = build_design_matrix(Xi_train, multi_indices)
    c, *_ = np.linalg.lstsq(Phi, Y_train, rcond=None)
    return c, Phi


def pce_predict(Xi, c, multi_indices):
    return build_design_matrix(Xi, multi_indices) @ c


def loocv_q2(Xi_train, Y_train, multi_indices):
    """Leave-one-out Q^2 without refitting: residual_i / (1 - h_ii)."""
    Phi = build_design_matrix(Xi_train, multi_indices)
    c, *_ = np.linalg.lstsq(Phi, Y_train, rcond=None)
    res = Y_train - Phi @ c
    PtP_inv = np.linalg.pinv(Phi.T @ Phi)
    H_diag = np.einsum('ij,jk,ik->i', Phi, PtP_inv, Phi)
    loo_res = res / np.clip(1.0 - H_diag, 1e-10, None)
    return 1.0 - np.sum(loo_res**2) / np.sum((Y_train - Y_train.mean())**2)


def holdout_test(Xi_train, Y_train, Xi_test, Y_test, multi_indices):
    c, _ = fit_pce(Xi_train, Y_train, multi_indices)
    Y_pred = pce_predict(Xi_test, c, multi_indices)
    rmse = np.sqrt(np.mean((Y_test - Y_pred)**2))
    r2 = 1.0 - np.sum((Y_test - Y_pred)**2) / np.sum((Y_test - Y_test.mean())**2)
    return rmse, r2, Y_pred


# ---------------------------------------------------------------------------
# Sobol' indices
# ---------------------------------------------------------------------------

def _term_variances(c, multi_indices):
    Psi_sq = np.prod(1.0 / (2 * multi_indices + 1), axis=1)       # prod of legendre_norm_sq
    contrib = c**2 * Psi_sq
    not_const = multi_indices.sum(axis=1) > 0
    return contrib, not_const


def sobol_from_pce(c, multi_indices, d):
    """First-order S_i, total S_T_i and the output variance, from the coefficients."""
    contrib, not_const = _term_variances(c, multi_indices)
    var_total = contrib[not_const].sum()
    S, S_T = np.zeros(d), np.zeros(d)
    for i in range(d):
        first = (multi_indices[:, i] > 0) & (multi_indices.sum(axis=1) == multi_indices[:, i])
        total = multi_indices[:, i] > 0
        S[i]   = contrib[first].sum() / var_total
        S_T[i] = contrib[total].sum() / var_total
    return S, S_T, var_total


def sobol_groups(c, multi_indices, groups: dict, names: list):
    """Sobol' indices for groups of inputs.

    groups: {group name: [input names]}. First-order = terms that use only inputs in the
    group (the group's effect on its own, interactions inside it included); total = terms
    that use any input in the group.
    """
    contrib, not_const = _term_variances(c, multi_indices)
    var_total = contrib[not_const].sum()
    active = multi_indices > 0
    out = {}
    for g, members in groups.items():
        cols = np.array([names.index(m) for m in members])
        inside = np.zeros(len(names), bool); inside[cols] = True
        uses_group = active[:, inside].any(axis=1)
        only_group = uses_group & ~active[:, ~inside].any(axis=1)
        out[g] = {"S": float(contrib[only_group].sum() / var_total),
                  "S_T": float(contrib[uses_group].sum() / var_total)}
    return out


def saltelli_sobol(predict_func, d, N=2048, seed=0):
    """Saltelli-Sobol' indices with Sobol' sequences (column-split for independence)
    and Jansen 1999 estimators. predict_func: (n, d) xi array -> (n,) outputs."""
    sampler = qmc.Sobol(d=2*d, scramble=True, seed=seed)
    M = 2.0 * sampler.random(N) - 1.0
    A, B = M[:, :d], M[:, d:]
    AB = np.empty((d, N, d))
    for i in range(d):
        AB[i] = A.copy()
        AB[i, :, i] = B[:, i]
    y_A = predict_func(A)
    y_B = predict_func(B)
    y_AB = np.array([predict_func(AB[i]) for i in range(d)])
    var_total = np.var(np.concatenate([y_A, y_B]), ddof=0)
    V_i = np.array([np.mean(y_B * (y_AB[i] - y_A)) for i in range(d)])          # Saltelli 2010
    V_T = np.array([0.5 * np.mean((y_A - y_AB[i])**2) for i in range(d)])       # Jansen 1999
    return V_i / var_total, V_T / var_total, var_total


def make_surrogate_predict(coefficients, multi_indices):
    def predict(Xi):
        return pce_predict(Xi, coefficients, multi_indices)
    return predict


# ---------------------------------------------------------------------------
# Build: sample, evaluate, fit, validate, Sobol'
# ---------------------------------------------------------------------------

def build_pce_surrogate(QoI_func, param_specs, P=3, N_train=150, N_test=30, seed=0,
                        verbose=True, vectorized=False):
    """Fit one PCE per QoI on a Latin hypercube design.

    QoI_func: vectorized=False -> theta dict of floats -> {qoi: float} (soil style);
              vectorized=True  -> theta dict of (N,) arrays -> {qoi: (N,) array}.
    """
    d = len(param_specs)
    multi_indices = total_order_indices(d, P)
    if verbose:
        print(f"  PCE: d={d}, P={P}, N_terms={multi_indices.shape[0]}")
        print(f"  Train={N_train}, test={N_test}, total evals={N_train + N_test}")
    Xi_train = lhs_samples(N_train, d, seed=seed)
    Xi_test  = lhs_samples(N_test,  d, seed=seed + 1000)

    def run_batch(Xi, label):
        if vectorized:
            return {k: np.asarray(v, float) for k, v in QoI_func(xi_to_theta(Xi, param_specs)).items()}
        N = Xi.shape[0]
        Ys = None
        for i in range(N):
            if verbose and (i % 25 == 0 or i == N - 1):
                print(f"    {label} eval {i+1}/{N}")
            y = QoI_func(xi_to_theta(Xi[i], param_specs))
            if Ys is None:
                Ys = {n_: np.zeros(N) for n_ in y}
            for n_ in y: Ys[n_][i] = y[n_]
        return Ys

    Y_train = run_batch(Xi_train, 'train')
    Y_test  = run_batch(Xi_test,  'test')
    surrogates = {}
    for qoi in Y_train:
        c, _ = fit_pce(Xi_train, Y_train[qoi], multi_indices)
        Q2 = loocv_q2(Xi_train, Y_train[qoi], multi_indices)
        rmse, r2, _ = holdout_test(Xi_train, Y_train[qoi], Xi_test, Y_test[qoi], multi_indices)
        S, S_T, var_total = sobol_from_pce(c, multi_indices, d)
        surrogates[qoi] = {
            'coefficients': c, 'multi_indices': multi_indices,
            'Q2_loocv': Q2, 'R2_holdout': r2, 'RMSE_holdout': rmse,
            'S': S, 'S_T': S_T, 'variance': var_total,
            'param_names': list(param_specs.keys()),
        }
    return {'surrogates': surrogates, 'param_specs': param_specs,
            'Xi_train': Xi_train, 'Y_train': Y_train,
            'Xi_test': Xi_test, 'Y_test': Y_test,
            'P': P, 'N_train': N_train, 'N_test': N_test}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pce_validation(pce_result, savepath=None):
    surrogates = pce_result['surrogates']
    fig, axes = plt.subplots(1, len(surrogates), figsize=(5*len(surrogates), 5), squeeze=False)
    for ax, (qoi, surr) in zip(axes[0], surrogates.items()):
        Y_test = pce_result['Y_test'][qoi]
        Y_pred = pce_predict(pce_result['Xi_test'], surr['coefficients'], surr['multi_indices'])
        lo, hi = min(Y_test.min(), Y_pred.min()), max(Y_test.max(), Y_pred.max())
        m = 0.05 * (hi - lo)
        ax.plot([lo - m, hi + m], [lo - m, hi + m], 'k--', alpha=0.5)
        ax.scatter(Y_test, Y_pred, s=8, alpha=0.5)
        ax.set_xlabel('Truth model')
        ax.set_ylabel('PCE surrogate')
        ax.set_title(f'{qoi}\n$R^2={surr["R2_holdout"]:.5f}$, $Q^2={surr["Q2_loocv"]:.5f}$')
        ax.grid(alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=110)
    return fig


def plot_sobol(pce_result, savepath=None):
    surrogates = pce_result['surrogates']
    fig, axes = plt.subplots(1, len(surrogates), figsize=(6*len(surrogates), 5), squeeze=False)
    for ax, (qoi, surr) in zip(axes[0], surrogates.items()):
        names = surr['param_names']
        x = np.arange(len(names)); w = 0.35
        ax.bar(x - w/2, surr['S'],   w, label=r'$S_i$ (first order)')
        ax.bar(x + w/2, surr['S_T'], w, label=r'$S_{T,i}$ (total)')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right')
        ax.set_ylabel('Sobol index')
        ax.set_title(f'Sobol indices: {qoi}')
        ax.grid(axis='y', alpha=0.3); ax.legend()
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=110)
    return fig


# ---------------------------------------------------------------------------
# Known-answer test
# ---------------------------------------------------------------------------

ISHIGAMI_SPECS = {f"x{i}": ('linear', -np.pi, np.pi) for i in (1, 2, 3)}


def ishigami(theta, a=7.0, b=0.1):
    """Ishigami function (vectorized): sin x1 + a sin^2 x2 + b x3^4 sin x1."""
    x1, x2, x3 = theta["x1"], theta["x2"], theta["x3"]
    return {"ishigami": np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)}


def ishigami_exact(a=7.0, b=0.1):
    """Analytic Sobol' indices of the Ishigami function on [-pi, pi]^3."""
    pi = np.pi
    V1 = 0.5 * (1 + b * pi**4 / 5)**2
    V2 = a**2 / 8
    V13 = b**2 * pi**8 * (1/18 - 1/50)
    V = V1 + V2 + V13
    S = np.array([V1, V2, 0.0]) / V
    S_T = np.array([V1 + V13, V2, V13]) / V
    return S, S_T, V