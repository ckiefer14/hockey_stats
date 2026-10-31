"""
am_sampler.py
=============
Adaptive Metropolis sampler and diagnostics, ported unchanged from the soil
UQ project (driver_fast.py). Model-agnostic: takes any log_post(x) and d.

Home: src/models/stochastic/am_sampler.py
"""

import numpy as np
import matplotlib.pyplot as plt


def adaptive_metropolis(log_post, d, n_burn=5000, n_main=50000, seed=0,
                         x0=None, sigma0=0.1, target_accept=0.234,
                         verbose=True):
    """
    Adaptive Metropolis with random-walk Gaussian proposal.
    
    Phase 1 (burn-in): isotropic proposal Sigma_p = sigma0^2 * I.
                       Used to explore and accumulate samples.
    Phase 2 (main):    Sigma_p = (2.38^2 / d) * Cov(burn-in samples),
                       per Eq. 115 of the course notes.
    
    Parameters
    ----------
    log_post: callable xi -> float, log posterior (up to a constant)
    d:        number of parameters
    n_burn:   burn-in samples (used to estimate posterior covariance)
    n_main:   post-burn-in samples (these are the kept posterior draws)
    sigma0:   initial isotropic step size (in unit-cube xi-space)
    
    Returns
    -------
    samples_main: (n_main, d) accepted post-burn-in samples
    info: dict with diagnostics (acceptance rates, Sigma_p, full chain, etc.)
    """
    rng = np.random.default_rng(seed)
    
    # Initial point: center of cube unless specified
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float).copy()
    lp = log_post(x)
    if not np.isfinite(lp):
        raise ValueError("Initial point has zero posterior probability")
    
    # ---- Phase 1: burn-in with isotropic proposal ----
    Sigma_p = (sigma0**2) * np.eye(d)
    L = np.linalg.cholesky(Sigma_p)
    burn_chain = np.zeros((n_burn, d))
    burn_accept = 0
    for k in range(n_burn):
        proposal = x + L @ rng.standard_normal(d)
        lp_new = log_post(proposal)
        # Symmetric proposal: alpha = min(1, pi(y)/pi(x))  -> log-form
        log_alpha = lp_new - lp
        if np.log(rng.uniform()) < log_alpha:
            x, lp = proposal, lp_new
            burn_accept += 1
        burn_chain[k] = x
        if verbose and (k + 1) % max(1, n_burn // 10) == 0:
            print(f"  burn-in {k+1}/{n_burn}, acc rate so far: "
                   f"{burn_accept/(k+1):.3f}")
    burn_acc_rate = burn_accept / n_burn
    
    # ---- Adaptation: Sigma_p = (2.38^2 / d) * empirical covariance ----
    Sigma_emp = np.cov(burn_chain, rowvar=False)
    s = (2.38**2) / d                                    # Roberts et al.
    Sigma_p_main = s * Sigma_emp + 1e-8 * np.eye(d)      # nugget for stability
    L_main = np.linalg.cholesky(Sigma_p_main)
    
    if verbose:
        print(f"  Burn-in acceptance rate: {burn_acc_rate:.3f}")
        print(f"  Adapted Sigma_p scale:   {s:.4f}")
    
    # ---- Phase 2: main sampling with adapted proposal ----
    main_chain = np.zeros((n_main, d))
    main_accept = 0
    for k in range(n_main):
        proposal = x + L_main @ rng.standard_normal(d)
        lp_new = log_post(proposal)
        log_alpha = lp_new - lp
        if np.log(rng.uniform()) < log_alpha:
            x, lp = proposal, lp_new
            main_accept += 1
        main_chain[k] = x
        if verbose and (k + 1) % max(1, n_main // 10) == 0:
            print(f"  main {k+1}/{n_main}, acc rate so far: "
                   f"{main_accept/(k+1):.3f}")
    main_acc_rate = main_accept / n_main
    
    info = {
        'burn_chain':    burn_chain,
        'burn_acc_rate': burn_acc_rate,
        'main_acc_rate': main_acc_rate,
        'Sigma_emp':     Sigma_emp,
        'Sigma_p_main':  Sigma_p_main,
        'd':             d,
        'n_burn':        n_burn,
        'n_main':        n_main,
    }
    return main_chain, info


def autocorrelation(x, max_lag=100):
    """
    Empirical autocorrelation function up to max_lag.
    x: (n,) 1D series
    """
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(x[:n-k], x[k:]) / (n * var)
    return acf


def plot_traces(samples, param_names, savepath=None, thin=1):
    """Trace plot for each parameter; one panel per dimension."""
    n_steps, d = samples.shape
    fig, axes = plt.subplots(d, 1, figsize=(10, 2*d), sharex=True)
    if d == 1:
        axes = [axes]
    steps = np.arange(0, n_steps, thin)
    for i, ax in enumerate(axes):
        ax.plot(steps, samples[::thin, i], lw=0.4)
        ax.set_ylabel(param_names[i])
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('MCMC step')
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=110)
    return fig


def plot_acf(samples, param_names, max_lag=100, savepath=None):
    """Autocorrelation plot for each parameter."""
    n_steps, d = samples.shape
    fig, axes = plt.subplots(1, d, figsize=(3*d, 3.5), sharey=True)
    if d == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        acf = autocorrelation(samples[:, i], max_lag=max_lag)
        ax.fill_between(np.arange(max_lag+1), 0, acf, alpha=0.6)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('lag')
        ax.set_title(param_names[i], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('ACF')
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=110)
    return fig