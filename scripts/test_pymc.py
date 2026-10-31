"""
test_pymc.py
============
Checks that PyMC is installed and samples correctly and fast enough.

Home: scripts/test_pymc.py
Run:  py scripts/test_pymc.py
"""

import time

import numpy as np
from scipy import special


def check(label, ok, note=""):
    print(f"{'PASS ' if ok else 'CHECK'}  {label}" + (f"   ({note})" if note else ""))


# 1. Imports and versions
import pymc as pm
import pytensor
import arviz as az

print(f"pymc {pm.__version__}, pytensor {pytensor.__version__}, arviz {az.__version__}")

# 2. C compiler: without one, PyTensor falls back to pure Python and runs far slower
cxx = pytensor.config.cxx
check("C compiler found", bool(cxx), cxx or "none: sampling will be slow, see notes")

# 3. A tiny model with a known answer: goals ~ Poisson(exp(a)), flat-ish prior on a
rng = np.random.default_rng(0)
goals = rng.poisson(3.0, size=200)

with pm.Model():
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    pm.Poisson("goals", mu=pm.math.exp(a), observed=goals)

    start = time.perf_counter()
    idata = pm.sample(draws=1000, tune=1000, chains=4, cores=1,
                      random_seed=42, progressbar=False)
    seconds = time.perf_counter() - start

draws = idata.posterior["a"].values            # (chains, draws)

# Exact answer (same Gamma result as Phase 4): E[a] = digamma(Y) - log(n)
exact_mean = special.digamma(goals.sum()) - np.log(len(goals))
exact_sd = np.sqrt(special.polygamma(1, goals.sum()))
mc_mean, mc_sd = draws.mean(), draws.std()

print(f"\nposterior mean {mc_mean:.4f} (exact {exact_mean:.4f}), "
      f"sd {mc_sd:.4f} (exact {exact_sd:.4f}), took {seconds:.1f} s")

check("4 chains x 1000 draws", draws.shape == (4, 1000))
check("mean matches exact (within 0.01)", abs(mc_mean - exact_mean) < 0.01)
check("sd matches exact (within 10%)", abs(mc_sd / exact_sd - 1) < 0.10)
check("R-hat < 1.01", float(az.rhat(draws)) < 1.01, f"{float(az.rhat(draws)):.4f}")
divergences = int(idata.sample_stats["diverging"].values.sum())
check("no divergences", divergences == 0, f"{divergences}")
check("fast enough (< 60 s, includes compiling)", seconds < 60, f"{seconds:.1f} s")
