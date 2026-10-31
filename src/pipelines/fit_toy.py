"""
fit_toy.py
==========
Fit the Track A toy model to one team pair and save the posterior.

Home: src/pipelines/fit_toy.py

Usage (from the project root):
    py -m src.pipelines.fit_toy COL DAL
    py -m src.pipelines.fit_toy COL DAL --overwrite

Writes to output/posteriors/:
    toy_<A>_<B>.nc          posterior samples + observed games (ArviZ NetCDF)
    toy_<A>_<B>_meta.json   settings, diagnostics, summary
    toy_<A>_<B>_trace.png   trace plot (chain 1)
    toy_<A>_<B>_acf.png     autocorrelation plot (chain 1)
"""

import argparse
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                  # save plots without opening windows
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.processing.team_strength import team_game_xg
from src.models.stochastic.toy_mcmc import (
    PARAM_NAMES, league_baseline, pair_games, make_log_posterior,
)
from src.models.stochastic.am_sampler import adaptive_metropolis, plot_traces, plot_acf

POSTERIOR_DIR = OUTPUT_DIR / "posteriors"

RANDOM_SEED = 42
N_CHAINS = 4
N_BURN, N_MAIN, SIGMA0 = 5000, 50000, 0.1
STARTS = [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]   # spread out for R-hat

# Pass thresholds for the printed checks
MAX_RHAT = 1.01
MIN_ESS = 1000
ACCEPT_RANGE = (0.2, 0.5)


def run_chains(log_post) -> tuple[np.ndarray, list[float]]:
    """Run N_CHAINS independent AM chains from spread-out starting points.

    Returns:
        samples: (N_CHAINS, N_MAIN, 2) array, post-burn-in draws
        accept:  main-phase acceptance rate per chain
    """
    samples, accept = [], []

    for i, start in enumerate(STARTS[:N_CHAINS]):
        chain, info = adaptive_metropolis(
            log_post,
            d=len(PARAM_NAMES),
            n_burn=N_BURN,
            n_main=N_MAIN,
            sigma0=SIGMA0,
            seed=RANDOM_SEED + i,          # different per chain, repeatable per run
            x0=np.asarray(start, dtype=float),
            verbose=False,
        )
        samples.append(chain)
        accept.append(float(info["main_acc_rate"]))
        print(f"  chain {i + 1}/{N_CHAINS}: acceptance {accept[-1]:.3f}")

    return np.stack(samples), accept


def to_inference_data(samples, games, team_a, team_b, alpha0, eta):
    """Package samples and the games they came from in ArviZ format.

    ArviZ 1.x returns an xarray DataTree, 0.x an InferenceData; both save with
    .to_netcdf() and load with az.from_netcdf().
    """
    posterior = {name: samples[:, :, j] for j, name in enumerate(PARAM_NAMES)}
    observed = {
        "goals_a": games["goals_a"].to_numpy(),
        "goals_b": games["goals_b"].to_numpy(),
        "a_home": games["a_home"].to_numpy(),
    }
    if int(az.__version__.split(".")[0]) >= 1:
        idata = az.from_dict({"posterior": posterior, "observed_data": observed})
    else:
        idata = az.from_dict(posterior=posterior, observed_data=observed)

    # Model settings travel with the samples
    idata.posterior.attrs.update({
        "model": "track_a_toy",
        "team_a": team_a,
        "team_b": team_b,
        "alpha0": alpha0,
        "eta": eta,
    })
    return idata


def diagnostics(samples: np.ndarray, accept: list[float]) -> dict:
    """R-hat, ESS, acceptance and a 90% summary; prints PASS / CHECK lines.

    samples: (chain, draw, param). R-hat and ESS come from ArviZ; mean, sd and
    the equal-tailed 90% interval are computed directly (same as the notebook).
    """
    pooled = samples.reshape(-1, samples.shape[2])          # all chains together
    out = {"acceptance": accept, "params": {}}
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = np.quantile(pooled[:, j], [0.05, 0.95])
        out["params"][name] = {
            "mean": float(pooled[:, j].mean()),
            "sd": float(pooled[:, j].std()),
            "q05": float(lo),
            "q95": float(hi),
            "rhat": float(az.rhat(samples[:, :, j])),
            "ess": float(az.ess(samples[:, :, j])),
        }

    def check(label, ok, note=""):
        print(f"{'PASS ' if ok else 'CHECK'}  {label}" + (f"   ({note})" if note else ""))

    p = out["params"]
    check(f"R-hat < {MAX_RHAT}", all(v["rhat"] < MAX_RHAT for v in p.values()),
          ", ".join(f"{k} {v['rhat']:.4f}" for k, v in p.items()))
    check(f"ESS > {MIN_ESS}", all(v["ess"] > MIN_ESS for v in p.values()),
          ", ".join(f"{k} {v['ess']:,.0f}" for k, v in p.items()))
    lo, hi = ACCEPT_RANGE
    check(f"acceptance in {lo}-{hi}", all(lo <= a <= hi for a in accept),
          ", ".join(f"{a:.3f}" for a in accept))
    return out


def main():
    parser = argparse.ArgumentParser(description="Fit the Track A toy model to one team pair.")
    parser.add_argument("team_a", help="team abbreviation, e.g. COL")
    parser.add_argument("team_b", help="team abbreviation, e.g. DAL")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output")
    args = parser.parse_args()
    team_a, team_b = args.team_a.upper(), args.team_b.upper()

    # 1. Paths and overwrite protection
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    stem = POSTERIOR_DIR / f"toy_{team_a}_{team_b}"
    nc_path = stem.with_suffix(".nc")
    if nc_path.exists() and not args.overwrite:
        print(f"{nc_path} already exists. Use --overwrite to replace it.")
        return

    # 2. Data
    conn = get_connection()
    tg = team_game_xg(conn)
    conn.close()
    alpha0, eta = league_baseline(tg)
    games = pair_games(tg, team_a, team_b)
    if games.empty:
        raise ValueError(
            f"No games for {team_a} vs {team_b}. Teams available: {sorted(tg['team'].unique())}"
        )
    print(f"{team_a} vs {team_b}: {len(games)} games "
          f"({games['game_date'].min()} to {games['game_date'].max()}), "
          f"{team_a} home in {int(games['a_home'].sum())}")
    print(f"alpha0 = {alpha0:.4f}, eta = {eta:.4f}")

    # 3. Fit
    log_post = make_log_posterior(games, alpha0, eta)
    print(f"Running {N_CHAINS} chains ({N_BURN:,} burn-in + {N_MAIN:,} draws each)...")
    samples, accept = run_chains(log_post)

    # 4. Package and diagnose
    idata = to_inference_data(samples, games, team_a, team_b, alpha0, eta)
    diag = diagnostics(samples, accept)

    # 5. Save posterior, plots, metadata
    idata.to_netcdf(nc_path)

    plot_traces(samples[0], PARAM_NAMES, thin=10, savepath=f"{stem}_trace.png")
    plot_acf(samples[0], PARAM_NAMES, max_lag=100, savepath=f"{stem}_acf.png")
    plt.close("all")

    meta = {
        "model": "track_a_toy",
        "team_a": team_a,
        "team_b": team_b,
        "n_games": int(len(games)),
        "first_game": str(games["game_date"].min()),
        "last_game": str(games["game_date"].max()),
        "game_ids": [int(g) for g in games["game_id"]],
        "alpha0": alpha0,
        "eta": eta,
        "goals_target": "gf_xg / ga_xg (periods 1-4, empty-net goals excluded)",
        "sampler": {
            "name": "adaptive_metropolis",
            "n_chains": N_CHAINS,
            "n_burn": N_BURN,
            "n_main": N_MAIN,
            "sigma0": SIGMA0,
            "seed": RANDOM_SEED,
            "starts": STARTS[:N_CHAINS],
        },
        "diagnostics": diag,
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {"numpy": np.__version__, "arviz": az.__version__},
    }
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 6. Summary
    print()
    for name, v in diag["params"].items():
        print(f"{name}: mean {v['mean']:+.3f}   90% [{v['q05']:+.3f}, {v['q95']:+.3f}]"
              f"   rate x{np.exp(v['mean']):.2f}")
    print(f"\nSaved {nc_path.name}, {stem.name}_meta.json, _trace.png, _acf.png to {POSTERIOR_DIR}")


if __name__ == "__main__":
    main()