"""
build_pce.py
============
Phase 8: PCE surrogate of the exact game model and Sobol' indices (analysis A).

Home: src/pipelines/build_pce.py

Usage (from the project root, after the rebuild order: fit_hierarchical -> late_game):
    py -m src.pipelines.build_pce
    py -m src.pipelines.build_pce --degree 5 --overwrite
    py -m src.pipelines.build_pce --saltelli 0          # skip the Monte Carlo check

Uses the active fit (output/posteriors/active_fit.json) for the input ranges.
Writes to output/models/:
    pce_<fit stem>.pkl              coefficients, multi-indices, specs, Sobol' indices
    pce_<fit stem>_summary.json     input ranges, accuracy, Sobol' (single and grouped), Saltelli check
    pce_<fit stem>_validation.png   PCE vs exact model on the hold-out points
    pce_<fit stem>_sobol.png        first-order and total indices per QoI
"""

import argparse
import json
import pickle
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.processing.late_game import load_params
from src.models.active_fit import read_active, active_fit_path, window_games, check_params_match
from src.models.game_model import load_track_b
from src.models.stochastic.pce import (
    build_pce_surrogate, sobol_groups, saltelli_sobol, plot_pce_validation, plot_sobol,
)
from src.analysis.sensitivity import (
    INPUT_NAMES, GROUPS, QOI_LABELS, input_specs, game_qoi, truth_predictor, sd_contributions,
)

MODEL_DIR = OUTPUT_DIR / "models"
LATE_GAME_PATH = MODEL_DIR / "late_game.json"
DEFAULT_DEGREE = 4
TRAIN_PER_TERM = 3           # training points per PCE coefficient
N_TEST = 2000
SEED = 42


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="PCE surrogate + Sobol' indices of the game model.")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE, help=f"total degree P (default {DEFAULT_DEGREE})")
    parser.add_argument("--n-train", type=int, default=None,
                        help=f"training points (default {TRAIN_PER_TERM} x number of terms)")
    parser.add_argument("--saltelli", type=int, default=2**14,
                        help="Saltelli base sample on the exact model (default 16384; 0 = skip)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # 1. Inputs from the active fit
    info = read_active()
    fit_path = active_fit_path()
    stem = MODEL_DIR / f"pce_{fit_path.stem}"
    if stem.with_suffix(".pkl").exists() and not args.overwrite:
        print(f"{stem.with_suffix('.pkl')} already exists. Use --overwrite to replace it.")
        return
    draws, teams = load_track_b(fit_path)
    late = load_params(LATE_GAME_PATH)
    warning = check_params_match(late, "late_game.json")
    if warning:
        raise SystemExit(warning)
    conn = get_connection()
    gt, _ = window_games(conn, info)
    conn.close()
    specs, spec_info = input_specs(draws, gt, late)

    print(f"Active fit {fit_path.name} ({len(gt):,} games). Inputs (uniform, variance-matched):")
    for n in INPUT_NAMES:
        kind, lo, hi = specs[n]
        print(f"  {n:12s} [{lo:+.4f}, {hi:+.4f}]   mean {spec_info[n]['mean']:+.4f}, sd {spec_info[n]['sd']:.4f}")

    # 2. PCE
    from src.models.stochastic.pce import total_order_indices
    n_terms = len(total_order_indices(len(INPUT_NAMES), args.degree))
    n_train = args.n_train or TRAIN_PER_TERM * n_terms
    t0 = time.time()
    result = build_pce_surrogate(lambda th: game_qoi(th, late), specs, P=args.degree,
                                 N_train=n_train, N_test=N_TEST, seed=SEED, vectorized=True)
    print(f"  built in {time.time() - t0:.1f} s")

    # 3. Sobol' (single inputs, groups) and the Saltelli check on the exact model
    summary = {"fit_file": fit_path.name, "degree": args.degree, "n_terms": n_terms,
               "n_train": n_train, "n_test": N_TEST, "seed": SEED,
               "inputs": {n: {"range": [specs[n][1], specs[n][2]], **spec_info[n]} for n in INPUT_NAMES},
               "qoi": {}, "built_at": datetime.now().isoformat(timespec="seconds")}
    for qoi, s in result["surrogates"].items():
        q = {"label": QOI_LABELS[qoi], "Q2_loocv": float(s["Q2_loocv"]), "R2_holdout": float(s["R2_holdout"]),
             "RMSE_holdout": float(s["RMSE_holdout"]), "variance": float(s["variance"]),
             "sd": float(np.sqrt(s["variance"])), "mean": float(s["coefficients"][0]),
             "S": dict(zip(INPUT_NAMES, map(float, s["S"]))),
             "S_T": dict(zip(INPUT_NAMES, map(float, s["S_T"]))),
             "sd_alone": dict(zip(INPUT_NAMES, map(float, sd_contributions(s["S"], s["variance"])))),
             "groups": sobol_groups(s["coefficients"], s["multi_indices"], GROUPS, INPUT_NAMES)}
        if args.saltelli:
            S_sal, ST_sal, var_sal = saltelli_sobol(truth_predictor(specs, late, qoi), len(INPUT_NAMES),
                                                    N=args.saltelli, seed=SEED)
            q["saltelli"] = {"N": args.saltelli, "variance": float(var_sal),
                             "S": dict(zip(INPUT_NAMES, map(float, S_sal))),
                             "S_T": dict(zip(INPUT_NAMES, map(float, ST_sal)))}
        summary["qoi"][qoi] = q

    # 4. Print
    for qoi, q in summary["qoi"].items():
        print(f"\n{q['label']}: mean {q['mean']:.4f}, sd {q['sd']:.4f} | Q2 {q['Q2_loocv']:.5f}, "
              f"hold-out R2 {q['R2_holdout']:.5f}, RMSE {q['RMSE_holdout']:.2e}")
        sal = q.get("saltelli")
        print(f"  {'input':12s} {'S_i':>7s} {'S_T,i':>7s}" + (f" {'S_i MC':>7s} {'S_T MC':>7s}" if sal else "")
              + f" {'sd alone':>9s}")
        for n in sorted(INPUT_NAMES, key=lambda n: -q["S_T"][n]):
            row = f"  {n:12s} {q['S'][n]:7.4f} {q['S_T'][n]:7.4f}"
            if sal:
                row += f" {sal['S'][n]:7.4f} {sal['S_T'][n]:7.4f}"
            print(row + f" {q['sd_alone'][n]:9.4f}")
        print("  groups: " + ";  ".join(f"{g} S {v['S']:.3f} / S_T {v['S_T']:.3f}" for g, v in q["groups"].items()))
        print(f"  sum of first-order indices {sum(q['S'].values()):.4f} (1 - sum = share from interactions)")

    # 5. Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keep = {"param_specs": specs, "P": args.degree, "input_names": INPUT_NAMES,
            "multi_indices": next(iter(result["surrogates"].values()))["multi_indices"],
            "surrogates": {q: {k: v for k, v in s.items() if k != "multi_indices"}
                           for q, s in result["surrogates"].items()},
            "Xi_test": result["Xi_test"], "Y_test": result["Y_test"], "fit_file": fit_path.name}
    with open(stem.with_suffix(".pkl"), "wb") as f:
        pickle.dump(keep, f)
    with open(f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_pce_validation(result, savepath=f"{stem}_validation.png")
    plot_sobol(result, savepath=f"{stem}_sobol.png")
    plt.close("all")
    print(f"\nSaved {stem.name}.pkl, _summary.json, _validation.png, _sobol.png to {MODEL_DIR}")


if __name__ == "__main__":
    main()