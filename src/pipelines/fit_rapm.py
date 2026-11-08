"""
fit_rapm.py
===========
v2.2 reference: fit ridge RAPM (5v5, xG) on the stints and report the player ratings.

Home: src/pipelines/fit_rapm.py

Usage (from the project root, after build_stints):
    py -m src.pipelines.fit_rapm                      # last 730 days of stints
    py -m src.pipelines.fit_rapm --days 365 --min-toi 200
    py -m src.pipelines.fit_rapm --end 2025-04-20     # window ending before that date

Prints the cross-validation curve (held-out error by lambda, folds grouped by game) with the
controls-only baseline, the covariate effects (home, score state, zone start), and the top and
bottom players by net impact. Saves output/models/rapm_<end>.csv (one row per player:
off, def, net in xG/60, 5v5 minutes, name, position) and rapm_<end>.json (settings, lambda,
held-out errors). The Bayesian lineup model must agree with these before it is trusted.
"""

import argparse
import json

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.deterministic import rapm

MODEL_DIR = OUTPUT_DIR / "models"
LAMBDAS = np.logspace(1, 4.5, 15)


def names(conn) -> pd.DataFrame:
    sk = pd.read_sql("SELECT player_id, name, position FROM skater_game_stats", conn)
    pos = sk.groupby("player_id")["position"].agg(lambda x: x.mode().iloc[0]).rename("position")
    nm = sk.groupby("player_id")["name"].last().rename("name")
    return pd.concat([nm, pos], axis=1).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Ridge RAPM on 5v5 stints (xG).")
    parser.add_argument("--end", default=None, help="window end (exclusive), default: day after the last stint")
    parser.add_argument("--days", type=int, default=730, help="window length in days (default 730)")
    parser.add_argument("--min-toi", type=float, default=100.0,
                        help="5v5 minutes for a player of his own (others share 'other' columns; default 100)")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    conn = get_connection()
    last = conn.execute("SELECT MAX(sc.game_date) FROM stints s JOIN schedule sc ON sc.game_id = s.game_id").fetchone()[0]
    if last is None:
        raise SystemExit("No stints. Run: py -m src.pipelines.build_stints")
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(last) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=args.days)
    st = rapm.load_stints(conn, start, end)
    who = names(conn)
    conn.close()
    print(f"5v5 stints {start.date()} to {end.date()}: {len(st):,} stints, {st['game_id'].nunique():,} games, "
          f"{st['duration'].sum() / 3600:,.0f} hours, xG/60 {(st['home_xg'] + st['away_xg']).sum() / st['duration'].sum() * 1800:.2f} per team")

    X, y, w, groups, info = rapm.design(st, min_toi=args.min_toi)
    print(f"Design: {X.shape[0]:,} rows x {X.shape[1]:,} columns; {info['P']:,} players with {args.min_toi:.0f}+ minutes")

    cv = rapm.cross_validate(X, y, w, groups, info["penalty"], LAMBDAS, folds=args.folds)
    num = cv[cv["lambda"] != "baseline"].copy()
    num["lambda"] = num["lambda"].astype(float)
    best = float(num.loc[num["heldout_mse"].idxmin(), "lambda"])
    base = float(cv.loc[cv["lambda"] == "baseline", "heldout_mse"].iloc[0])
    best_mse = float(num["heldout_mse"].min())
    print(f"\nCross-validation ({args.folds} folds by game), held-out weighted MSE of xG/60:")
    print(num.assign(**{"vs baseline": num["heldout_mse"] / base - 1}).round(5).to_string(index=False))
    print(f"  controls only (no players): {base:.5f}")
    print(f"Chosen lambda {best:,.0f}: players cut held-out error by {1 - best_mse / base:.2%} vs controls only"
          + ("" if best not in (LAMBDAS[0], LAMBDAS[-1]) else "  (at the edge of the grid: widen it)"))

    beta = rapm.fit(X, y, w, info["penalty"], best)
    names_ = info["names"]
    print("\nControls (xG/60 for the offense; tied, on-the-fly, road = 0):")
    for k in ["intercept"] + rapm.COVARIATES:
        print(f"  {k:12s} {beta[names_.index(k)]:+.3f}")
    print(f"  'other' players (under {args.min_toi:.0f} min): off {beta[names_.index('O_other')]:+.3f}, "
          f"def {-beta[names_.index('D_other')]:+.3f}")

    tab = rapm.player_table(beta, info).merge(who, on="player_id", how="left")
    tab = tab[["player_id", "name", "position", "toi_5v5", "off", "def", "net"]]
    show = tab[tab["toi_5v5"] >= 500].sort_values("net", ascending=False)
    fmt = lambda d: d.assign(toi_5v5=d["toi_5v5"].round(0)).round(3).to_string(index=False)
    print(f"\nTop 15 by net impact (xG/60 differential, 500+ 5v5 minutes; {len(show):,} players):")
    print(fmt(show.head(15)))
    print("\nBottom 10:")
    print(fmt(show.tail(10)))
    print("\nTop 10 offense:")
    print(fmt(show.sort_values("off", ascending=False).head(10)))
    print("\nTop 10 defense:")
    print(fmt(show.sort_values("def", ascending=False).head(10)))
    for pos, grp in (("forwards", show[show["position"].isin(["C", "L", "R"])]), ("defense", show[show["position"] == "D"])):
        print(f"  spread of net impact, {pos}: sd {grp['net'].std():.3f} ({len(grp)} players)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = MODEL_DIR / f"rapm_{(end - pd.Timedelta(days=1)).date()}"
    tab.sort_values("net", ascending=False).to_csv(f"{stem}.csv", index=False)
    with open(f"{stem}.json", "w") as f:
        json.dump({"start": str(start.date()), "end": str(end.date()), "days": args.days, "min_toi": args.min_toi,
                   "lambda": best, "heldout_mse": best_mse, "baseline_mse": base, "n_stints": len(st),
                   "players": int(info["P"]),
                   "controls": {k: float(beta[names_.index(k)]) for k in ["intercept"] + rapm.COVARIATES},
                   "cv": [{"lambda": (float(r["lambda"]) if r["lambda"] != "baseline" else "baseline"),
                           "heldout_mse": float(r["heldout_mse"])} for _, r in cv.iterrows()]}, f, indent=2)
    print(f"\nSaved {stem.name}.csv and .json to {MODEL_DIR}")


if __name__ == "__main__":
    main()
