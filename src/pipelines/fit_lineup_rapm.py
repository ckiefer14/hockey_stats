"""
fit_lineup_rapm.py
==================
v2.2: fit the Bayesian RAPM (lineup_rapm.py) on 5v5 stints and check it against the ridge
reference (fit_rapm.py).

Home: src/pipelines/fit_lineup_rapm.py

Usage (from the project root, after build_stints and fit_rapm):
    py -m src.pipelines.fit_lineup_rapm               # last 730 days, with the held-out comparison
    py -m src.pipelines.fit_lineup_rapm --teams       # + team-season system effects
    py -m src.pipelines.fit_lineup_rapm --no-cv       # skip the comparison (fit only, ~1 min)

Prints the learned spreads (forward / defense, offense / defense, team if --teams), the held-out
comparison on the same game folds (controls only, ridge, Bayes), agreement with the latest
ridge table (output/models/rapm_*.csv), top players with posterior sds, and team effects.
Saves output/models/lineup_rapm_<end>[_teams].csv (player ratings with sds), .json (spreads,
settings, held-out errors) and _cov.npy (posterior covariance of all effects, for lineup sums).
"""

import argparse
import json

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.deterministic import rapm
from src.models.stochastic import lineup_rapm as lr
from src.pipelines.fit_rapm import names

MODEL_DIR = OUTPUT_DIR / "models"
RIDGE_LAMBDAS = np.logspace(2, 4, 9)


def main():
    parser = argparse.ArgumentParser(description="Bayesian RAPM on 5v5 stints (xG).")
    parser.add_argument("--end", default=None, help="window end (exclusive), default: day after the last stint")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--min-toi", type=float, default=100.0)
    parser.add_argument("--teams", action="store_true", help="add team-season system effects")
    parser.add_argument("--no-cv", action="store_true", help="skip the held-out comparison")
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
    pos = who.set_index("player_id")["position"]
    print(f"5v5 stints {start.date()} to {end.date()}: {len(st):,} stints, {st['game_id'].nunique():,} games, "
          f"{st['duration'].sum() / 3600:,.0f} hours")

    X, y, w, games, info = lr.design(st, pos, min_toi=args.min_toi, teams=args.teams)
    print(f"Design: {X.shape[0]:,} rows x {X.shape[1]:,} columns; {info['P']:,} players"
          + (f", {len(info['team_levels'])} team-seasons" if args.teams else ""))

    cv = None
    if not args.no_cv:
        print("\nHeld-out comparison (5 folds by game; Bayes spreads re-fit in every training fold)...", flush=True)
        cv = lr.cross_validate(X, y, w, games, info, RIDGE_LAMBDAS)
        base = float(cv["heldout_mse"].iloc[0])
        print(cv.assign(**{"vs controls": cv["heldout_mse"] / base - 1}).round(5).to_string(index=False))
        r, b = float(cv["heldout_mse"].iloc[1]), float(cv["heldout_mse"].iloc[2])
        print(f"  Bayes minus ridge: {b - r:+.5f} ({'Bayes better' if b < r else 'ridge better'}; ridge's lambda was "
              f"picked on these same test folds, which flatters it slightly)")

    print("\nFitting on the whole window...", flush=True)
    f = lr.fit(X, y, w, info, verbose=False)
    th = f["theta"]
    print("Learned spreads (sd of true effects, xG/60):")
    print(f"  residual sigma {th['sigma']:.2f} (per minute-weighted stint)")
    for g in lr.free_groups(info):
        print(f"  {g:9s} {th[g]:.3f}")
    rj = sorted(MODEL_DIR.glob("rapm_*.json"))
    if rj:
        lam = json.load(open(rj[-1]))["lambda"]
        print(f"  (ridge's lambda {lam:,.0f} in {rj[-1].name} amounts to one shared spread of "
              f"{np.sqrt(th['sigma'] ** 2 / lam):.3f})")

    tab = lr.player_table(f).merge(who, on="player_id", how="left")
    ridge_files = sorted(MODEL_DIR.glob("rapm_*.csv"))
    if ridge_files:
        rd = pd.read_csv(ridge_files[-1])
        mm = tab.merge(rd[["player_id", "off", "def", "net"]], on="player_id", suffixes=("", "_ridge"))
        print(f"\nAgreement with {ridge_files[-1].name} ({len(mm):,} players): correlation "
              + ", ".join(f"{c} {np.corrcoef(mm[c], mm[c + '_ridge'])[0, 1]:.3f}" for c in ("off", "def", "net")))

    show = tab[tab["toi_5v5"] >= 500].sort_values("net", ascending=False)
    cols = ["name", "position", "toi_5v5", "off", "off_sd", "def", "def_sd", "net", "net_sd"]
    fmt = lambda d: d[cols].assign(toi_5v5=d["toi_5v5"].round(0)).round(3).to_string(index=False)
    print(f"\nTop 15 by net impact (500+ 5v5 minutes; {len(show):,} players):")
    print(fmt(show.head(15)))
    print("\nBottom 10:")
    print(fmt(show.tail(10)))
    sig = show[(show["net"] - 1.645 * show["net_sd"]) > 0]
    print(f"\nPlayers whose net impact is above 0 with 95% confidence: {len(sig)} of {len(show)}; "
          f"median net sd {show['net_sd'].median():.3f}")

    if args.teams:
        tt = lr.team_table(f).sort_values("net", ascending=False)
        print("\nTeam-season system effects (xG/60), top and bottom 5. If their sds are close to the learned team "
              "spread, the stints cannot place them (team and roster move together).")
        print(tt.head(5).round(3).to_string(index=False))
        print(tt.tail(5).round(3).to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = MODEL_DIR / f"lineup_rapm_{(end - pd.Timedelta(days=1)).date()}{'_teams' if args.teams else ''}"
    tab.sort_values("net", ascending=False).to_csv(f"{stem}.csv", index=False)
    np.save(f"{stem}_cov.npy", f["cov"].astype(np.float32))
    meta = {"start": str(start.date()), "end": str(end.date()), "days": args.days, "min_toi": args.min_toi,
            "teams": args.teams, "spreads": {k: v for k, v in th.items() if not k.startswith("_")},
            "converged": th.get("_converged"), "names": f["info"]["names"],
            "cv": None if cv is None else cv.to_dict(orient="records")}
    with open(f"{stem}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    if args.teams:
        lr.team_table(f).to_csv(f"{stem}_team_effects.csv", index=False)
    print(f"\nSaved {stem.name}.csv, .json, _cov.npy to {MODEL_DIR}")


if __name__ == "__main__":
    main()