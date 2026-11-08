"""
compare_vs_live.py
==================
Compare backtest variants with the live model on exactly the games they all share, per
season and pooled. Works when some models have more backtest seasons than others (where the
Phase 7 notebook would leave them out).

Home: scripts/compare_vs_live.py

Usage (from the project root):
    py .\\scripts\\compare_vs_live.py                                    # every variant built on the live model
    py .\\scripts\\compare_vs_live.py --variants trackbtv_w730_r_lu trackbtv_w730_rg
    py .\\scripts\\compare_vs_live.py --live trackbtv_w730_r --seasons 20232024 20242025 20252026

For each variant: games compared, log loss of both, the paired difference with a 90% range
(bootstrap over games; negative = variant better), calibration slope, verdict by the usual
rule (adopt only if the whole range is below zero), and the same per season.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import OUTPUT_DIR
from src.validation.scoring import calibration_slope, log_loss, paired_difference

BT = OUTPUT_DIR / "backtests"


def load(label: str, seasons: list[str] | None) -> pd.DataFrame:
    parts = []
    for f in sorted(BT.glob(f"{label}_*_scored.csv")):
        season = f.name[len(label) + 1:].split("_")[0]
        if not season.isdigit() or (seasons and season not in seasons):
            continue
        d = pd.read_csv(f, usecols=["game_id", "p_home", "home_won"]).drop_duplicates("game_id", keep="last")
        parts.append(d.assign(season=season))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["game_id", "p_home", "home_won", "season"])


def verdict(r: dict) -> str:
    return "clearly better: adopt" if r["hi"] < 0 else "clearly worse" if r["lo"] > 0 else "within luck"


def main():
    parser = argparse.ArgumentParser(description="Backtest variants vs the live model on shared games.")
    parser.add_argument("--live", default="trackbtv_w730_r", help="label of the live model's backtest")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="labels to compare (default: every trackbtv_w730_r* label other than the live one)")
    parser.add_argument("--seasons", nargs="*", default=None, help="seasons to use (default: all present)")
    args = parser.parse_args()

    labels = sorted({f.name.rsplit("_", 2)[0] for f in BT.glob("trackb*_scored.csv")})
    variants = args.variants or [l for l in labels if l.startswith(args.live) and l != args.live]
    live = load(args.live, args.seasons)
    if live.empty:
        raise SystemExit(f"No backtest files for the live model {args.live}")
    print(f"Live model {args.live}: seasons {', '.join(sorted(live['season'].unique()))}, {len(live):,} games\n")
    rows = []
    for v in variants:
        d = load(v, args.seasons)
        if d.empty:
            print(f"{v}: no files")
            continue
        m = live.merge(d[["game_id", "p_home"]], on="game_id", suffixes=("_live", "_var"))
        r = paired_difference(m, "p_home_var", "p_home_live")
        slope = calibration_slope(m["p_home_var"], m["home_won"])[1]
        rows.append({"variant": v, "seasons": ", ".join(sorted(m["season"].unique())), "games": len(m),
                     "live": log_loss(m["p_home_live"], m["home_won"]), "variant LL": log_loss(m["p_home_var"], m["home_won"]),
                     "difference": r["difference"], "lo": r["lo"], "hi": r["hi"], "slope": slope, "verdict": verdict(r)})
        for s, g in m.groupby("season"):
            rs = paired_difference(g, "p_home_var", "p_home_live")
            print(f"  {v} {s}: {len(g):,} games, {rs['difference']:+.4f} [{rs['lo']:+.4f}, {rs['hi']:+.4f}]")
    if rows:
        t = pd.DataFrame(rows)
        pd.set_option("display.width", 220)
        print("\nPooled over the shared games (negative = variant better than live):")
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
