"""
check_backtests.py
==================
Sanity check of the walk-forward backtest files before running the notebooks.

Home: scripts/check_backtests.py

Usage (from the project root):
    py .\\scripts\\check_backtests.py

For every output/backtests/<label>_<season>_scored.csv: rows, distinct games, duplicated
games, games with no prediction, and how many of the reference model's games
(trackbtv_w730, plain v2.0) it is missing. Also checks the game table itself for duplicate
game ids. A file is OK only if it has exactly one row per game and covers the same games
as the reference. Duplicated or partial files make the notebooks' comparisons wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.processing.game_table import build_game_table

BT = OUTPUT_DIR / "backtests"
REFERENCE = "trackbtv_w730"


def main():
    files = sorted(BT.glob("trackb*_scored.csv"))
    if not files:
        raise SystemExit(f"No scored backtest files in {BT}")
    ref = {}
    for f in files:
        label, season = f.name[: -len("_scored.csv")].rsplit("_", 1)
        if label == REFERENCE:
            ref[season] = set(pd.read_csv(f, usecols=["game_id"])["game_id"])

    rows = []
    for f in files:
        label, season = f.name[: -len("_scored.csv")].rsplit("_", 1)
        d = pd.read_csv(f, usecols=lambda c: c in {"game_id", "p_home", "refit_date"})
        games = set(d["game_id"])
        missing = len(ref[season] - games) if season in ref else None
        raw = BT / f"{label}_{season}.csv"
        raw_rows = sum(1 for _ in open(raw)) - 1 if raw.exists() else None
        rows.append({
            "label": label, "season": season, "rows": len(d), "games": len(games),
            "duplicated": int(d["game_id"].duplicated().sum()), "no p_home": int(d["p_home"].isna().sum()),
            f"missing vs {REFERENCE}": missing, "raw file rows": raw_rows,
            "refits": d["refit_date"].nunique() if "refit_date" in d else None,
        })
    t = pd.DataFrame(rows)
    t["OK"] = (t["duplicated"] == 0) & (t["no p_home"] == 0) & (t[f"missing vs {REFERENCE}"].fillna(0) == 0)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))

    bad = t[~t["OK"]]
    if len(bad):
        print("\nNot OK:")
        for _, r in bad.iterrows():
            why = []
            if r["duplicated"]:
                why.append(f"{r['duplicated']} duplicated games (two runs wrote to the same file?)")
            if r["no p_home"]:
                why.append(f"{r['no p_home']} games with no prediction")
            if pd.notna(r[f"missing vs {REFERENCE}"]) and r[f"missing vs {REFERENCE}"]:
                why.append(f"{int(r[f'missing vs {REFERENCE}'])} games missing (run not finished, or stopped?)")
            print(f"  {r['label']} {r['season']}: " + "; ".join(why))
        print("Fix: make sure no other backtest is running, then rerun that season with --restart.")
    else:
        print("\nAll backtest files OK.")

    conn = get_connection()
    seasons = sorted(t["season"].unique())
    gt, _ = build_game_table(conn, seasons)
    conn.close()
    dup = int(gt["game_id"].duplicated().sum())
    print(f"\nGame table ({', '.join(seasons)}): {len(gt):,} rows, {gt['game_id'].nunique():,} games, "
          f"{dup} duplicated" + ("  <-- problem in the game table" if dup else "  (OK)"))


if __name__ == "__main__":
    main()