"""
make_playoff_charts.py
======================
Playoff odds charts from the files playoff_odds.py writes (it also calls this at the end).

Home: src/pipelines/make_playoff_charts.py

Usage (from the project root):
    py -m src.pipelines.make_playoff_charts                    # latest run + the season so far
    py -m src.pipelines.make_playoff_charts --date 2026-11-08  # the board for an earlier run

Writes to output/playoffs/:
    playoff_board_<date>.png       one run: P(playoffs) per team by division, split by route in
    playoff_trend_<season>.png     every run this season: P(playoffs) lines per division
    cup_odds_<date>.png            one run: every team's chance to get through each playoff round
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from config.settings import OUTPUT_DIR

PLAYOFF_DIR = OUTPUT_DIR / "playoffs"


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_charts(date: str | None = None) -> list[Path]:
    from src.viz.playoff_chart import plot_playoff_board, plot_playoff_trend
    runs = sorted(PLAYOFF_DIR.glob("playoff_odds_????-??-??.csv"))
    if date:
        runs = [f for f in runs if f.stem[-10:] == str(date)]
    if not runs:
        raise SystemExit(f"No playoff_odds_{date or '<date>'}.csv in {PLAYOFF_DIR}. Run playoff_odds first.")
    run = runs[-1]
    tab = pd.read_csv(run)
    day = run.stem[-10:]
    note = f"fit {tab['fit_file'].iloc[0]}" if "fit_file" in tab else ""
    out = [_save(plot_playoff_board(tab, day, note=note), PLAYOFF_DIR / f"playoff_board_{day}.png")]
    if "p_cup" in tab and tab["p_cup"].notna().all():
        from src.viz.playoff_chart import plot_cup_odds
        out.append(_save(plot_cup_odds(tab, day, note=note), PLAYOFF_DIR / f"cup_odds_{day}.png"))

    hist_files = sorted(PLAYOFF_DIR.glob("playoff_odds_history_*.csv"))
    hist_files = [f for f in hist_files if day in set(pd.read_csv(f, usecols=["date"])["date"].astype(str))]
    if hist_files:
        h = hist_files[-1]
        season = h.stem.split("_")[-1]
        hist = pd.read_csv(h)
        hist = hist[hist["date"].astype(str) <= day]
        out.append(_save(plot_playoff_trend(hist, season), PLAYOFF_DIR / f"playoff_trend_{season}.png"))
    return out


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Playoff odds charts.")
    parser.add_argument("--date", help="run date of the board (default: the latest run)")
    args = parser.parse_args()
    for p in make_charts(args.date):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()