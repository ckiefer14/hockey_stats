"""
run_baselines.py
================
Baseline predictions (home rate, Elo, rolling xG) for Phase 7 test seasons.

Home: src/pipelines/run_baselines.py

Usage (from the project root):
    py -m src.pipelines.run_baselines --seasons 20242025 20252026

Writes output/backtests/baselines_<season>.csv and baselines_<season>_info.json
(fitted settings). Takes seconds to a few minutes (the Elo tuning grid).
"""

import argparse
import json

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.validation.baselines import all_baselines
from src.validation.scoring import log_loss, accuracy

BACKTEST_DIR = OUTPUT_DIR / "backtests"


def main():
    parser = argparse.ArgumentParser(description="Baseline predictions for Phase 7.")
    parser.add_argument("--seasons", nargs="+", required=True, help="test seasons, e.g. 20242025 20252026")
    args = parser.parse_args()
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    for season in args.seasons:
        preds, info = all_baselines(conn, season)
        preds.to_csv(BACKTEST_DIR / f"baselines_{season}.csv", index=False)
        with open(BACKTEST_DIR / f"baselines_{season}_info.json", "w") as f:
            json.dump(info, f, indent=2)
        e, r = info["elo"], info["rolling_xg"]
        print(f"\n{season}: {len(preds):,} games")
        print(f"  home rate {info['home_rate']:.3f}")
        print(f"  Elo: k {e['k']}, home {e['home']}, carry {e['carry']} (tuning log loss {e['tuning_log_loss']:.4f})")
        print(f"  rolling xG: intercept {r['intercept']:.3f}, slope {r['slope']:.3f} per xG/60 ({r['train_games']:,} training games)")
        for name, col in [("home rate", "p_home_rate"), ("Elo", "p_elo"), ("rolling xG", "p_rolling_xg")]:
            print(f"  {name:10s} log loss {log_loss(preds[col], preds['home_won']):.4f}   "
                  f"accuracy {accuracy(preds[col], preds['home_won']):.1%}")
    conn.close()
    print(f"\nSaved to {BACKTEST_DIR}")


if __name__ == "__main__":
    main()