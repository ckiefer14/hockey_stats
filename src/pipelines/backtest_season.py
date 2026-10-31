"""
backtest_season.py
==================
Run the Phase 7 walk-forward backtest of Track B for one season.

Home: src/pipelines/backtest_season.py

Usage (from the project root):
    py -m src.pipelines.backtest_season --season 20252026
    py -m src.pipelines.backtest_season --season 20252026 --window 365
    py -m src.pipelines.backtest_season --season 20242025 --window 730 --refit-days 7
    py -m src.pipelines.backtest_season --season 20252026 --model xg --window 365
    py -m src.pipelines.backtest_season --season 20252026 --model xg --window 730 --half-life 120

Writes output/backtests/trackb_w<window>_<season>.csv (predictions, appended after every
refit; stop any time and rerun to resume) and, when the season is complete,
trackb_w<window>_<season>_scored.csv (predictions + how each game ended), which the
Phase 7 notebook reads. Use --restart to delete both and start over.

Time: about 50 s per refit (about 26 per season) plus the pre-season fit.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.validation.backtest import walk_forward, model_label

BACKTEST_DIR = OUTPUT_DIR / "backtests"


def add_results(conn, preds: pd.DataFrame) -> pd.DataFrame:
    """Attach how each game actually ended (final score, REG/OT/SO, winner)."""
    final = pd.read_sql("""
        SELECT id AS game_id, home_score AS final_home, away_score AS final_away, period_type AS decided
        FROM games WHERE game_type = 2
    """, conn)
    out = preds.drop(columns=[c for c in ["final_home", "final_away", "decided", "home_won"] if c in preds],
                     errors="ignore").merge(final, on="game_id", how="left")
    out["home_won"] = (out["final_home"] > out["final_away"]).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of Track B for one season.")
    parser.add_argument("--season", required=True, help="test season, e.g. 20252026")
    parser.add_argument("--window", type=int, default=730, help="training window in days (default 730)")
    parser.add_argument("--refit-days", type=int, default=7, help="days between refits (default 7)")
    parser.add_argument("--score-sims", type=int, default=5000, help="score-state simulations per game")
    parser.add_argument("--model", choices=["goals", "xg"], default="goals",
                        help="goals = Track B v1; xg = Track B v1.1 (ratings from xG + goals)")
    parser.add_argument("--half-life", type=float, default=None,
                        help="v1.1 only: recency half-life in days (older games count less)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart", action="store_true", help="delete existing results and start over")
    args = parser.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.half_life and args.model != "xg":
        raise SystemExit("--half-life is only supported with --model xg")
    label = model_label(args.model, args.window, args.half_life)
    out = BACKTEST_DIR / f"{label}_{args.season}.csv"
    scored = BACKTEST_DIR / f"{label}_{args.season}_scored.csv"
    if args.restart:
        for f in (out, scored):
            if f.exists():
                os.remove(f)

    start = time.time()
    conn = get_connection()
    walk_forward(conn, args.season, window_days=args.window, refit_days=args.refit_days,
                 score_sims=args.score_sims, seed=args.seed, out_path=out,
                 model=args.model, half_life=args.half_life)

    preds = add_results(conn, pd.read_csv(out, parse_dates=["game_date"]))
    conn.close()
    preds.to_csv(scored, index=False)

    y, p = preds["home_won"], preds["p_home"].clip(1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"\n{len(preds):,} games predicted in {(time.time() - start) / 60:.1f} min. "
          f"Log loss {ll:.4f} (coin flip 0.6931); picks the winner {((p > 0.5) == y).mean():.1%}")
    print(f"Saved {scored}")


if __name__ == "__main__":
    main()