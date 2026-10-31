"""
score_forward.py
================
Score the forward-test predictions (predict_day.py) against the games played so far.

Home: src/pipelines/score_forward.py

Usage (from the project root, after load_all.py has loaded the results):
    py -m src.pipelines.score_forward                     # current season log
    py -m src.pipelines.score_forward --season 20262027

For each game, the prediction scored is the last one made on or before the game date
(so a re-prediction with confirmed starters counts; one made after the game never does).
Reports log loss, Brier, accuracy, calibration slope (with 90% bootstrap ranges) next to
the historical home-win rate and the Phase 7 backtest, calibration by probability bin,
results by week, score outputs, and how often the default starting goalie was right.
Writes output/forward/scored_<season>.csv.
"""

import argparse

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.validation.scoring import score_models, calibration_table, score_outputs, log_loss

FORWARD_DIR = OUTPUT_DIR / "forward"
BACKTEST_LOG_LOSS = 0.6773          # Phase 7: Track B v1.1, 1-yr window, 2,624 held-out games
COIN_FLIP = float(np.log(2))


def latest_before_game(log: pd.DataFrame) -> pd.DataFrame:
    """One prediction per game: the last one made on or before the game date."""
    log = log.copy()
    log["made_date"] = pd.to_datetime(log["predicted_at"]).dt.normalize()
    log["game_date"] = pd.to_datetime(log["game_date"])
    ok = log[log["made_date"] <= log["game_date"]]
    late = len(log) - len(ok)
    if late:
        print(f"Ignored {late} predictions made after their game date.")
    return ok.sort_values("predicted_at").groupby("game_id").tail(1)


def add_results(conn, preds: pd.DataFrame) -> pd.DataFrame:
    final = pd.read_sql("""
        SELECT id AS game_id, home_score AS final_home, away_score AS final_away, period_type AS decided
        FROM games WHERE game_type = 2
    """, conn)
    out = preds.merge(final, on="game_id", how="inner")
    out["home_won"] = (out["final_home"] > out["final_away"]).astype(int)
    return out


def actual_starters(conn) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT game_id, team_abbreviation AS team, player_id AS actual_goalie
        FROM goalie_game_stats WHERE starter = 1
    """, conn)


def main():
    parser = argparse.ArgumentParser(description="Score forward-test predictions.")
    parser.add_argument("--season", default=None, help="season, e.g. 20262027 (default: newest log)")
    args = parser.parse_args()

    logs = sorted(FORWARD_DIR.glob("predictions_*.csv"))
    if not logs:
        raise SystemExit(f"No prediction logs in {FORWARD_DIR}. Run: py -m src.pipelines.predict_day")
    path = FORWARD_DIR / f"predictions_{args.season}.csv" if args.season else logs[-1]
    season = path.stem.split("_")[1]
    log = pd.read_csv(path)
    preds = latest_before_game(log)

    conn = get_connection()
    df = add_results(conn, preds)
    starters = actual_starters(conn)
    conn.close()
    pending = len(preds) - len(df)
    print(f"Season {season}: {len(preds):,} games predicted, {len(df):,} played and loaded, {pending:,} pending")
    if df.empty:
        print("Nothing to score yet: run load_all.py after the games are played.")
        return
    print(f"Fits used: {', '.join(sorted(df['fit_file'].unique()))}")

    # 1. Win probability vs the historical home rate, with the backtest as reference
    table = score_models(df, {"Track B (live)": "p_home", "Home rate": "p_home_rate"},
                         reference="Home rate", n_boot=1000)
    print("\nP(home win):")
    print(table.drop(columns=["games", "intercept"]).round(4).to_string(index=False))
    ll = log_loss(df["p_home"], df["home_won"])
    se = np.std(-(df["home_won"] * np.log(df["p_home"]) + (1 - df["home_won"]) * np.log(1 - df["p_home"]))) / np.sqrt(len(df))
    print(f"Live log loss {ll:.4f} ± {1.645 * se:.4f} (90%); Phase 7 backtest {BACKTEST_LOG_LOSS:.4f}; coin flip {COIN_FLIP:.4f}")
    if len(df) < 300:
        print(f"Only {len(df)} games: the range is wide. Differences of 0.005 need roughly 1,000+ games to show.")

    # 2. Calibration by probability bin
    bins = 5 if len(df) < 400 else 10
    print(f"\nCalibration ({bins} equal-size groups):")
    print(calibration_table(df["p_home"], df["home_won"], bins=bins).round(3).to_string())

    # 3. By week
    df["week"] = pd.to_datetime(df["game_date"]).dt.to_period("W").dt.start_time.dt.date
    wk = df.groupby("week").apply(lambda g: pd.Series({
        "games": len(g), "log loss": log_loss(g["p_home"], g["home_won"]),
        "mean predicted": g["p_home"].mean(), "home won": g["home_won"].mean()}), include_groups=False)
    print("\nBy week:")
    print(wk.round(4).to_string())

    # 4. Score outputs (totals, puck lines)
    print("\nScore outputs (Brier; RPS for totals):")
    print(score_outputs(df).round(4).to_string(index=False))

    # 5. Default starting goalie: how often was the guess right?
    guesses = []
    for side in ("home", "away"):
        g = df[df[f"{side}_goalie_source"] == "default"][["game_id", side, f"{side}_goalie"]]
        g = g.rename(columns={side: "team", f"{side}_goalie": "guess"})
        guesses.append(g)
    g = pd.concat(guesses).merge(starters, on=["game_id", "team"], how="inner")
    if len(g):
        right = (g["guess"] == g["actual_goalie"])
        print(f"\nDefault starter right {right.mean():.1%} of the time ({right.sum()} of {len(g)} team-games)")
        df_wrong = set(g.loc[~right, "game_id"])
        if df_wrong:
            w = df[df["game_id"].isin(df_wrong)]
            print(f"  games with a wrong guess: {len(w)}, log loss {log_loss(w['p_home'], w['home_won']):.4f} "
                  f"vs {log_loss(df[~df['game_id'].isin(df_wrong)]['p_home'], df[~df['game_id'].isin(df_wrong)]['home_won']):.4f} otherwise")

    out = FORWARD_DIR / f"scored_{season}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()