"""
log_odds.py
===========
Log sportsbook moneylines for upcoming NHL games next to the forward-test predictions.

Home: src/pipelines/log_odds.py

Usage (from the project root):
    py -m src.pipelines.log_odds                               # moneylines, US books (1 credit)
    py -m src.pipelines.log_odds --markets h2h,totals,spreads  # + totals and puck lines (3 credits)
    py -m src.pipelines.log_odds --bookmakers draftkings,fanduel

Appends to output/forward/odds_<season>.csv: one row per game x bookmaker x snapshot,
with the American moneylines, the book's margin and P(home win) with the margin removed.
Run it after predict_day each morning. For closing lines, run log_closing.py (it waits and
snapshots about 10 minutes before each group of start times): score_forward uses, for each
game, the last snapshot taken before its start time (the API gives start times, so this is exact).

Prints, per game, the market's P(home win) (median over books, margin removed) next to
the model's logged prediction.
"""

import argparse

import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.data.odds import fetch_odds, events_to_rows, match_games, consensus

FORWARD_DIR = OUTPUT_DIR / "forward"


def schedule(conn) -> pd.DataFrame:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(schedule)")}
    where = "WHERE game_type = 2" if "game_type" in cols else ""
    return pd.read_sql(f"""SELECT game_id, season, game_date, home_team_abbrev AS home, away_team_abbrev AS away
                           FROM schedule {where}""", conn)


def latest_predictions() -> pd.DataFrame:
    files = sorted(FORWARD_DIR.glob("predictions_*.csv"))
    if not files:
        return pd.DataFrame(columns=["game_id", "p_home"])
    log = pd.concat((pd.read_csv(f, usecols=["game_id", "p_home", "predicted_at"]) for f in files))
    return log.sort_values("predicted_at").groupby("game_id").tail(1)[["game_id", "p_home"]]


def save_rows(rows: pd.DataFrame) -> None:
    """Append snapshot rows to output/forward/odds_<season>.csv (one file per season)."""
    FORWARD_DIR.mkdir(parents=True, exist_ok=True)
    for season, part in rows.groupby("season"):
        path = FORWARD_DIR / f"odds_{season}.csv"
        if path.exists():                         # columns can differ between runs (markets), so merge
            part = pd.concat([pd.read_csv(path), part], ignore_index=True)
        part.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Log NHL sportsbook odds for the forward test.")
    parser.add_argument("--markets", default="h2h", help="h2h (default), or e.g. h2h,totals,spreads")
    parser.add_argument("--regions", default="us", help="bookmaker region (default us)")
    parser.add_argument("--bookmakers", default=None, help="only these books, e.g. draftkings,fanduel")
    args = parser.parse_args()

    events, quota = fetch_odds(args.markets, args.regions, args.bookmakers)
    rows = events_to_rows(events)
    conn = get_connection()
    rows = match_games(rows, schedule(conn))
    conn.close()
    if rows.empty:
        print(f"No NHL games with odds right now. Credits left: {quota['remaining']}")
        return

    save_rows(rows)

    # Summary: market vs model, games starting soonest first
    mk = consensus(rows, before_start=False).merge(latest_predictions(), on="game_id", how="left")
    info = rows.drop_duplicates("game_id")[["game_id", "game_date", "commence_time", "home", "away"]]
    mk = info.merge(mk, on="game_id").sort_values("commence_time")
    print(f"{len(mk)} games, {rows['book'].nunique()} books. Market P(home win) = median over books, margin removed.\n")
    for r in mk.itertuples():
        model = f"model {r.p_home:.1%}  (model - market {100 * (r.p_home - r.p_market):+.1f} pts)" \
            if pd.notna(r.p_home) else "model: not predicted yet"
        print(f"  {str(r.game_date)[:10]}  {r.away:>3s} @ {r.home:<3s}  market {r.p_market:.1%} "
              f"({r.books} books, margin {100 * (r.overround - 1):.1f}%)   {model}")
    print(f"\nLogged {len(rows)} rows to {FORWARD_DIR}. Credits used by this call: {quota['last']}, "
          f"left this month: {quota['remaining']}")


if __name__ == "__main__":
    main()