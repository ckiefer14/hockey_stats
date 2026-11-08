"""
log_closing.py
==============
Closing lines: wait through the day and snapshot the moneylines about 10 minutes before each
group of start times, so every game gets a line taken just before puck drop.

Home: src/pipelines/log_closing.py

Usage (from the project root; start it any time before the first game, e.g. at 11 am):
    py -m src.pipelines.log_closing                 # today's games, 10 minutes before each start
    py -m src.pipelines.log_closing --lead 5        # 5 minutes before instead
    py -m src.pipelines.log_closing --dry-run       # show the plan (start times, snapshot times), no credits

1. Start times from The Odds API events list (free, no credits) for games starting in the next
   --hours hours (default 16).
2. Games starting within --group minutes of each other share one snapshot (NHL nights have
   3-5 start times). Each snapshot = 1 credit (moneylines, US books; --markets adds more).
3. Sleeps until each snapshot time, takes it, appends to output/forward/odds_<season>.csv
   (same file and format as log_odds), prints market vs model for those games, and exits after
   the last group. A start time that moves is picked up: the list is refreshed before each one.
If the computer was asleep and a snapshot time passed, it takes the snapshot at once if the
games have not started yet, otherwise skips them and says so. Stops if fewer than
--min-credits credits are left this month.

Keep the window open (or let Windows Task Scheduler start it; see DAILY_RUN.md). The computer
must be awake at the snapshot times.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.db import get_connection
from src.data.odds import TEAM_ABBR, consensus, events_to_rows, fetch_events, fetch_odds, match_games
from src.pipelines.log_odds import latest_predictions, save_rows, schedule

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def upcoming(hours: float) -> pd.DataFrame:
    """Games starting in the next `hours` hours: event_id, home, away, start (UTC)."""
    ev = fetch_events()
    rows = [{"event_id": e["id"], "home": TEAM_ABBR.get(e["home_team"], e["home_team"]),
             "away": TEAM_ABBR.get(e["away_team"], e["away_team"]),
             "start": pd.Timestamp(e["commence_time"]).tz_convert("UTC")} for e in ev]
    df = pd.DataFrame(rows, columns=["event_id", "home", "away", "start"])
    if df.empty:
        return df
    t0 = pd.Timestamp(now_utc())
    return df[(df["start"] > t0) & (df["start"] <= t0 + pd.Timedelta(hours=hours))].sort_values("start")


def groups(df: pd.DataFrame, within_min: float) -> list[pd.DataFrame]:
    """Games whose start times are within `within_min` minutes of the group's first game."""
    out, cur, first = [], [], None
    for _, r in df.iterrows():
        if first is not None and (r["start"] - first).total_seconds() / 60 > within_min:
            out.append(pd.DataFrame(cur))
            cur, first = [], None
        if first is None:
            first = r["start"]
        cur.append(r)
    if cur:
        out.append(pd.DataFrame(cur))
    return out


def local(ts) -> str:
    """Clock time on this computer's time zone, e.g. '6:50 PM'."""
    return pd.Timestamp(ts).to_pydatetime().astimezone().strftime("%I:%M %p").lstrip("0")


def snapshot(event_ids: set, markets: str, regions: str, bookmakers: str | None) -> tuple[int, str | None]:
    """Take one snapshot, keep the rows of the given games (all games are logged; harmless)."""
    events, quota = fetch_odds(markets, regions, bookmakers)
    rows = events_to_rows(events, snapshot_at=now_utc().isoformat(timespec="seconds"))
    if rows.empty:
        return 0, quota.get("remaining")
    conn = get_connection()
    rows = match_games(rows, schedule(conn))
    conn.close()
    if rows.empty:
        return 0, quota.get("remaining")
    save_rows(rows)
    mine = rows[rows["event_id"].isin(event_ids)]
    mk = consensus(mine, before_start=True).merge(latest_predictions(), on="game_id", how="left")
    info = mine.drop_duplicates("game_id")[["game_id", "commence_time", "home", "away"]]
    for r in info.merge(mk, on="game_id").sort_values("commence_time").itertuples():
        model = f"model {r.p_home:.1%} ({100 * (r.p_home - r.p_market):+.1f} pts)" if pd.notna(r.p_home) else "model: no prediction"
        print(f"      {r.away:>3s} @ {r.home:<3s}  market {r.p_market:.1%} ({r.books} books, "
              f"{r.minutes_before:.0f} min before start)   {model}")
    return len(rows), quota.get("remaining")


def main():
    parser = argparse.ArgumentParser(description="Snapshot moneylines just before each group of start times.")
    parser.add_argument("--lead", type=float, default=10, help="minutes before the start (default 10)")
    parser.add_argument("--group", type=float, default=20,
                        help="games starting within this many minutes share a snapshot (default 20)")
    parser.add_argument("--hours", type=float, default=16, help="look this far ahead (default 16)")
    parser.add_argument("--markets", default="h2h")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", default=None)
    parser.add_argument("--min-credits", type=int, default=20, help="stop if fewer credits are left (default 20)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only (uses no credits)")
    args = parser.parse_args()

    games = upcoming(args.hours)
    if games.empty:
        print(f"No NHL games starting in the next {args.hours:.0f} hours.")
        return
    plan = groups(games, args.group)
    print(f"{len(games)} games in {len(plan)} start-time groups; a snapshot {args.lead:.0f} min before each "
          f"(1 credit each for {args.markets}, {args.regions}):")
    for g in plan:
        when = g["start"].min() - pd.Timedelta(minutes=args.lead)
        print(f"  snapshot {local(when)}  for {', '.join(f'{a}@{h} {local(s)}' for a, h, s in zip(g['away'], g['home'], g['start']))}")
    if args.dry_run:
        return

    done = set()
    while True:
        games = upcoming(args.hours)                         # refresh: start times can move
        games = games[~games["event_id"].isin(done)]
        if games.empty:
            break
        g = groups(games, args.group)[0]
        first = g["start"].min()
        when = first - pd.Timedelta(minutes=args.lead)
        wait = (when - pd.Timestamp(now_utc())).total_seconds()
        if wait > 0:
            print(f"\n{local(now_utc())}  waiting until {local(when)} "
                  f"for {len(g)} game{'s' if len(g) != 1 else ''} starting {local(first)}...", flush=True)
            # sleep in short steps so a sleeping laptop or a moved start time is noticed
            while (when - pd.Timestamp(now_utc())).total_seconds() > 0:
                time.sleep(min(60.0, max((when - pd.Timestamp(now_utc())).total_seconds(), 1.0)))
            fresh = upcoming(args.hours)
            fresh = fresh[fresh["event_id"].isin(set(g["event_id"]))]
            if len(fresh) and fresh["start"].min() - pd.Timedelta(minutes=args.lead) > pd.Timestamp(now_utc()) + pd.Timedelta(minutes=2):
                continue                                     # start time moved later: plan again
        if pd.Timestamp(now_utc()) >= first:
            print(f"  missed: {', '.join(g['away'] + '@' + g['home'])} already started (computer asleep?)")
            done |= set(g["event_id"])
            continue
        print(f"{local(now_utc())}  snapshot for games starting {local(first)}:",
              flush=True)
        try:
            n, left = snapshot(set(g["event_id"]), args.markets, args.regions, args.bookmakers)
        except Exception as e:                                # noqa: BLE001  network hiccup: one retry
            print(f"  snapshot failed ({type(e).__name__}: {e}); retrying in 60 s")
            time.sleep(60)
            n, left = snapshot(set(g["event_id"]), args.markets, args.regions, args.bookmakers)
        print(f"  logged {n} rows; credits left this month: {left}")
        done |= set(g["event_id"])
        if left is not None and int(float(left)) < args.min_credits:
            print(f"Fewer than {args.min_credits} credits left this month: stopping (no more snapshots today).")
            return
    print("Done: every game today has a closing snapshot (or was reported missed).")


if __name__ == "__main__":
    main()
