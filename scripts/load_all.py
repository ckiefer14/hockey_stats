# scripts/load_all.py
r"""
Load teams, rosters, schedules and every finished game (boxscore, play-by-play,
shifts) into the database. Safe to rerun: games already loaded are skipped, and
schedules are saved with INSERT OR REPLACE, so games that finished since the last
run get picked up.

Usage (from the project root):
    py .\scripts\load_all.py                    # seasons from config/settings.py
    py .\scripts\load_all.py --all-schedules    # refetch schedules even for complete seasons

Rate limits: the NHL API answers "429 Too Many Requests" when called too fast.
Every request here waits SLEEP_BETWEEN_CALLS seconds first, and a 429 is retried
after the wait the API asks for (Retry-After), or 10 s, 20 s, 40 s, ... if it doesn't say.
"""
import argparse
import sys
import time
sys.path.insert(0, ".")

import requests

from config.settings import SEASONS          # list of seasons to load, e.g. [..., "20262027"]
from src.data.nhl_api import (
    fetch_teams, fetch_roster, fetch_schedule, fetch_boxscore,
    fetch_pbp_json, parse_shot_events, parse_game_events, fetch_shifts
)
from src.data.db import (
    get_connection, create_tables, populate_team_ids,
    store_teams, store_players,
    store_schedule, get_finished_game_ids,
    store_game, store_all_skater_stats, store_all_goalie_stats,
    store_shot_events, store_game_events, store_shifts, game_already_loaded
)
from src.processing.shot_context import build_shot_context

FAILURE_LOG = "data_load_failures.log"
SLEEP_BETWEEN_CALLS = 1.0       # seconds before every API request
MAX_TRIES = 6                   # attempts per request when rate-limited
BACKOFF_START = 10.0            # first wait after a 429 if the API gives no Retry-After


def log_failure(msg: str) -> None:
    print(msg)
    with open(FAILURE_LOG, "a") as f:
        f.write(f"{msg}\n")


def call(fn, *args, **kwargs):
    """Call an API function politely: pause first, retry on 429 with a growing wait."""
    wait = BACKOFF_START
    for attempt in range(1, MAX_TRIES + 1):
        time.sleep(SLEEP_BETWEEN_CALLS)
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            resp = e.response
            if resp is None or resp.status_code != 429 or attempt == MAX_TRIES:
                raise
            retry_after = resp.headers.get("Retry-After")
            pause = float(retry_after) if retry_after and retry_after.isdigit() else wait
            print(f"    rate-limited (429); waiting {pause:.0f} s (attempt {attempt}/{MAX_TRIES})")
            time.sleep(pause)
            wait *= 2


def complete_seasons(conn) -> set[str]:
    """Seasons already in the schedule table with every game finished (state OFF)."""
    rows = conn.execute("""
        SELECT season, COUNT(*), SUM(CASE WHEN game_state = 'OFF' THEN 0 ELSE 1 END)
        FROM schedule GROUP BY season
    """).fetchall()
    return {str(season) for season, n, not_done in rows if n > 0 and not_done == 0}


def main():
    parser = argparse.ArgumentParser(description="Load NHL data into the database.")
    parser.add_argument("--all-schedules", action="store_true",
                        help="refetch schedules even for seasons that are already complete")
    args = parser.parse_args()

    conn = get_connection()
    create_tables(conn)

    # 1. Teams
    teams = call(fetch_teams)
    store_teams(conn, teams)
    print(f"Stored {len(teams)} teams")

    # 2. Rosters (current)
    for team in teams:
        try:
            players = call(fetch_roster, team.abbreviation)
            store_players(conn, players)
            print(f"  {team.abbreviation}: {len(players)} players")
        except Exception as e:
            log_failure(f"Roster fetch failed for {team.abbreviation}: {e}")

    # 3. Schedules: skip seasons that are already complete in the database
    wanted = [str(s) for s in SEASONS]
    done = set() if args.all_schedules else complete_seasons(conn) & set(wanted)
    if done:
        print(f"Schedules already complete, skipped: {', '.join(sorted(done))}")
    for season in [s for s in wanted if s not in done]:
        stored = 0
        for team in teams:
            try:
                schedule = call(fetch_schedule, team.abbreviation, season=season)
                store_schedule(conn, schedule)
                stored += len(schedule)
            except Exception as e:
                log_failure(f"Schedule fetch failed for {team.abbreviation} {season}: {e}")
        conn.commit()
        print(f"Season {season}: stored {stored} team-schedule entries")

    # 4. Populate team IDs
    populate_team_ids(conn)
    print("Populated team IDs")

    # 5. Unique finished game IDs across all teams, and which are new
    all_game_ids = set()
    for team in teams:
        all_game_ids.update(get_finished_game_ids(conn, team.abbreviation))
    game_ids = sorted(all_game_ids)
    new_ids = [g for g in game_ids if not game_already_loaded(conn, g)]
    print(f"Finished games in schedule: {len(game_ids):,}; new to load: {len(new_ids):,}")

    # 6. Per-game pull with failure logging
    for i, game_id in enumerate(new_ids):
        try:
            result = call(fetch_boxscore, game_id)
            store_game(conn, result["game"])
            store_all_skater_stats(conn, result["skaters"])
            store_all_goalie_stats(conn, result["goalies"])

            # One PBP fetch feeds both tables: shot_events (shot types only)
            # and game_events (every event — needed for rebound/rush context)
            pbp = call(fetch_pbp_json, game_id)
            shots = parse_shot_events(pbp, game_id)
            store_shot_events(conn, shots)
            events = parse_game_events(pbp, game_id)
            store_game_events(conn, events)

            try:
                shifts = call(fetch_shifts, game_id)
                store_shifts(conn, shifts)
                shift_count = len(shifts)
            except Exception:
                shift_count = 0

            print(
                f"  [{i+1}/{len(new_ids)}] "
                f"{result['game'].away_team_abbrev} @ {result['game'].home_team_abbrev} - "
                f"{result['game'].game_date} ({len(shots)} shots, {len(events)} events, {shift_count} shifts)"
            )
        except Exception as e:
            log_failure(f"[{i+1}/{len(new_ids)}] Game {game_id} FAILED: {e}")

    # 7. Derive shot context (rebound / rush flags) for any new games.
    # Processing step, kept out of the per-game acquisition loop on purpose.
    print("Deriving shot context (rebound / rush flags)...")
    n = build_shot_context(conn)
    print(f"Wrote {n} shot_context rows.")

    conn.close()
    print(f"Done! Loaded {len(new_ids):,} new games; skipped {len(game_ids) - len(new_ids):,} already loaded.")


if __name__ == "__main__":
    main()