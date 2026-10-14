# scripts/load_all.py
import sys
import time
sys.path.insert(0, ".")

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

SEASONS = ["20212022","20222023","20232024","20242025","20252026"]
# SEASONS_2026 = ["20252026"]
FAILURE_LOG = "data_load_failures.log"
SLEEP_BETWEEN_CALLS = 1.0
SLEEP_AFTER_FAILURE = 3.0

def log_failure(msg: str) -> None:
    print(msg)
    with open(FAILURE_LOG, "a") as f:
        f.write(f"{msg}\n")

def main():

    conn = get_connection()
    create_tables(conn)

    # 1. Teams
    teams = fetch_teams()
    store_teams(conn, teams)
    print(f"Stored {len(teams)} teams")

    # 2. Rosters
    for team in teams:
        players = fetch_roster(team.abbreviation)
        store_players(conn, players)
        print(f"  {team.abbreviation}: {len(players)} players")

    # 3. Schedule
    for season in SEASONS:
        for team in teams:
            try:
                schedule = fetch_schedule(team.abbreviation, season=season)
                store_schedule(conn,schedule)
                print(f"Stored {len(schedule)} {team.abbreviation} schedule entries")
            except Exception as e:
                log_failure(f"Schedule fetch failed for {team.abbreviation} {season}: {e}")
            
    # 4. Populate team IDs
    populate_team_ids(conn)
    print("Populated team IDs")

    all_game_ids = set()

    # 5. Collect unique game IDs across all teams (deduplicated)
    for team in teams:
        team_game_ids = get_finished_game_ids(conn, team.abbreviation)
        all_game_ids.update(team_game_ids)
    game_ids = sorted(all_game_ids)
    print(f"Total unique finished games to process: {len(game_ids)}")

    # 6. Per-game pull with resume + failure logging
    skipped = 0
    for i, game_id in enumerate(game_ids):
        # Resume logic - skip if already loaded
        if game_already_loaded(conn, game_id):
            skipped += 1
            continue


        try:
            result = fetch_boxscore(game_id)
            store_game(conn, result["game"])
            store_all_skater_stats(conn, result["skaters"])
            store_all_goalie_stats(conn, result["goalies"])

            # One PBP fetch feeds both tables: shot_events (shot types only)
            # and game_events (every event — needed for rebound/rush context)
            pbp = fetch_pbp_json(game_id)
            shots = parse_shot_events(pbp, game_id)
            store_shot_events(conn, shots)
            events = parse_game_events(pbp, game_id)
            store_game_events(conn, events)

            try:
                shifts = fetch_shifts(game_id)
                store_shifts(conn, shifts)
                shift_count = len(shifts)
            except Exception:
                shift_count = 0

            print(
                f"  [{i+1}/{len(game_ids)}] "
                f"{result['game'].away_team_abbrev} @ {result['game'].home_team_abbrev} - "
                f"{result['game'].game_date} ({len(shots)} shots, {len(events)} events, {shift_count} shifts)"
            )

            # Pause to avoid rate limiting
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            log_failure(f"[{i+1}/{len(game_ids)}] Game {game_id} FAILED: {e}")
            time.sleep(SLEEP_AFTER_FAILURE)

    # 7. Derive shot context (rebound / rush flags) for any new games.
    # Processing step, kept out of the per-game acquisition loop on purpose.
    print("Deriving shot context (rebound / rush flags)...")
    n = build_shot_context(conn)
    print(f"Wrote {n} shot_context rows.")

    conn.close()
    print(f"Done! Skipped {skipped} already-loaded games.")


if __name__ == "__main__":
    main()