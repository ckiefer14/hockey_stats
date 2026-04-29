# scripts/load_all.py
import sys
import time
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_teams, fetch_roster, fetch_schedule, fetch_boxscore, fetch_play_by_play, fetch_shifts
from src.data.db import (
    get_connection, create_tables, populate_team_ids,
    store_teams, store_players,
    store_schedule, get_finished_game_ids,
    store_game, store_all_skater_stats, store_all_goalie_stats,
    store_shot_events, store_shifts
)

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
schedule = fetch_schedule("COL")
store_schedule(conn, schedule)
print(f"Stored {len(schedule)} COL schedule entries")

# 4. Populate team IDs
populate_team_ids(conn)
print("Populated team IDs")

# 5. Boxscores, shots, and shifts
game_ids = get_finished_game_ids(conn, "COL")
print(f"Loading {len(game_ids)} finished games...")

for i, game_id in enumerate(game_ids):
    try:
        result = fetch_boxscore(game_id)
        store_game(conn, result["game"])
        store_all_skater_stats(conn, result["skaters"])
        store_all_goalie_stats(conn, result["goalies"])

        shots = fetch_play_by_play(game_id)
        store_shot_events(conn, shots)

        try:
            shifts = fetch_shifts(game_id)
            store_shifts(conn, shifts)
            shift_count = len(shifts)
        except Exception:
            shift_count = 0

        print(f"  [{i+1}/{len(game_ids)}] {result['game'].away_team_abbrev} @ {result['game'].home_team_abbrev} - {result['game'].game_date} ({len(shots)} shots, {shift_count} shifts)")

        # Pause to avoid rate limiting
        time.sleep(1)

    except Exception as e:
        print(f"  [{i+1}/{len(game_ids)}] Game {game_id} FAILED: {e}")
        time.sleep(3)

conn.close()
print("Done!")