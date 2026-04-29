# scripts/load_games.py
import sys
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_boxscore
from src.data.db import (
    get_connection, create_tables, get_finished_game_ids,
    store_game, store_all_goalie_stats, store_all_skater_stats
)
'''
This script loads game data into the database. 
It fetches the schedule for a specific team (COL in this case),
retrieves the boxscore for each finished game, and stores the game
info along with skater and goalie stats in the database.
'''
conn = get_connection()
create_tables(conn)

game_ids = get_finished_game_ids(conn, "COL")
print(f"Found {len(game_ids)} finished COL games")

for i, game_id in enumerate(game_ids):
    result = fetch_boxscore(game_id)
    store_game(conn, result["game"])
    store_all_skater_stats(conn, result["skaters"])
    store_all_goalie_stats(conn, result["goalies"])
    print(f"  [{i+1}/{len(game_ids)}] {result['game'].away_team_abbrev} @ {result['game'].home_team_abbrev} - {result['game'].game_date}")

conn.close()
print("Done!")