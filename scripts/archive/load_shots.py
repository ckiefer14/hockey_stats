# scripts/load_shots.py
import sys
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_play_by_play
from src.data.db import get_connection, create_tables, get_finished_game_ids, store_shot_events

conn = get_connection()
create_tables(conn)

game_ids = get_finished_game_ids(conn, "COL")
print(f"Found {len(game_ids)} finished COL games")

for i, game_id in enumerate(game_ids):
    shots = fetch_play_by_play(game_id)
    store_shot_events(conn, shots)
    print(f"  [{i+1}/{len(game_ids)}] Game {game_id}: {len(shots)} shots")

conn.close()
print("Done!")