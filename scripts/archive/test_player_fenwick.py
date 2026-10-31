# scripts/test_player_fenwick.py
import sys
sys.path.insert(0, ".")

from src.processing.fenwick import calculate_player_game_fenwick, calculate_player_season_fenwick
from src.data.db import get_connection, get_player_id

conn = get_connection()
makar_id = get_player_id(conn, "Makar", "Cale", "COL")

print("Makar player ID: {makar_id}")
conn.close()

player_game = calculate_player_game_fenwick(makar_id, 2025020034, "COL")
if player_game:
    print(f"Makar:    FF={player_game['ff']} FA={player_game['fa']} FF%={player_game['ff_pct']}")
else:
    print("No shift data for this game")

player_season = calculate_player_season_fenwick(makar_id, "COL")
if player_season:
    print(f"Makar season: FF={player_season['total_ff']} FA={player_season['total_fa']} FF%={player_season['ff_pct']}")