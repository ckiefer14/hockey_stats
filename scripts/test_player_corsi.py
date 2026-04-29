# scripts/test_player_corsi.py
import sys
sys.path.insert(0, ".")

from src.processing.corsi import calculate_player_game_corsi, calculate_player_season_corsi
from src.data.db import get_connection, get_player_id

conn = get_connection()
makar_id = get_player_id(conn, "Makar", "Cale", "COL")

print("Makar player ID: {makar_id}")
conn.close()

player_game = calculate_player_game_corsi(makar_id, 2025020034, "COL")
if player_game:
    print(f"Makar:    CF={player_game['cf']} CA={player_game['ca']} CF%={player_game['cf_pct']}")
else:
    print("No shift data for this game")

player_season = calculate_player_season_corsi(makar_id, "COL")
if player_season:
    print(f"Makar season: CF={player_season['total_cf']} CA={player_season['total_ca']} CF%={player_season['cf_pct']}")