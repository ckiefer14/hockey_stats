# scripts/test_corsi.py
import sys
sys.path.insert(0, ".")

from src.processing.corsi import calculate_game_corsi, calculate_season_corsi
from src.data.db import get_connection, get_finished_game_ids

conn = get_connection()
game_ids = get_finished_game_ids(conn, "COL")
conn.close()

print("Game ID      CF   CA   CF%    Diff")
print("-" * 40)

for game_id in game_ids[:10]:
    result = calculate_game_corsi(game_id, "COL")
    win_marker = "W" if result["won"] else "L"
    print(f"{result['game_id']}   {result['cf']:3d}  {result['ca']:3d}  {result['cf_pct']:5.1f}%  {result['cf_diff']:+d}  {result['score']} {win_marker}")


result_season = calculate_season_corsi("COL")
print("\nSeason Totals:")
print(f"CF: {result_season['total_cf']}  CA: {result_season['total_ca']}  CF%: {result_season['cf_pct']:.1f}%  Diff: {result_season['cf_diff']:+d}")
