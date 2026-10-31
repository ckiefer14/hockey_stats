# scripts/test_fenwick.py
import sys
sys.path.insert(0, ".")

from src.processing.fenwick import calculate_game_fenwick, calculate_season_fenwick
from src.data.db import get_connection, get_finished_game_ids

conn = get_connection()
game_ids = get_finished_game_ids(conn, "COL")
conn.close()

print("Game ID      FF   FA   FF%    Diff")
print("-" * 40)

for game_id in game_ids[:10]:
    result = calculate_game_fenwick(game_id, "COL")
    win_marker = "W" if result["won"] else "L"
    print(f"{result['game_id']}   {result['ff']:3d}  {result['fa']:3d}  {result['ff_pct']:5.1f}%  {result['ff_diff']:+d}  {result['score']} {win_marker}")


result_season = calculate_season_fenwick("COL")
print("\nSeason Totals:")
print(f"CF: {result_season['total_ff']}  FA: {result_season['total_fa']}  FF%: {result_season['ff_pct']:.1f}%  Diff: {result_season['ff_diff']:+d}")
