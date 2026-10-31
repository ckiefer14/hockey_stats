import sys
sys.path.insert(0, ".")

from src.data.db import get_connection
from src.processing.shot_context import build_shot_context
conn = get_connection()
build_shot_context(conn, rebuild=True)

# from src.data.db import get_connection
# conn = get_connection()

# tables = [
#     "teams", "players", "schedule", "games",
#     "skater_game_stats", "goalie_game_stats", "shot_events", "shifts"
# ]

# print(f"{'Table':<25} {'Rows':>10}")
# print("-" * 37)
# for table in tables:
#     count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
#     print(f"{table:<25} {count:>10,}")

# # Some derived sanity checks
# print("\n--- Sanity checks ---")

# # Games per season
# print("\nGames per season:")
# for row in conn.execute(
#     "SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season"
# ).fetchall():
#     print(f"  {row[0]}: {row[1]:,} games")

# # Schedule entries per season
# print("\nSchedule entries per season:")
# for row in conn.execute(
#     "SELECT season, COUNT(*) FROM schedule GROUP BY season ORDER BY season"
# ).fetchall():
#     print(f"  {row[0]}: {row[1]:,} entries")

# # Shot events per season (via join to games)
# print("\nShot events per season:")
# for row in conn.execute("""
#     SELECT g.season, COUNT(*) 
#     FROM shot_events s
#     JOIN games g ON s.game_id = g.id
#     GROUP BY g.season
#     ORDER BY g.season
# """).fetchall():
#     print(f"  {row[0]}: {row[1]:,} shots")

# # Games with no shots (incomplete pulls)
# no_shots = conn.execute("""
#     SELECT COUNT(*) FROM games
#     WHERE id NOT IN (SELECT DISTINCT game_id FROM shot_events)
# """).fetchone()[0]
# print(f"\nGames with zero shot events: {no_shots}")

# # Games with no shifts (best-effort, expected to be > 0)
# no_shifts = conn.execute("""
#     SELECT COUNT(*) FROM games
#     WHERE id NOT IN (SELECT DISTINCT game_id FROM shifts)
# """).fetchone()[0]
# print(f"Games with zero shift records: {no_shifts}")

# # Number of teams (should be 32)
# n_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
# print(f"\nTeams in database: {n_teams}")

# conn.close()
# from src.data.db import get_connection
# conn = get_connection()

# # What tables exist?
# print("Tables in the database:")
# tables = conn.execute(
#     "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
# ).fetchall()
# for (name,) in tables:
#     print(f"  {name}")

# # What columns does each one have?
# for (name,) in tables:
#     print(f"\n{name}:")
#     for cid, col_name, col_type, notnull, default, pk in conn.execute(
#         f"PRAGMA table_info({name})"
#     ).fetchall():
#         marker = " (PK)" if pk else ""
#         print(f"  {col_name:30s} {col_type}{marker}")

# conn.close()