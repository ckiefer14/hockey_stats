# scripts/load_schedule.py
import sys
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_schedule
from src.data.db import get_connection, create_tables, store_schedule

conn = get_connection()
create_tables(conn)

schedule = fetch_schedule("COL")
store_schedule(conn, schedule)

finished = [g for g in schedule if g.game_state == "OFF"]
upcoming = [g for g in schedule if g.game_state != "OFF"]

print(f"Stored {len(schedule)} COL games")
print(f"  {len(finished)} finished")
print(f"  {len(upcoming)} upcoming")

conn.close()