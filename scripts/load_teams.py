# scripts/load_teams.py
import sys
sys.path.insert(0, ".") # Add the project root to the Python path so we can import from src

from src.data.nhl_api import fetch_teams, fetch_roster
from src.data.db import get_connection, create_tables, store_teams, store_players
'''
This script loads team and player data into the database.
It fetches all teams from the NHL API, stores them in the database,
then fetches and stores the current roster for each team.'''

conn = get_connection()
create_tables(conn)

teams = fetch_teams()
store_teams(conn, teams)
print(f"Stored {len(teams)} teams in the database.")

for team in teams:
    players = fetch_roster(team.abbreviation)
    store_players(conn,players)
    print(f"    {team.abbreviation}: {len(players)} players")

conn.close()
print("Done.")