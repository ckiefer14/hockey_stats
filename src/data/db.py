# src/data/db.py
import sqlite3
from config.settings import DB_PATH
from src.models.schemas import Team

## Database functions for SELite. Uses SQLite for simplicity, but could be swapped out for something more robust if needed.
def get_connection():
    """Get a connection to the SELite database. Creates the file if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


## Creats a teams table with a column for each field in Pydantic Team model
def create_tables(conn: sqlite3.Connection):
    """Create all tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            abbreviation TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            common_name TEXT,
            logo_url TEXT,
            place_name TEXT,
            division TEXT,
            division_abbrev TEXT,
            conference TEXT,
            conference_abbrev TEXT,
            primary_color TEXT,
            secondary_color TEXT
        )
    """)
    conn.commit()

## Team functions

## Insert or update a single team in the database
def store_team(conn: sqlite3.Connection, team: Team):
    """Insert or update a single team."""
    conn.execute("""
        INSERT OR REPLACE INTO teams
        (abbreviation, name, common_name, logo_url, place_name,
         division, division_abbrev, conference, conference_abbrev,
         primary_color, secondary_color)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,(
        team.abbreviation,
        team.name,
        team.common_name,
        team.logo_url,
        team.place_name,
        team.division,
        team.division_abbrev,
        team.conference,
        team.conference_abbrev,
        team.colors.primary,
        team.colors.secondary
    ))
    conn.commit()

## Store a list of teams in the database
def store_teams(conn: sqlite3.Connection, teams: list[Team]):
    """Insert or update a list of temas."""
    for team in teams:
        store_team(conn, team)

## Get a single team by abbreviation. Returns None if not found.
def get_team(conn: sqlite3.Connection, abbreviation: str) -> Team | None:
    """Get a single team by abbreviation. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM teams WHERE abbreviation = ?",
        (abbreviation,)
    ).fetchone()

    if row is None:
        return None
    
    return Team(
        abbreviation=row[0],
        name=row[1],
        common_name=row[2],
        logo_url=row[3],
        place_name=row[4],
        division=row[5],
        division_abbrev=row[6],
        conference=row[7],
        conference_abbrev=row[8],
        colors={"primary": row[9], "secondary": row[10]},
    )

## Get all teams from the database. Returns an empty list if no teams found.
def get_all_teams(conn:sqlite3.Connection) -> list[Team]:
    """GEt all teams from a database."""
    rows = conn.execute("SELECT * FROM teams").fetchall()

    return [
        Team(
            abbreviation=row[0],
            name=row[1],
            common_name=row[2],
            logo_url=row[3],
            place_name=row[4],
            division=row[5],
            division_abbrev=row[6],
            conference=row[7],
            conference_abbrev=row[8],
            colors={"primary": row[9], "secondary": row[10]},
        )
        for row in rows
    ]
