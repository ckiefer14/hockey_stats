# src/data/db.py
import sqlite3
from config.settings import DB_PATH
from src.models.schemas import (Team, Player, Game, SkaterGameStats, 
                                GoalieGameStats, ShotEvent, ScheduleGame, Shift)

'''
This module contains functions for interacting with the SQLite database,
including creating tables and inserting/updating data for teams, players, games, and stats.'''


## Database functions for SQLite. Uses SQLite for simplicity, but could be swapped out for something more robust if needed.
def get_connection():
    """Get a connection to the SQLite database. Creates the file if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def migrate_tables(conn: sqlite3.Connection):
    """Add new columns to existing tables if they don't exist."""
    try:
        conn.execute("ALTER TABLE shot_events ADD COLUMN blocking_player_id INTEGER")
        conn.commit()
        print("Added blocking_player_id column")
    except sqlite3.OperationalError:
        pass  # Column already exists


## Creates a teams table with a column for each field in Pydantic Team model
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
        secondary_color TEXT,
        team_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            team_abbreviation TEST,
            headshot TEXT,
            sweaterNumber INTEGER,
            shoots_catches TEXT,
            position_code TEXT,
                height_inches INTEGER,
            weight_pounds INTEGER,
            birth_date TEXT,
            age INTEGER,
            birth_city TEXT,
            birth_country TEXT,
            birth_state_province TEXT,
            FOREIGN KEY (team_abbreviation) REFERENCES teams(abbreviation)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY,
            season INTEGER,
            game_type INTEGER,
            game_date TEXT,
            venue TEXT,
            home_team_abbrev TEXT,
            away_team_abbrev TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_sog INTEGER,
            away_sog INTEGER,
            game_state TEXT,
            period_type TEXT,
            FOREIGN KEY (home_team_abbrev) REFERENCES teams(abbreviation),
            FOREIGN KEY (away_team_abbrev) REFERENCES teams(abbreviation)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skater_game_stats (
            game_id INTEGER,
            player_id INTEGER,
            team_abbreviation TEXT,
            name TEXT,
            position TEXT,
            goals INTEGER,
            assists INTEGER,
            points INTEGER,
            plus_minus INTEGER,
            pim INTEGER,
            hits INTEGER,
            power_play_goals INTEGER,
            sog INTEGER,
            faceoff_win_percentage INTEGER,
            toi TEXT,
            blocked_shots INTEGER,
            shifts INTEGER,
            giveways INTEGER,
            takeaways INTEGER,
            PRIMARY KEY (game_id, player_id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goalie_game_stats (
            game_id INTEGER,
            player_id INTEGER,
            team_abbreviation TEXT,
            name TEXT,
            saves INTEGER,
            shots_against INTEGER,
            goals_against INTEGER,
            save_percentage REAL,
            toi TEXT,
            starter INTEGER,
            decision TEXT,
            even_strength_goals_against INTEGER,
            power_play_goals_against INTEGER,
            shorthanded_goals_against INTEGER,
            PRIMARY KEY (game_id, player_id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shot_events (
            game_id INTEGER,
            event_id INTEGER,
            period INTEGER,
            period_type TEXT,
            time_in_period TEXT,
            event_type TEXT,
            team_id INTEGER,
            shooting_player_id INTEGER,
            goalie_id INTEGER,
            x_coord REAL,
            y_coord REAL,
            zone_code TEXT,
            shot_type TEXT,
            reason TEXT,
            blocking_player_id INTEGER,
            PRIMARY KEY (game_id, event_id),
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schedule (
            game_id INTEGER PRIMARY KEY,
            season INTEGER,
            game_type INTEGER,
            game_date TEXT,
            home_team_abbrev TEXT,
            away_team_abbrev TEXT,
            game_state TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            FOREIGN KEY (home_team_abbrev) REFERENCES teams(abbreviations),
            FOREIGN KEY (away_team_abbrev) REFERENCES teams(abbreviation)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shifts (
            id INTEGER PRIMARY KEY,
            game_id INTEGER,
            player_id INTEGER,
            team_id INTEGER,
            team_abbrev TEXT,
            first_name TEXT,
            last_name TEXT,
            period INTEGER,
            shift_number INTEGER,
            start_time TEXT,
            end_time TEXT,
            duration TEXT,
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    """)
    conn.commit()

####################
##### Team functions
####################
## Insert or update a single team in the database
def store_team(conn: sqlite3.Connection, team: Team):
    """Insert or update a single team."""
    conn.execute("""
        INSERT OR REPLACE INTO teams
        (abbreviation, name, common_name, logo_url, place_name,
         division, division_abbrev, conference, conference_abbrev,
         primary_color, secondary_color, team_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
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
        team.colors.secondary,
        team.team_id,
    ))
    conn.commit()   

## Store a list of teams in the database
def store_teams(conn: sqlite3.Connection, teams: list[Team]):
    """Insert or update a list of teams."""
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
    """Get all teams from a database."""
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

####################
##### Player functions
####################

## Store a single player in the database
def store_player(conn: sqlite3.Connection, player: Player):
    """Insert or update a single player."""
    conn.execute("""
        INSERT OR REPLACE INTO players
        (id, first_name, last_name, team_abbreviation, headshot, sweaterNumber,
        shoots_catches, position_code, height_inches, weight_pounds,
        birth_date, age, birth_city, birth_country, birth_state_province)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        player.id,
        player.first_name,
        player.last_name,
        player.team_abbreviation,
        player.headshot,
        player.sweaterNumber,
        player.shoots_catches,
        player.position_code,
        player.height_inches,
        player.weight_pounds,
        str(player.birth_date),
        player.age,
        player.birth_city,
        player.birth_country,
        player.birth_state_province
    ))
    conn.commit()

## Store a list of players in the database
def store_players(conn: sqlite3.Connection, players: list[Player]):
    """Insert or update a list of players."""
    for player in players:
        store_player(conn, player)

## Get all players for a team. Returns an empty list if no players found.
def get_roster(conn:sqlite3.Connection, team_abbrev: str) -> list[Player]:
    """Get all players for a team."""
    rows = conn.execute(
        "SELECT * FROM players WHERE team_abbreviation = ?",
        (team_abbrev,)
        ).fetchall()
    
    return [
        Player(
            id=row[0],
            first_name=row[1],
            last_name=row[2],
            team_abbreviation=row[3],
            headshot=row[4],
            sweaterNumber=row[5],
            shoots_catches=row[6],
            position_code=row[7],
            height_inches=row[8],
            weight_pounds=row[9],
            birth_date=row[10],
            age=row[11],
            birth_city=row[12],
            birth_country=row[13],
            birth_state_province=row[14],
        )
        for row in rows
    ]

####################
##### Game functions
####################

## Store a single game in the database
def store_game(conn: sqlite3.Connection, game: Game):
    """Insert or update a single game."""
    conn.execute("""
        INSERT OR REPLACE INTO games
        (id,season, game_type, game_date, venue, home_team_abbrev,
        away_team_abbrev, home_score, away_score, home_sog, away_sog,
        game_state, period_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        game.id,
        game.season,
        game.game_type,
        str(game.game_date),
        game.venue,
        game.home_team_abbrev,
        game.away_team_abbrev,
        game.home_score,
        game.away_score,
        game.home_sog,
        game.away_sog,
        game.game_state,
        game.period_type,
    ))
    conn.commit()


####################
##### Skater Game Stats functions
####################
def get_player_id(conn: sqlite3.Connection, last_name: str, first_name: str, team_abbrev: str = None) -> int | None:
    """Look up a player's ID by last name. Optionally filter by team."""
    if team_abbrev:
        row = conn.execute("""
            SELECT id, first_name, last_name FROM players
            WHERE last_name LIKE ? AND first_name LIKE ? AND team_abbreviation = ?
        """, (f"%{last_name}%",f"%{first_name}%", team_abbrev)).fetchone()
    else:
        row = conn.execute("""
            SELECT id, first_name, last_name FROM players
            WHERE last_name LIKE ? AND first_name LIKE ?
        """, (f"%{last_name}%",f"%{first_name}%")).fetchone()

    if row is None:
        return None
    return row[0]


## Store a single skater's game stats in the database
def store_skater_stats(conn: sqlite3.Connection, stats: SkaterGameStats):
    """Insert or update a single skater's game stats."""
    conn.execute("""
        INSERT OR REPLACE INTO skater_game_stats
           (game_id, player_id, team_abbreviation, name, position, goals, 
            assists, points, plus_minus, pim, hits, power_play_goals,
            sog, faceoff_win_percentage, toi, blocked_shots, shifts, giveways,
            takeaways)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, ( 
        stats.game_id,
        stats.player_id,
        stats.team_abbreviation,
        stats.name,
        stats.position,
        stats.goals,
        stats.assists,
        stats.points,
        stats.plus_minus,
        stats.pim,
        stats.hits,
        stats.power_play_goals,
        stats.sog,
        stats.faceoff_winning_pctg,
        stats.toi,
        stats.blocked_shots,
        stats.shifts,
        stats.giveaways,
        stats.takeaways,
    ))
    conn.commit() 

## Store a list of skaters' game stats in the database
def store_all_skater_stats(conn: sqlite3.Connection, stats_list: list[SkaterGameStats]):
    """Insert or update a list of skater game stats."""
    for stats in stats_list:
        store_skater_stats(conn,stats)

####################
##### Goalie Game Stats functions
####################

## Store a single goalie's game stats in the database
def store_goalie_stats(conn: sqlite3.Connection, stats: GoalieGameStats):
    """Insert or update a single goalie's game stats."""
    conn.execute("""
        INSERT OR REPLACE INTO goalie_game_stats
            (game_id, player_id, team_abbreviation, name,
            saves, shots_against, goals_against, save_percentage,
            toi, starter, decision, even_strength_goals_against,
            power_play_goals_against, shorthanded_goals_against)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        stats.game_id,
        stats.player_id,
        stats.team_abbreviation,
        stats.name,
        stats.saves,
        stats.shots_against,
        stats.goals_against,
        stats.save_percentage,
        stats.toi,
        stats.starter,
        stats.decision,
        stats.even_strength_goals_against,
        stats.power_play_goals_against,
        stats.shorthanded_goals_against,
    ))
    conn.commit()

## Store a list of goalies' game stats in the database
def store_all_goalie_stats(conn: sqlite3.Connection, stats_list: list[GoalieGameStats]):
    """Insert or update a list of goalie game stats."""
    for stats in stats_list:
        store_goalie_stats(conn, stats)


####################
##### Shot Event functions
####################

def store_shot_event(conn: sqlite3.Connection, shot: ShotEvent):
    """Insert or update a single shot event."""
    conn.execute("""
        INSERT OR REPLACE INTO shot_events
        (game_id, event_id, period, period_type, time_in_period,
        event_type, team_id, shooting_player_id, goalie_id,
        x_coord, y_coord, zone_code, shot_type, reason, blocking_player_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        shot.game_id,
        shot.event_id,
        shot.period,
        shot.period_type,
        shot.time_in_period,
        shot.event_type,
        shot.team_id,
        shot.shooting_player_id,
        shot.goalie_id,
        shot.x_coord,
        shot.y_coord,
        shot.zone_code,
        shot.shot_type,
        shot.reason,
        shot.blocking_player_id
    ))
    conn.commit()

def store_shot_events(conn: sqlite3.Connection, shots: list[ShotEvent]):
    """Insert or update a list of shot events"""
    for shot in shots:
        store_shot_event(conn, shot)

####################
##### Schedule functions
####################

def store_schedule_game(conn: sqlite3.Connection, game: ScheduleGame):
    """Insert or update a single schedule entry."""
    conn.execute("""
        INSERT OR REPLACE INTO schedule
        (game_id, season, game_type, game_date,
         home_team_abbrev, away_team_abbrev, game_state, home_team_id, away_team_id)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        game.game_id,
        game.season,
        game.game_type,
        str(game.game_date),
        game.home_team_abbrev,
        game.away_team_abbrev,
        game.game_state,
        game.home_team_id,
        game.away_team_id
    ))
    conn.commit()


def store_schedule(conn: sqlite3.Connection, games: list[ScheduleGame]):
    """Insert or update a list of schedule entries."""
    for game in games:
        store_schedule_game(conn, game)

def get_finished_game_ids(conn: sqlite3.Connection, team_abbrev: str) -> list[int]:
    """Get all finished game IDs for a team from the schedule."""
    rows = conn.execute("""
        SELECT game_id FROM schedule
        WHERE game_state = 'OFF'
        AND (home_team_abbrev = ? OR away_team_abbrev = ?)
        ORDER BY game_date
    """, (team_abbrev, team_abbrev)).fetchall()

    return [row[0] for row in rows]

def get_team_id(conn: sqlite3.Connection, team_abbrev: str) -> int | None:
    """Look up a team's numeric ID from the schedule data."""
    row = conn.execute("""
        SELECT home_team_id FROM schedule WHERE home_team_abbrev = ?
        UNION
        SELECT away_team_id FROM schedule WHERE away_team_abbrev = ?
        LIMIT 1
    """, (team_abbrev, team_abbrev)).fetchone()

    if row is None:
        return None
    return row[0]

def populate_team_ids(conn: sqlite3.Connection):
    """Fill in team_id on the teams table using schedule data."""
    conn.execute("""
        UPDATE teams SET team_id = (
            SELECT home_team_id FROM schedule
            WHERE schedule.home_team_abbrev = teams.abbreviation
            LIMIT 1
        )
        WHERE team_id IS NULL
    """)
    conn.commit()

####################
##### Shift functions
####################

def store_shift(conn: sqlite3.Connection, shift: Shift):
    """Insert or update a single shift."""
    conn.execute("""
        INSERT OR REPLACE INTO shifts
        (id, game_id, player_id, team_id, team_abbrev,
         first_name, last_name, period, shift_number,
         start_time, end_time, duration)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        shift.id,
        shift.game_id,
        shift.player_id,
        shift.team_id,
        shift.team_abbrev,
        shift.first_name,
        shift.last_name,
        shift.period,
        shift.shift_number,
        shift.start_time,
        shift.end_time,
        shift.duration,
    ))
    conn.commit()

def store_shifts(conn: sqlite3.Connection, shifts: list[Shift]):
    """Insert or update a list of shifts."""
    for shift in shifts:
        store_shift(conn, shift)

