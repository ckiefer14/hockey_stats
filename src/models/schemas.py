# src/models/schema.py
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional
'''
This module defines the data models used in the application, using Pydantic for validation and parsing.
The models include Team, Player, Game, SkaterGameStats, and GoalieGameStats, which represent the core entities in our hockey stats application.
'''
class TeamColors(BaseModel):
    primary: str
    secondary: str

class Team(BaseModel):
    abbreviation: str
    name: str
    common_name: str
    logo_url: str
    place_name: str
    division: str
    division_abbrev: str
    conference: str
    conference_abbrev: str
    colors: TeamColors
    team_id: Optional[int] = None


class Player(BaseModel):
    id: int
    first_name : str
    last_name : str
    team_abbreviation : str
    headshot : str
    sweaterNumber : Optional[int] = None
    shoots_catches : Optional[str] = None
    position_code : str
    height_inches : int
    weight_pounds : int
    birth_date : date
    age : int
    birth_city : Optional[str] = None
    birth_country : Optional[str] = None
    birth_state_province : Optional[str] = None

class Game(BaseModel):
    id: int
    season: int
    game_type: int          # 1=preseason, 2=regular season, 3=playoffs
    game_date: str
    venue: str
    home_team_abbrev: str
    away_team_abbrev: str
    home_score: int
    away_score: int
    home_sog: int           # shots on goal
    away_sog: int
    game_state: str         # "OFF" = final, "LIVE" = in progress, "FUT" = scheduled
    period_type: str        # "REG", "OT", "SO"

class SkaterGameStats(BaseModel):
    game_id: int
    player_id: int
    team_abbreviation: str
    name: str
    position: str
    goals: int
    assists: int
    points: int
    plus_minus: int
    pim: int                    # penalties in minutes
    hits: int
    power_play_goals: int
    sog: int                    # shots on goal
    faceoff_winning_pctg: float
    toi: str                    # time on ice, formatted as "MM:SS"
    blocked_shots: int
    shifts: int
    giveaways: int
    takeaways: int

class GoalieGameStats(BaseModel):
    game_id: int
    player_id: int
    team_abbreviation: str
    name: str
    saves: int
    shots_against: int
    goals_against: int
    save_percentage: Optional[float] = None
    toi: str
    starter: bool
    decision: Optional[str] = None
    even_strength_goals_against: int
    power_play_goals_against: int
    shorthanded_goals_against: int

class ShotEvent(BaseModel):
    game_id: int
    event_id: int
    period: int
    period_type: str
    time_in_period: str
    event_type: str                     # "shot-on-goal", "goal", "missed-shot"
    team_id: int
    shooting_player_id: Optional[int] = None
    goalie_id: Optional[int] = None
    x_coord: float                      # rink coordinates: center ice is (0, 0)
    y_coord: float
    zone_code: str                      # "O"=offensive, "D"=defensive, "N"=neutral
    shot_type: Optional[str] = None     # e.g. "wrist", "snap", "slap", "backhand"
    reason: Optional[str] = None        # for missed shots: "wide-right", "high-and-wide", etc.
    blocking_player_id: Optional[int] = None # for blocked shots

class GameEvent(BaseModel):
    """One play-by-play event of ANY type (faceoff, hit, takeaway, shot, ...).

    The full event stream backs the shot-context derivation (rebound/rush):
    rush detection needs the non-shot events immediately preceding a shot.
    Joins to ShotEvent on (game_id, event_id); chronological order comes
    from sort_order, since the API's eventId is NOT in play order.
    """
    game_id: int
    event_id: int                       # API eventId — join key to shot_events
    sort_order: int                     # API sortOrder — chronological ordering
    period: int
    period_type: str                    # "REG", "OT", "SO"
    time_in_period: str                 # "MM:SS", counts up
    game_seconds: float                 # (period-1)*1200 + seconds elapsed in period
    event_type: str                     # typeDescKey: "hit", "faceoff", "shot-on-goal", ...
    team_id: Optional[int] = None       # eventOwnerTeamId; None for stoppages etc.
    zone_code: Optional[str] = None     # O/N/D relative to event owner; None when absent
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    shot_type: Optional[str] = None
    shooting_player_id: Optional[int] = None
    goalie_id: Optional[int] = None
    situation_code: Optional[str] = None  # strength state encoding, e.g. "1551"

class ScheduleGame(BaseModel):
    game_id: int
    season: int
    game_type: int
    game_date: date
    home_team_abbrev: str
    away_team_abbrev: str
    game_state: str
    home_team_id: int
    away_team_id: int

class Shift(BaseModel):
    id: int
    game_id: int
    player_id: int
    team_id: int
    team_abbrev: str
    first_name: str
    last_name: str
    period: int
    shift_number: int
    start_time: str
    end_time: str
    duration: Optional[str] = None