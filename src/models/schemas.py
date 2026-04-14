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
    game_type: int
    game_date: str
    venue: str
    home_team_abbrev: str
    away_team_abbrev: str
    home_score: int
    away_score: int
    home_sog: int
    away_sog: int
    game_state: str
    period_type: str # "REG","OT", "SO"

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
    pim: int
    hits: int
    power_play_goals: int
    sog: int
    faceoff_winning_pctg: float
    toi: str
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
    event_type: str          # "shot-on-goal", "goal", "missed-shot"
    team_id: int
    shooting_player_id: Optional[int] = None
    goalie_id: Optional[int] = None
    x_coord: float
    y_coord: float
    zone_code: str
    shot_type: Optional[str] = None
    reason: Optional[str] = None    # for missed shots: "wide-right", "high-and-wide", etc.
