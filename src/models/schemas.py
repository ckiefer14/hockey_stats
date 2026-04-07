# src/models/schema.py
from pydantic import BaseModel, Field
from datetime import date

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

    