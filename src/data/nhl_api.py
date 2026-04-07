# src/data/nhl_api.py
import requests
from config.settings import NHL_API_BASE_URL
from config.teams import get_team_colors
from src.models.schemas import Team

## Fetches team info from NHL API and returns a list of Team objects
def fetch_teams() -> list[Team]:
    """Fetch all team info from NHL standings endpoint."""
    response = requests.get(f"{NHL_API_BASE_URL}/v1/standings/now")
    response.raise_for_status()

    standings = response.json()["standings"]

    teams = []
    for entry in standings:
        teams.append({
            "abbreviation": entry["teamAbbrev"]["default"],
            "name": entry["teamName"]["default"],
            "common_name": entry["teamCommonName"]["default"],
            "logo_url": entry["teamLogo"],
            "place_name": entry["placeName"]["default"],
            "division": entry["divisionName"],
            "division_abbrev": entry["divisionAbbrev"],
            "conference": entry["conferenceName"],
            "conference_abbrev": entry["conferenceAbbrev"],
            "colors": get_team_colors(entry["teamAbbrev"]["default"]),
        })

    return teams