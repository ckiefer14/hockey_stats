# src/data/nhl_api.py
import requests
from config.settings import NHL_API_BASE_URL
from config.teams import get_team_colors
from src.models.schemas import (
    Team, Player, Game, SkaterGameStats, 
    GoalieGameStats, ShotEvent, ScheduleGame, Shift)

from datetime import date

'''
This module contains functions for fetching data from the NHL API,
including team info, rosters, game boxscores, and schedules.
The functions return Pydantic models defined in src/models/schemas.py
for easy validation and parsing.
'''

## Fetches team info from NHL API and returns a list of Team objects
def fetch_teams() -> list[Team]:
    """Fetch all team info from NHL standings endpoint."""
    response = requests.get(f"{NHL_API_BASE_URL}/v1/standings/now")
    response.raise_for_status()

    standings = response.json()["standings"]

    teams = []
    for entry in standings:
        teams.append(Team(
            abbreviation=entry["teamAbbrev"]["default"],
            name=entry["teamName"]["default"],
            common_name=entry["teamCommonName"]["default"],
            logo_url=entry["teamLogo"],
            place_name=entry["placeName"]["default"],
            division=entry["divisionName"],
            division_abbrev=entry["divisionAbbrev"],
            conference=entry["conferenceName"],
            conference_abbrev=entry["conferenceAbbrev"],
            colors=get_team_colors(entry["teamAbbrev"]["default"]),
            team_id=entry.get("teamId"),
        ))

    return teams

## Fetches the current roster for a team and returns a list of Player objects
def fetch_roster(team_abbrev: str) -> list[Player]:
    """Fetch the current roster for a team"""
    response = requests.get(f"{NHL_API_BASE_URL}/v1/roster/{team_abbrev}/current")
    response.raise_for_status()

    data = response.json()
    players = []

    today = date.today()

    for group in ["forwards", "defensemen", "goalies"]:
        for entry in data[group]:
            birth = date.fromisoformat(entry['birthDate'])
            # Subtract 1 if the birthday hasn't occurred yet this calendar year
            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            players.append(Player(
                id=entry["id"],
                first_name=entry["firstName"]["default"],
                last_name=entry["lastName"]["default"],
                team_abbreviation=team_abbrev,
                headshot=entry["headshot"],
                sweaterNumber=entry.get("sweaterNumber"),
                shoots_catches=entry.get("shootsCatches"),
                position_code=entry["positionCode"],
                height_inches=entry["heightInInches"],
                weight_pounds=entry["weightInPounds"],
                birth_date=entry["birthDate"],
                age=age,
                birth_city=entry.get("birthCity", {}).get("default"),
                birth_country=entry.get("birthCountry"),
                birth_state_province=entry.get("birthStateProvince", {}).get("default"),
            ))

    return players

## Fetches boxscore data for a game and returns a dict with Game, SkaterGameStats, and GoalieGameStats
def fetch_boxscore(game_id: int) -> dict:
    """Fetch boxscore for a single game. Returns game, skater stats and goalie stats."""
    response = requests.get(f"{NHL_API_BASE_URL}/v1/gamecenter/{game_id}/boxscore")
    response.raise_for_status()

    data = response.json()

    game = Game(
        id=data["id"],
        season=data["season"],
        game_type=data["gameType"],
        game_date=data["gameDate"],
        venue=data["venue"]["default"],
        home_team_abbrev=data["homeTeam"]["abbrev"],
        away_team_abbrev=data["awayTeam"]["abbrev"],
        home_score=data["homeTeam"]["score"],
        away_score=data["awayTeam"]["score"],
        home_sog=data["homeTeam"]["sog"],
        away_sog=data["awayTeam"]["sog"],
        game_state=data["gameState"],
        period_type=data["periodDescriptor"]["periodType"],
    )

    skaters = []
    goalies = []

    for side in ["awayTeam", "homeTeam"]:
        abbrev = data[side]["abbrev"]
        stats = data["playerByGameStats"][side]

        for group in ["forwards", "defense"]:
            for p in stats[group]:
                skaters.append(SkaterGameStats(
                    game_id=game_id,
                    player_id=p["playerId"],
                    team_abbreviation=abbrev,
                    name=p["name"]["default"],
                    position=p["position"],
                    goals=p["goals"],
                    assists=p["assists"],
                    points=p["points"],
                    plus_minus=p["plusMinus"],
                    pim=p["pim"],
                    hits=p["hits"],
                    power_play_goals=p["powerPlayGoals"],
                    sog=p["sog"],
                    faceoff_winning_pctg=p["faceoffWinningPctg"],
                    toi=p["toi"],
                    blocked_shots=p["blockedShots"],
                    shifts=p["shifts"],
                    giveaways=p["giveaways"],
                    takeaways=p["takeaways"],
                ))

        for g in stats["goalies"]:
            goalies.append(GoalieGameStats(
                game_id=game_id,
                player_id=g["playerId"],
                team_abbreviation=abbrev,
                name=g["name"]["default"],
                saves=g["saves"],
                shots_against=g["shotsAgainst"],
                goals_against=g["goalsAgainst"],
                save_pctg=g.get("savePctg"),
                toi=g["toi"],
                starter=g["starter"],
                decision=g.get("decision"),
                even_strength_goals_against=g["evenStrengthGoalsAgainst"],
                power_play_goals_against=g["powerPlayGoalsAgainst"],
                shorthanded_goals_against=g["shorthandedGoalsAgainst"],
            ))

    return {"game": game, "skaters": skaters, "goalies": goalies}

## Fetches the schedule for a team and returns a list of game IDs for finished regular season games
def fetch_schedule(team_abbrev: str, season: str | None = None) -> list[ScheduleGame]:
    """Fetch all game IDs for a team's current season."""
    if season is None:
        url = f"{NHL_API_BASE_URL}/v1/club-schedule-season/{team_abbrev}/now"
    else:
        url = f"{NHL_API_BASE_URL}/v1/club-schedule-season/{team_abbrev}/{season}"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    games = []

    for game in data["games"]:
        # Only grab regular season games
        if game["gameType"] == 2:
            games.append(ScheduleGame(
            game_id=game["id"],
            season=game["season"],
            game_type=game["gameType"],
            game_date=game["gameDate"],
            home_team_abbrev=game["homeTeam"]["abbrev"],
            away_team_abbrev=game["awayTeam"]["abbrev"],
            game_state=game["gameState"],
            home_team_id=game["homeTeam"]["id"],
            away_team_id=game["awayTeam"]["id"],
        ))
    return games

## Fetches play-by-play data for a game and filters to shot events only
def fetch_play_by_play(game_id: int) -> list[ShotEvent]:
    """Fetch shot events from play-by-play data for a single game."""
    response = requests.get(f"{NHL_API_BASE_URL}/v1/gamecenter/{game_id}/play-by-play")
    response.raise_for_status()

    data = response.json()
    shot_types = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
    shots = []

    for play in data["plays"]:
        if play["typeDescKey"] not in shot_types:
            continue

        details = play.get("details", {})

        # Skip if no coordinates
        if "xCoord" not in details or "yCoord" not in details:
            continue

        shots.append(ShotEvent(
            game_id=game_id,
            event_id=play["eventId"],
            period=play["periodDescriptor"]["number"],
            period_type=play["periodDescriptor"]["periodType"],
            time_in_period=play["timeInPeriod"],
            event_type=play["typeDescKey"],
            team_id=details["eventOwnerTeamId"],
            shooting_player_id=details.get("shootingPlayerId"),
            goalie_id=details.get("goalieInNetId"),
            x_coord=details["xCoord"],
            y_coord=details["yCoord"],
            zone_code=details["zoneCode"],
            shot_type=details.get("shotType"),
            reason=details.get("reason"),
            blocking_player_id=details.get("blockingPlayerId"),
        ))

    return shots

def fetch_shifts(game_id: int) -> list[Shift]:
    """Fetch shift data for a single game."""
    response = requests.get(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
    )
    response.raise_for_status()

    data = response.json()
    shifts = []

    for entry in data["data"]:
        shifts.append(Shift(
            id=entry["id"],
            game_id=entry["gameId"],
            player_id=entry["playerId"],
            team_id=entry["teamId"],
            team_abbrev=entry["teamAbbrev"],
            first_name=entry["firstName"],
            last_name=entry["lastName"],
            period=entry["period"],
            shift_number=entry["shiftNumber"],
            start_time=entry["startTime"],
            end_time=entry["endTime"],
            duration=entry["duration"],
        ))

    return shifts