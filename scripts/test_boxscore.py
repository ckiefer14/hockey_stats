# scripts/test_boxscore.py
import sys
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_boxscore

result = fetch_boxscore(2025020034)  # DAL @ COL game

game = result["game"]
print(f"Game: {game.away_team_abbrev} @ {game.home_team_abbrev}")
print(f"Score: {game.away_score} - {game.home_score} ({game.period_type})")
print(f"Venue: {game.venue}")

print(f"\nSkaters: {len(result['skaters'])}")
for s in result["skaters"][:3]:
    print(f"  {s.name:20s} {s.team_abbreviation} {s.goals}G {s.assists}A {s.toi}")

print(f"\nGoalies: {len(result['goalies'])}")
for g in result["goalies"]:
    print(f"  {g.name:20s} {g.team_abbreviation} {g.saves}SV/{g.shots_against}SA {g.toi}")