# scripts/test_heatmap.py
import sys
sys.path.insert(0, ".")

from src.data.db import get_connection
from src.models.schemas import ShotEvent
from src.viz.shot_heatmap import plot_game_shot_heatmap

conn = get_connection()

# DAL @ COL game
game_id = 2025020034

# Get shots from database
rows = conn.execute(
    "SELECT * FROM shot_events WHERE game_id = ?",
    (game_id,)
).fetchall()

# Build ShotEvent objects
shots = []
for row in rows:
    shots.append(ShotEvent(
        game_id=row[0],
        event_id=row[1],
        period=row[2],
        period_type=row[3],
        time_in_period=row[4],
        event_type=row[5],
        team_id=row[6],
        shooting_player_id=row[7],
        goalie_id=row[8],
        x_coord=row[9],
        y_coord=row[10],
        zone_code=row[11],
        shot_type=row[12],
        reason=row[13],
    ))

# Split by team (COL home team id = 21, DAL away team id = 25)
home_shots = [s for s in shots if s.team_id == 21]
away_shots = [s for s in shots if s.team_id == 25]

away_abbrev = "DAL"
home_abbrev = "COL"
game_date = "2025-10-11"

print(f"Home (COL): {len(home_shots)} shots")
print(f"Away (DAL): {len(away_shots)} shots")

for event_type in ["shot-on-goal", "goal", "missed-shot"]:
    home_count = len([s for s in home_shots if s.event_type == event_type])
    away_count = len([s for s in away_shots if s.event_type == event_type])
    print(f"  {event_type:15s}  COL: {home_count}  DAL: {away_count}")

plot_game_shot_heatmap(
    home_shots, away_shots,
    home_abbrev, away_abbrev, game_date,
    save_path=f"output/{away_abbrev}_at_{home_abbrev}_{game_date}_heatmap.png"
)

conn.close()