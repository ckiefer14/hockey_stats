# scripts/test_shots.py
import sys
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_play_by_play

shots = fetch_play_by_play(2025020034)  # DAL @ COL

print(f"Total shot events: {len(shots)}\n")

for shot in shots[:5]:
    print(f"  {shot.time_in_period} P{shot.period} {shot.event_type:15s} ({shot.x_coord}, {shot.y_coord}) {shot.shot_type}")