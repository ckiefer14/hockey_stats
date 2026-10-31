import sys
sys.path.insert(0, ".")

from src.data.nhl_api import fetch_shifts

shifts = fetch_shifts(2025020034) # DAL @ COL

print(f"Total shifts: {len(shifts)}\n")

burns_shifts = [s for s in shifts if s.last_name == "Burns"]
print(f"Burns had {len(burns_shifts)} shifts\n")

for s in burns_shifts[:5]:
    print(f" P{s.period} {s.shift_number}: {s.start_time} - {s.end_time} ({s.duration})")