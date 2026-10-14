# scripts/check_block_zone_frame.py
"""
DIAGNOSTIC: is `zone_code` on a blocked-shot recorded in the EVENT OWNER's frame
or the DEFENDING team's frame?

WHY IT MATTERS: check_block_attribution.py established the owner is the SHOOTING
team (90/10). But 96% of blocked shots carry zone_code 'D'. A shooter's blocked
attempt physically happens in their OFFENSIVE zone, so those two facts are
inconsistent unless the zone is written from the defending team's perspective.

If it is, derive_shot_context mis-frames every block in the rush rule: the
shooter's follow-up sees prev_zone='D' with no team flip and gets flagged as a
rush, when it is really a rebound off a block. That would explain why blocked
shots are ~26% of attempts but 50.5% of rush precursors.

METHOD (no zone codes involved, so it cannot beg the question): for each
game/period/team, compare the mean x_coord of that team's SHOTS ON GOAL — which
are unambiguously in their offensive zone — against the mean x_coord of that
team's BLOCKED shots. If both cluster on the same end of the ice, the blocked
attempt happened in that team's offensive zone, and the 'D' label must therefore
be in the defending frame.

Run from the project root:  python scripts/check_block_zone_frame.py
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data.db import get_connection

MIN_EVENTS = 3   # per team/period, for a stable mean


def main():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT game_id, period, team_id, event_type, x_coord
        FROM game_events
        WHERE event_type IN ('shot-on-goal', 'blocked-shot')
          AND x_coord IS NOT NULL
          AND team_id IS NOT NULL
          AND period <= 4
    """, conn)
    conn.close()
    print(f"Loaded {len(df):,} located shot/block events.")

    grp = (df.groupby(["game_id", "period", "team_id", "event_type"])["x_coord"]
             .agg(["mean", "size"]).reset_index())
    wide = grp.pivot_table(index=["game_id", "period", "team_id"],
                           columns="event_type", values=["mean", "size"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna()
    wide = wide[(wide["size_shot-on-goal"] >= MIN_EVENTS)
                & (wide["size_blocked-shot"] >= MIN_EVENTS)]
    print(f"Comparable team-periods: {len(wide):,}")
    if wide.empty:
        print("Not enough data.")
        return

    same_side = ((wide["mean_shot-on-goal"] > 0) == (wide["mean_blocked-shot"] > 0))
    pct = same_side.mean()
    print(f"\nBlocked shots on the SAME end as that team's shots on goal: {pct:.1%}")
    print(f"Blocked shots on the OPPOSITE end:                          {1-pct:.1%}")

    print()
    if pct >= 0.85:
        print("VERDICT: blocked shots happen in the OWNER's OFFENSIVE zone.")
        print("  The 'D' zone_code is therefore written in the DEFENDING frame,")
        print("  inconsistent with the owner being the shooter.")
        print("  -> ACTION: normalize blocked-shot zones to the owner frame")
        print("     (flip D<->O) before the rush comparison, then rebuild.")
    elif pct <= 0.15:
        print("VERDICT: blocked shots happen in the owner's DEFENSIVE zone.")
        print("  Zone code and owner are already in the same frame.")
        print("  -> ACTION: none. The rush rule is correct as written.")
    else:
        print(f"AMBIGUOUS ({pct:.0%}). Inspect individual games before changing anything.")


if __name__ == "__main__":
    main()