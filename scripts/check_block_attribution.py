# scripts/check_block_attribution.py
"""
STEP 1 of the shot_context fixes. Run this FIRST — its verdict decides how you
configure src/processing/shot_context.py in step 4.

QUESTION: for a `blocked-shot` event, does the NHL API's eventOwnerTeamId refer
to the team that SHOT the puck, or the team that BLOCKED it?

This matters because the rebound rule credits a shot attempt to the shooting
team. Getting it backwards either invents rebounds (the blocker "shooting
again") or loses real ones (a shooter following up their own blocked attempt).

METHOD: shot_events already stores shooting_player_id on blocked shots. Look up
that player's team and compare it to the stored team_id.

CAVEAT: players.team_abbreviation is the player's CURRENT team, so mid-season
trades add noise. Restricted to the current season to limit that. Read the
result as a landslide (>80% one way), not a precise number.

Run from the project root:  python scripts/check_block_attribution.py
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

from config.settings import CURRENT_SEASON
from src.data.db import get_connection


def main():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT CASE WHEN t.team_id = se.team_id
                    THEN 'owner = SHOOTER' ELSE 'owner = BLOCKER' END AS verdict,
               COUNT(*) AS n
        FROM shot_events se
        JOIN schedule sc ON sc.game_id = se.game_id
        JOIN players  p  ON p.id = se.shooting_player_id
        JOIN teams    t  ON t.abbreviation = p.team_abbreviation
        WHERE se.event_type = 'blocked-shot'
          AND sc.season = ?
        GROUP BY verdict
        ORDER BY n DESC
    """, conn, params=(int(CURRENT_SEASON),))
    conn.close()

    if df.empty:
        print("No rows. Check that shooting_player_id is populated on blocked shots.")
        return

    total = df.n.sum()
    df["pct"] = (df.n / total).round(3)
    print(df.to_string(index=False))

    top = df.iloc[0]
    share = top.n / total
    print()
    if share < 0.8:
        print(f"AMBIGUOUS ({share:.0%}). Too much roster noise to call it. "
              "Inspect a few blocked shots by hand against nhl.com.")
    elif top.verdict == "owner = SHOOTER":
        print(f"VERDICT ({share:.0%}): owner is the SHOOTING team.")
        print("  -> In shot_context.py keep the default:")
        print("     blocked_shots_owned_by_blocker: bool = False")
    else:
        print(f"VERDICT ({share:.0%}): owner is the BLOCKING team.")
        print("  -> In shot_context.py change the default back to:")
        print("     blocked_shots_owned_by_blocker: bool = True")


if __name__ == "__main__":
    main()