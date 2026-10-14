# scripts/fix_goal_shooters.py
"""
STEP 2 of the fixes. One-time repair, safe to re-run (idempotent).

PROBLEM: every `goal` row in shot_events has shooting_player_id = NULL
(data_sanity section 6: null_shooter == the exact total goal count). The NHL API
puts the scorer under details.scoringPlayerId, but parse_shot_events only read
details.shootingPlayerId.

FIX: game_events already stores the scorer correctly (its parser falls back to
scoringPlayerId), so we can repair shot_events from local data — no re-download.

Also fix the parser so this doesn't come back — see STEP 3 in the chat.

Run from the project root:  python scripts/fix_goal_shooters.py
"""
import sys

sys.path.insert(0, ".")

from src.data.db import get_connection


def main():
    conn = get_connection()

    before = conn.execute("""
        SELECT COUNT(*) FROM shot_events
        WHERE event_type = 'goal' AND shooting_player_id IS NULL
    """).fetchone()[0]
    print(f"Goals missing a shooter before: {before:,}")

    if before == 0:
        print("Nothing to do.")
        conn.close()
        return

    conn.execute("""
        UPDATE shot_events
        SET shooting_player_id = (
            SELECT ge.shooting_player_id
            FROM game_events ge
            WHERE ge.game_id = shot_events.game_id
              AND ge.event_id = shot_events.event_id
        )
        WHERE event_type = 'goal'
          AND shooting_player_id IS NULL
    """)
    conn.commit()

    after = conn.execute("""
        SELECT COUNT(*) FROM shot_events
        WHERE event_type = 'goal' AND shooting_player_id IS NULL
    """).fetchone()[0]
    print(f"Goals missing a shooter after:  {after:,}")
    print(f"Repaired: {before - after:,}")
    if after:
        print("  (leftovers = goals whose game_events row also lacks a scorer, "
              "e.g. some empty-net or awarded goals. Expected to be a small number.)")
    conn.close()


if __name__ == "__main__":
    main()