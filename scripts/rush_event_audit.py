# scripts/rush_event_audit.py
"""
EVENT-LEVEL rush audit for a single game.

Counts of 0-3 per team are too small for ratio/correlation tests to mean
anything, but small enough to verify by hand. This lists every rush-flagged
shot with its time, shooter and precursor, so you can match them against NST's
Individual table (which player, how many) and against the play-by-play.

It also lists FACEOFF-DRIVEN near-misses. Open question worth settling:
NST's rule is "within 4 seconds of any event in the neutral or defensive zone
without a stoppage in play in between". Our derivation treats a faceoff as a
chain-breaker, but the stoppage happens BEFORE the faceoff -- the faceoff is
where play RESUMES, and is itself an event in a zone. Under a looser reading, a
shot 3s after a neutral-zone faceoff is a rush. That reading was rejected on
conversion evidence (those shots scored at 0.8% vs a 7.1% baseline), but if NST
credits rush attempts we don't flag, this is the likely reason.

Run from the project root:
    python scripts/rush_event_audit.py 2025020993
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

import src.processing.shot_context as sc
from src.data.db import get_connection


def load_events(conn, game_id):
    return pd.read_sql("""
        SELECT game_id, event_id, sort_order, period, game_seconds,
               event_type, team_id, zone_code
        FROM game_events WHERE game_id = ?
    """, conn, params=(game_id,))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    game_id = int(sys.argv[1])

    conn = get_connection()
    meta = conn.execute("""
        SELECT game_date, away_team_abbrev, home_team_abbrev, home_team_id, away_team_id
        FROM schedule WHERE game_id = ?""", (game_id,)).fetchone()
    if meta is None:
        print(f"game_id {game_id} not found in schedule")
        return
    date, away, home, home_id, away_id = meta
    abbrev = {home_id: home, away_id: away}

    events = load_events(conn, game_id)
    names = dict(conn.execute("""
        SELECT id, last_name FROM players""").fetchall())
    shooters = dict(conn.execute("""
        SELECT event_id, shooting_player_id FROM shot_events WHERE game_id = ?
    """, (game_id,)).fetchall())
    conn.close()

    strict = sc.derive_shot_context(events, team_pair=(home_id, away_id))

    # same derivation, but a faceoff no longer breaks the chain
    original = sc.STOPPAGE_EVENTS
    try:
        sc.STOPPAGE_EVENTS = original - {"faceoff"}
        loose = sc.derive_shot_context(events, team_pair=(home_id, away_id))
    finally:
        sc.STOPPAGE_EVENTS = original

    ev = events.set_index("event_id")

    def describe(ctx, mask_col="is_rush"):
        rows = []
        for r in ctx[ctx[mask_col]].itertuples():
            e = ev.loc[r.event_id]
            mmss = f"{int(e.game_seconds % 1200)//60:02d}:{int(e.game_seconds % 1200)%60:02d}"
            pid = shooters.get(r.event_id)
            rows.append({
                "team": abbrev.get(e.team_id, e.team_id),
                "P": int(e.period), "time": mmss,
                "shot": e.event_type,
                "shooter": names.get(pid, "?"),
                "precursor": r.last_event_type,
                "gap_s": None if pd.isna(r.time_since_last_event) else round(r.time_since_last_event, 1),
            })
        return pd.DataFrame(rows)

    print(f"\n=== {game_id}  {away} @ {home}  {date} ===")
    print(f"\nRUSH-FLAGGED SHOTS (current derivation) — compare to NST Rush Attempts")
    strict_rush = describe(strict)
    print(strict_rush.to_string(index=False) if len(strict_rush) else "  (none)")
    if len(strict_rush):
        print("\n  per team:", strict_rush.team.value_counts().to_dict())

    extra_ids = set(loose[loose.is_rush].event_id) - set(strict[strict.is_rush].event_id)
    print(f"\nFACEOFF-DRIVEN NEAR-MISSES ({len(extra_ids)}) — flagged only if a faceoff")
    print("is allowed to start a rush chain. If NST credits attempts we miss, look here.")
    if extra_ids:
        extra = describe(loose[loose.event_id.isin(extra_ids)])
        print(extra.to_string(index=False))
        print("\n  per team:", extra.team.value_counts().to_dict())
    else:
        print("  (none)")

    print("\nNOTE: NST Rush Attempts also counts BLOCKED shots; we never flag those.")
    print("So expect NST >= our count even when every flagged event agrees.")


if __name__ == "__main__":
    main()