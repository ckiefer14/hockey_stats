# scripts/backfill_game_events.py
"""
Backfill the `game_events` table (full play-by-play, ALL event types) for
games already loaded by load_all.py, then materialize shot_context.

Why this exists: shot_events only stores shot-type events, but rush
detection needs the surrounding non-shot events (N/D-zone hits, takeaways,
giveaways). One run covers all backfilled seasons (~6,560 games at 1s
sleep is roughly 2.5 hours — run it overnight); thereafter load_all.py
keeps game_events current for new games.

Prerequisites (all delivered alongside this script):
  - GameEvent model in src/models/schemas.py
  - fetch_pbp_json / parse_game_events in src/data/nhl_api.py
  - game_events DDL + store_game_events + game_events_already_loaded
    in src/data/db.py
  - src/processing/shot_context.py
"""
import sys
import time

sys.path.insert(0, ".")

from src.data.nhl_api import fetch_pbp_json, parse_game_events
from src.data.db import (
    get_connection, create_tables,
    store_game_events, game_events_already_loaded,
)
from src.processing.shot_context import build_shot_context

FAILURE_LOG = "data_load_failures.log"
SLEEP_BETWEEN_CALLS = 1.0
SLEEP_AFTER_FAILURE = 3.0


def log_failure(msg: str) -> None:
    print(msg)
    with open(FAILURE_LOG, "a") as f:
        f.write(f"{msg}\n")


def main():
    conn = get_connection()
    create_tables(conn)  # picks up the new game_events DDL

    # every game we already have shot data for needs full events too
    game_ids = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT game_id FROM shot_events ORDER BY game_id"
        )
    ]
    print(f"Games with shot data: {len(game_ids)}")

    skipped = 0
    for i, game_id in enumerate(game_ids):
        # Resume logic - skip if already loaded
        if game_events_already_loaded(conn, game_id):
            skipped += 1
            continue

        try:
            events = parse_game_events(fetch_pbp_json(game_id), game_id)
            if not events:
                log_failure(f"[{i+1}/{len(game_ids)}] Game {game_id}: empty PBP")
                continue
            store_game_events(conn, events)
            print(f"  [{i+1}/{len(game_ids)}] game {game_id}: {len(events)} events")
            time.sleep(SLEEP_BETWEEN_CALLS)
        except Exception as e:
            log_failure(f"[{i+1}/{len(game_ids)}] Game {game_id} FAILED: {e}")
            time.sleep(SLEEP_AFTER_FAILURE)

    print(f"Backfill done. Skipped {skipped} already-loaded games.")

    print("Deriving shot context (rebound / rush flags)...")
    n = build_shot_context(conn)
    print(f"Wrote {n} shot_context rows.")
    conn.close()


if __name__ == "__main__":
    main()
