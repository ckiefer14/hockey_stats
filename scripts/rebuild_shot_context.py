# scripts/rebuild_shot_context.py
"""
STEP 5 of the fixes. Re-derives every shot_context row from game_events using
the CURRENT derivation code. No network calls — a few minutes on ~566k rows.

Run this after ANY change to src/processing/shot_context.py: the rebound/rush
windows, the blocked-shot attribution flag, or the stoppage rules.

Run from the project root:  python scripts/rebuild_shot_context.py
"""
import sys
import time

sys.path.insert(0, ".")

from src.data.db import get_connection
from src.processing.shot_context import build_shot_context


def main():
    conn = get_connection()
    print("Rebuilding shot_context from game_events (this wipes and re-derives)...")
    start = time.time()
    n = build_shot_context(conn, rebuild=True)
    conn.close()
    print(f"Wrote {n:,} rows in {time.time() - start:.0f}s.")
    print("Now re-run notebooks/data_sanity.ipynb sections 10-11.")


if __name__ == "__main__":
    main()