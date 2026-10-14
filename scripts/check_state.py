# scripts/check_state.py
"""
Where am I? Verifies project state against the build sequence in README.md.

Checks three things the eye can't:
  1. Which modules exist on disk
  2. Which tables exist and how full they are
  3. Whether shot_context is actually IN SYNC with the current
     derive_shot_context code — i.e. whether you remembered to rebuild
     after changing windows or the blocked-shot attribution setting

Run from the project root:  python scripts/check_state.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

OK, NO, WARN = "  [x]", "  [ ]", "  [!]"


def check_files():
    print("\nFILES")
    expected = {
        # Phase 1 + this conversation's additions
        "src/data/nhl_api.py": "NHL API client",
        "src/data/db.py": "SQLite operations",
        "src/models/schemas.py": "Pydantic models",
        "src/processing/shot_context.py": "rebound/rush derivation",
        "scripts/backfill_game_events.py": "full-PBP backfill",
        "src/viz/rink_plot.py": "rink plotting",
        "notebooks/data_sanity.ipynb": "sanity checks",
        # Phase 2 deliverables
        "src/models/deterministic/xg.py": "xG model class  <- PHASE 2",
        "src/pipelines/train_xg.py": "xG training pipeline  <- PHASE 2",
    }
    for path, desc in expected.items():
        print(f"{OK if Path(path).exists() else NO} {path:45s} {desc}")

    models = sorted(Path("output/models").glob("*.pkl")) if Path("output/models").exists() else []
    print(f"{OK if models else NO} output/models/*.pkl{'':27s} trained xG pickles: "
          f"{[m.name for m in models] if models else 'none yet'}")


def check_tables(conn):
    print("\nTABLES")
    existing = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ["teams", "players", "schedule", "games", "skater_game_stats",
              "goalie_game_stats", "shifts", "shot_events",
              "game_events", "shot_context"]:
        if t in existing:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{OK} {t:22s} {n:>10,} rows")
        else:
            print(f"{NO} {t:22s} {'MISSING':>10s}")
    return existing


def check_coverage(conn, existing):
    print("\nCOVERAGE")
    if "game_events" not in existing:
        print(f"{NO} game_events missing -> run scripts/backfill_game_events.py")
        return
    gaps = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT game_id FROM shot_events
            WHERE game_id NOT IN (SELECT DISTINCT game_id FROM game_events))
    """).fetchone()[0]
    print(f"{OK if gaps == 0 else NO} games with shots but no full events: {gaps:,}")

    if "shot_context" not in existing:
        print(f"{NO} shot_context missing -> run build_shot_context(conn)")
        return
    no_ctx = conn.execute("""
        SELECT COUNT(*) FROM shot_events se
        LEFT JOIN shot_context sc
               ON sc.game_id = se.game_id AND sc.event_id = se.event_id
        WHERE se.period <= 4
          AND se.event_type IN ('shot-on-goal','goal','missed-shot')
          AND sc.event_id IS NULL
    """).fetchone()[0]
    print(f"{OK if no_ctx == 0 else NO} eligible shots missing context: {no_ctx:,}")

    reb, rush, n = conn.execute("""
        SELECT AVG(is_rebound), AVG(is_rush), COUNT(*) FROM shot_context
    """).fetchone()
    print(f"      flag rates over {n:,} shots: rebound {reb:.1%}, rush {rush:.1%}")


def check_sync(conn, existing, n_games=8):
    """Re-derive a few games with the CURRENT code and diff against stored rows.
    Mismatches mean the code changed after the last build -> rebuild needed."""
    print("\nSYNC (stored shot_context vs current derivation code)")
    if "shot_context" not in existing:
        print(f"{NO} nothing to compare")
        return
    try:
        import pandas as pd
        from src.processing.shot_context import derive_shot_context
    except ImportError as e:
        print(f"{WARN} could not import derivation: {e}")
        return

    gids = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_id FROM shot_context ORDER BY RANDOM() LIMIT ?", (n_games,))]
    mismatched = 0
    for gid in gids:
        events = pd.read_sql("""
            SELECT game_id, event_id, sort_order, period, game_seconds,
                   event_type, team_id, zone_code
            FROM game_events WHERE game_id = ?""", conn, params=(gid,))
        pair = conn.execute(
            "SELECT home_team_id, away_team_id FROM schedule WHERE game_id = ?",
            (gid,)).fetchone()
        fresh = derive_shot_context(events, team_pair=tuple(pair) if pair else None)
        stored = pd.read_sql(
            "SELECT event_id, is_rebound, is_rush FROM shot_context WHERE game_id = ?",
            conn, params=(gid,))
        merged = fresh.merge(stored, on="event_id", suffixes=("_new", "_old"))
        diff = ((merged.is_rebound_new.astype(int) != merged.is_rebound_old.astype(int))
                | (merged.is_rush_new.astype(int) != merged.is_rush_old.astype(int))).sum()
        mismatched += int(diff)

    if mismatched == 0:
        print(f"{OK} {len(gids)} games re-derived, 0 flag mismatches — table matches code")
    else:
        print(f"{NO} {mismatched} flag mismatches across {len(gids)} games")
        print("      -> code changed since last build. Run:")
        print("         build_shot_context(conn, rebuild=True)")


def main():
    from src.data.db import get_connection
    conn = get_connection()
    check_files()
    existing = check_tables(conn)
    check_coverage(conn, existing)
    check_sync(conn, existing)
    conn.close()
    print("\nAll [x] through SYNC => Phase 1 + feature prep complete; "
          "Phase 2 is writing xg.py.\n")


if __name__ == "__main__":
    main()