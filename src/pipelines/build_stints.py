"""
build_stints.py
===============
v2.2: build stints (stretches with no line change) for every loaded game with a shift chart.

Home: src/pipelines/build_stints.py

Usage (from the project root):
    py -m src.pipelines.build_stints              # games not built yet (weekly / after load_all)
    py -m src.pipelines.build_stints --rebuild    # all games again (first run: several minutes)

Tables:
    stints       one row per stint (src/processing/stints.py for the columns and conventions)
    stint_games  one row per game: status (ok / skipped), reason, n_stints and the quality numbers

A game is rebuilt when its shift chart changed since (more or fewer shift rows than when it was
built: e.g. load_all filled it in later). Skipped games are reported with their reasons.
"""

import argparse
import time

import pandas as pd

from src.data.db import get_connection
from src.processing.stints import STINT_COLS, build_game, load_batch

BATCH = 200

DDL = [
    """CREATE TABLE IF NOT EXISTS stints (
           game_id INTEGER, stint INTEGER, period INTEGER, start REAL, "end" REAL, duration REAL,
           home_skaters TEXT, away_skaters TEXT, home_goalie INTEGER, away_goalie INTEGER,
           n_home INTEGER, n_away INTEGER, strength TEXT, score_diff INTEGER, zone_start TEXT,
           home_shots INTEGER, away_shots INTEGER, home_xg REAL, away_xg REAL,
           home_goals INTEGER, away_goals INTEGER,
           PRIMARY KEY (game_id, stint))""",
    """CREATE TABLE IF NOT EXISTS stint_games (
           game_id INTEGER PRIMARY KEY, status TEXT, reason TEXT, n_stints INTEGER, toi_off REAL,
           bad_count_share REAL, no_goalie_share REAL, unassigned_shots INTEGER, shift_rows INTEGER,
           built_at TEXT)""",
]


def games_to_build(conn, rebuild: bool) -> list[int]:
    """Regular-season games with shift rows that are new, or whose shift-row count changed."""
    cur = pd.read_sql("""SELECT g.id AS game_id, COUNT(s.game_id) AS shift_rows
                         FROM games g JOIN shifts s ON s.game_id = g.id
                         WHERE g.game_type = 2 GROUP BY g.id""", conn)
    if rebuild:
        return cur["game_id"].tolist()
    done = pd.read_sql("SELECT game_id, shift_rows AS built_rows FROM stint_games", conn)
    m = cur.merge(done, on="game_id", how="left")
    return m.loc[m["built_rows"].isna() | (m["built_rows"] != m["shift_rows"]), "game_id"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Build stints from the shift charts.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild every game")
    args = parser.parse_args()
    conn = get_connection()
    if args.rebuild:
        conn.execute("DROP TABLE IF EXISTS stints")
        conn.execute("DROP TABLE IF EXISTS stint_games")
    for ddl in DDL:
        conn.execute(ddl)
    conn.commit()

    ids = games_to_build(conn, args.rebuild)
    print(f"Building stints for {len(ids):,} games...", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_rows = 0
    for b in range(0, len(ids), BATCH):
        chunk = ids[b:b + BATCH]
        data = load_batch(conn, chunk)
        rows = data["shifts"].groupby("game_id").size()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for gid in chunk:
            st, q = build_game(gid, data)
            conn.execute("DELETE FROM stints WHERE game_id = ?", (gid,))
            if len(st):
                conn.executemany(f"INSERT INTO stints VALUES ({','.join('?' * len(STINT_COLS))})",
                                 [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
                                  for r in st[STINT_COLS].itertuples(index=False)])
                n_rows += len(st)
            conn.execute("INSERT OR REPLACE INTO stint_games VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (gid, q["status"], q["reason"], q["n_stints"],
                          None if pd.isna(q["toi_off"]) else q["toi_off"],
                          None if pd.isna(q["bad_count_share"]) else q["bad_count_share"],
                          None if pd.isna(q["no_goalie_share"]) else q["no_goalie_share"],
                          q["unassigned_shots"], int(rows.get(gid, 0)), now))
            n_ok += q["status"] == "ok"
            n_skip += q["status"] != "ok"
        conn.commit()
        el = time.time() - t0
        done = min(b + BATCH, len(ids))
        print(f"  [{done:,}/{len(ids):,}] ok {n_ok:,}, skipped {n_skip:,}, {n_rows:,} stints  "
              f"({el / 60:.1f} min, about {el / done * (len(ids) - done) / 60:.0f} min left)", flush=True)

    # Summary over everything built
    s = pd.read_sql("""SELECT g.season, sg.status, COUNT(*) AS games, SUM(sg.n_stints) AS stints
                       FROM stint_games sg JOIN games g ON g.id = sg.game_id GROUP BY g.season, sg.status""", conn)
    print("\nBy season:")
    print(s.pivot_table(index="season", columns="status", values="games", fill_value=0).astype(int).to_string())
    tot = pd.read_sql("""SELECT strength, COUNT(*) AS stints, SUM(duration) / 3600.0 AS hours,
                                SUM(home_xg + away_xg) AS xg, SUM(home_goals + away_goals) AS goals
                         FROM stints GROUP BY strength ORDER BY hours DESC LIMIT 8""", conn)
    print("\nBy strength (top 8 by time):")
    print(tot.round(1).to_string(index=False))
    sk = pd.read_sql("SELECT reason, COUNT(*) AS games FROM stint_games WHERE status != 'ok' GROUP BY reason "
                     "ORDER BY games DESC LIMIT 10", conn)
    if len(sk):
        print("\nSkipped games, by reason:")
        print(sk.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
