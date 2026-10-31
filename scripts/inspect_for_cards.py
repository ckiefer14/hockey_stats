"""Print everything needed to finish the card queries.

    py -m scripts.inspect_for_cards > card_inspect.txt

Read-only: opens the database, lists table columns and one example row,
and prints the signatures of the Phase 6 functions. Paste the output
(card_inspect.txt) back into the chat, or use it to edit the CHECK block in
src/pipelines/make_cards.py and load_game_shots() in src/viz/postgame_card.py.
"""

from __future__ import annotations

import importlib
import inspect
import sqlite3
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

TABLES = ["shot_xg", "shot_events", "game_events", "games", "schedule",
          "teams", "players", "rosters", "goalie_game_stats"]

FUNCTIONS = [
    ("src.models.game_model", "predict_matchup"),
    ("src.models.game_model", "outcome_probs"),
    ("src.models.game_model", "load_track_b"),
    ("src.models.stochastic.montecarlo", "simulate_matchup_score_state"),
    ("src.analysis.aggregation", "final_from_sims"),
    ("src.analysis.aggregation", "result_to_json"),
]


def find_db_path():
    import config.settings as s
    names = [n for n in dir(s) if not n.startswith("_")]
    print("config.settings names:", ", ".join(names))
    for n in names:
        v = getattr(s, n)
        if isinstance(v, (str, Path)) and str(v).endswith(".db"):
            print(f"-> database path setting: {n} = {v}")
            return n, v
    raise SystemExit("No *.db path found in config/settings.py")


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    section("1. Database path")
    name, path = find_db_path()

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        section("2. All tables")
        tabs = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
        print(", ".join(tabs["name"]))

        section("3. Columns and one example row per table")
        for t in TABLES:
            if t not in set(tabs["name"]):
                print(f"\n--- {t}: (no such table)")
                continue
            cols = pd.read_sql_query(f"PRAGMA table_info({t})", conn)
            print(f"\n--- {t}")
            print(cols[["name", "type"]].to_string(index=False))
            row = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 1", conn)
            print("example:", row.iloc[0].to_dict() if len(row) else "(empty)")

        section("4. One goal from the latest game in shot_xg")
        try:
            gid = pd.read_sql_query("SELECT MAX(game_id) g FROM shot_xg", conn)["g"][0]
            print("game_id:", gid)
            ev = pd.read_sql_query(
                "SELECT * FROM shot_events WHERE game_id = ? "
                "AND event_type = 'goal' LIMIT 1", conn, params=(int(gid),))
            print(ev.T.to_string())
        except Exception as e:                       # noqa: BLE001
            print("skipped:", e)

    section("5. Function signatures")
    for mod, fn in FUNCTIONS:
        try:
            f = getattr(importlib.import_module(mod), fn)
            print(f"\n{mod}.{fn}{inspect.signature(f)}")
            doc = inspect.getdoc(f)
            if doc:
                print("    " + doc.replace("\n", "\n    ")[:1200])
        except Exception as e:                       # noqa: BLE001
            print(f"\n{mod}.{fn}: not found ({e})")

    section("6. Forward-test log columns")
    files = sorted(Path("output/forward").glob("predictions_*.csv"))
    if files:
        print(files[-1])
        print(pd.read_csv(files[-1], nrows=1).T.to_string())
    else:
        print("no output/forward/predictions_*.csv yet")


if __name__ == "__main__":
    main()