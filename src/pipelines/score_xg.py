"""
score_xg.py
===========
Scores every shot with the xG model and stores the result in the shot_xg table.
Phase 3's team strength and goalie ratings are built from this table.

Home: src/pipelines/score_xg.py
Run from the project root (after train_xg.py --final):

    py -m src.pipelines.score_xg                    # score every season
    py -m src.pipelines.score_xg --season 20262027  # score just one season

Which model scores which season
-------------------------------
A shot's xG must come from a model that did not train on it.

  * Seasons the live model trained on (2021-22 to 2025-26): leave-one-season-out.
    Each season is scored by a model trained on the other seasons, with the same
    settings and tree count as the evaluated model.
  * Seasons the live model has not seen (2026-27 onward): the live model.

Blocked and empty-net shots get no xG (same population as xg_processing.py).
Scoring a season replaces that season's rows, so reruns never duplicate.
"""

import argparse
import json
from pathlib import Path

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.deterministic.xg import MODEL_VERSION, ExpectedGoalsModel, load_active_model
from src.processing.xg_processing import TARGET_COL, build_xg_features

MODEL_DIR = Path(OUTPUT_DIR) / "models"

SHOT_XG_DDL = """
    CREATE TABLE IF NOT EXISTS shot_xg (
        game_id INTEGER,
        event_id INTEGER,
        season INTEGER,
        xg REAL,
        model_version TEXT,
        scored_by TEXT,
        PRIMARY KEY (game_id, event_id),
        FOREIGN KEY (game_id) REFERENCES games(id)
    )
"""


def evaluated_tree_count() -> int:
    """Tree count from the evaluated model's metrics file for this version."""
    for path in sorted(MODEL_DIR.glob(f"xg_lgbm_{MODEL_VERSION}_*_metrics.json")):
        meta = json.loads(path.read_text())
        if "test_metrics" in meta:              # the evaluated model, not the live one
            return int(meta["trees"])
    raise SystemExit(f"No evaluated {MODEL_VERSION} model found in {MODEL_DIR}. "
                     "Run: py -m src.pipelines.train_xg --final")


def write_season(conn, season, shots, xg, scored_by) -> None:
    """Replace one season's rows in shot_xg."""
    conn.execute("DELETE FROM shot_xg WHERE season = ?", (int(season),))
    conn.executemany(
        "INSERT INTO shot_xg (game_id, event_id, season, xg, model_version, scored_by) "
        "VALUES (?,?,?,?,?,?)",
        [(int(g), int(e), int(season), float(x), MODEL_VERSION, scored_by)
         for g, e, x in zip(shots.game_id, shots.event_id, xg)],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Score every shot with the xG model.")
    parser.add_argument("--season", type=int, help="score only this season, e.g. 20262027")
    args = parser.parse_args()

    conn = get_connection()
    conn.execute(SHOT_XG_DDL)

    print("Building features...")
    feat = build_xg_features(conn)
    live = load_active_model()
    trees = evaluated_tree_count()

    seasons = sorted(int(s) for s in feat.season.unique())
    if args.season:
        if args.season not in seasons:
            raise SystemExit(f"No shots for season {args.season} in the database.")
        seasons = [args.season]

    print(f"Model {MODEL_VERSION}, {trees} trees per leave-one-out model\n")
    print(f"{'season':>9} {'shots':>8} {'xG':>9} {'goals':>7} {'xG/goals':>9}  scored by")

    for season in seasons:
        shots = feat[feat.season == season]

        if season in live.trained_seasons:
            # Leave-one-season-out: train on every other season
            fold = ExpectedGoalsModel()
            fold.n_trees = trees
            fold.fit(feat[feat.season != season])
            xg = fold.predict(shots)
            scored_by = f"loso_{MODEL_VERSION}_without_{season}"
        else:
            # A season the live model never saw
            xg = live.predict(shots)
            scored_by = f"live_{MODEL_VERSION}"

        write_season(conn, season, shots, xg, scored_by)
        goals = int(shots[TARGET_COL].sum())
        print(f"{season:>9} {len(shots):>8,} {xg.sum():>9,.1f} {goals:>7,} "
              f"{xg.sum() / goals:>9.3f}  {scored_by}")

    total = conn.execute("SELECT COUNT(*) FROM shot_xg").fetchone()[0]
    conn.close()
    print(f"\nshot_xg now holds {total:,} shots.")


if __name__ == "__main__":
    main()