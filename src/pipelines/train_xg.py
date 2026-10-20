"""
train_xg.py
===========
Builds features, trains the xG model, evaluates it, and saves it.

Home: src/pipelines/train_xg.py
Run from the project root:

    py -m src.pipelines.train_xg              # evaluate on the latest season, save
    py -m src.pipelines.train_xg --final      # also retrain on every season for live use
    py -m src.pipelines.train_xg --overwrite  # replace files that already exist

Files are named with MODEL_VERSION (src/models/deterministic/xg.py). Bump it
when features or settings change; existing files are never replaced unless
--overwrite is passed.

Season roles come from config/settings.py:
    CURRENT_SEASON  -> test       (graded once, never used for decisions)
    LAST_SEASON     -> validation (picks the number of trees)
    everything older -> train
"""

import argparse
import json
from pathlib import Path

from config.settings import CURRENT_SEASON, LAST_SEASON, OUTPUT_DIR
from src.data.db import get_connection
from src.models.deterministic.xg import MODEL_VERSION, ExpectedGoalsModel
from src.processing.xg_processing import build_xg_features

MODEL_DIR = Path(OUTPUT_DIR) / "models"


def data_cutoff(conn, seasons) -> str:
    """Date of the last game in the given seasons: the newest data a model saw."""
    marks = ",".join("?" * len(seasons))
    row = conn.execute(
        f"SELECT MAX(game_date) FROM schedule WHERE season IN ({marks})",
        [int(s) for s in seasons]).fetchone()
    return str(row[0])


def save_with_metadata(model, path, cutoff, overwrite, extra=None):
    """Save the model plus a JSON file with its metadata (and scores, if any)."""
    if path.exists() and not overwrite:
        raise SystemExit(
            f"{path.name} already exists. Bump MODEL_VERSION in "
            f"src/models/deterministic/xg.py, or pass --overwrite.")
    model.save(path)
    meta = {**model.metadata(), "data_cutoff": cutoff, **(extra or {})}
    path.with_name(path.stem + "_metrics.json").write_text(json.dumps(meta, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description="Train the xG model.")
    parser.add_argument("--final", action="store_true",
                        help="after evaluating, retrain on every season (incl. the "
                             "test season) and save that as the live model")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace model files that already exist")
    args = parser.parse_args()

    test_season, valid_season = int(CURRENT_SEASON), int(LAST_SEASON)

    # 1. Features
    print("Building features...")
    conn = get_connection()
    feat = build_xg_features(conn)

    train = feat[feat.season < valid_season]
    valid = feat[feat.season == valid_season]
    test = feat[feat.season == test_season]
    print(f"  train {len(train):,}  valid {len(valid):,}  test {len(test):,}")
    if min(len(train), len(valid), len(test)) == 0:
        raise SystemExit("A split is empty. Check CURRENT_SEASON / LAST_SEASON in config/settings.py.")

    # 2. Train (early stopping on valid, then refit on train + valid)
    print("Training...")
    model = ExpectedGoalsModel().fit(train, valid)
    print(f"  trees: {model.n_trees}   seasons: {model.trained_seasons}")

    # 3. Evaluate once on the held-out season
    metrics = model.evaluate(test)
    print("\nTest season", test_season)
    for k, v in metrics.items():
        print(f"  {k:17} {v}")
    print("\nCalibration (test):")
    print(model.calibration(test).round(3).to_string())
    print("\nFeature importance:")
    print(model.feature_importance().round(3).to_string())

    # 4. Save the evaluated model + its scores and metadata
    tag = f"{min(model.trained_seasons)}_{max(model.trained_seasons)}"
    path = save_with_metadata(
        model, MODEL_DIR / f"xg_lgbm_{MODEL_VERSION}_{tag}.pkl",
        data_cutoff(conn, model.trained_seasons), args.overwrite,
        extra={"test_season": test_season, "test_metrics": metrics})
    print(f"\nSaved {path}")

    # 5. Optional: live model trained on everything, same tree count
    if args.final:
        print("\nRetraining on all seasons for live use...")
        live = ExpectedGoalsModel()
        live.n_trees = model.n_trees
        live.fit(feat)
        live_path = save_with_metadata(
            live, MODEL_DIR / f"xg_lgbm_{MODEL_VERSION}_live.pkl",
            data_cutoff(conn, live.trained_seasons), args.overwrite,
            extra={"evaluated_by": path.name})
        print(f"  seasons: {live.trained_seasons}")
        print(f"Saved {live_path}  (no test score: it has seen every season)")
        print("Point XG_MODEL_PATH in config/settings.py at this file.")

    conn.close()


if __name__ == "__main__":
    main()