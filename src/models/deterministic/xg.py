"""
xg.py
=====
Expected goals (xG) model: the probability that an unblocked shot becomes a goal.

Home: src/models/deterministic/xg.py

Wraps a LightGBM classifier with the choices validated in Phase 2:
  - features from src/processing/xg_processing.py (FEATURE_COLS)
  - recency weighting, so recent seasons count more (recording habits drift)
  - early stopping on a validation season to pick the number of trees,
    then a refit on train + validation with that tree count

Typical use (see src/pipelines/train_xg.py):
    model = ExpectedGoalsModel()
    model.fit(train_df, valid_df)
    xg = model.predict(test_df)
    model.save("output/models/xg_model.pkl")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.processing.xg_processing import FEATURE_COLS, TARGET_COL

# Settings confirmed by the Phase 2 grid search (31/200 tied the best; 63 leaves was worse)
DEFAULT_PARAMS = {
    "n_estimators": 2000,        # ceiling only; early stopping picks the real number
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

# Weight by age, newest season first. Older seasons than listed get the last value.
RECENCY_WEIGHTS = (2.0, 1.5, 1.25, 1.0)

EARLY_STOPPING_ROUNDS = 100


def recency_weights(seasons: pd.Series) -> pd.Series:
    """Map each row's season to a training weight: newest season counts most."""
    ordered = sorted(seasons.unique(), reverse=True)        # newest first
    lookup = {
        s: RECENCY_WEIGHTS[min(i, len(RECENCY_WEIGHTS) - 1)]
        for i, s in enumerate(ordered)
    }
    return seasons.map(lookup)


class ExpectedGoalsModel:
    """LightGBM xG model with recency weighting and early stopping."""

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMClassifier | None = None
        self.feature_cols = list(FEATURE_COLS)
        self.n_trees: int | None = None
        self.trained_seasons: list[int] = []
        self.trained_at: str | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "ExpectedGoalsModel":
        """Train the model.

        With `valid`: early stopping on `valid` picks the tree count, then the
        final model is refit on train + valid with that count, so it also learns
        from the most recent season. Without `valid`: trains with the current
        tree setting directly (use after a validated run has set n_trees).
        """
        X, y = train[self.feature_cols], train[TARGET_COL]

        if valid is not None:
            # weights come from the full train+valid season set, so a season gets
            # the same weight here as in the final refit
            all_seasons = pd.concat([train.season, valid.season])
            w = recency_weights(all_seasons).iloc[: len(train)].to_numpy()

            probe = lgb.LGBMClassifier(**self.params)
            probe.fit(
                X, y, sample_weight=w,
                eval_set=[(valid[self.feature_cols], valid[TARGET_COL])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            self.n_trees = probe.best_iteration_
            train = pd.concat([train, valid], ignore_index=True)
            X, y = train[self.feature_cols], train[TARGET_COL]

        n_trees = self.n_trees or self.params["n_estimators"]
        self.model = lgb.LGBMClassifier(**{**self.params, "n_estimators": n_trees})
        self.model.fit(X, y, sample_weight=recency_weights(train.season).to_numpy())

        self.n_trees = n_trees
        self.trained_seasons = sorted(int(s) for s in train.season.unique())
        self.trained_at = datetime.now().isoformat(timespec="seconds")
        return self

    # -------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """xG for each shot: the probability it becomes a goal."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        return self.model.predict_proba(df[self.feature_cols])[:, 1]

    # ------------------------------------------------------------- evaluate
    def evaluate(self, df: pd.DataFrame) -> dict:
        """Scores on a labelled set: AUC, log loss, predicted vs actual goals."""
        p, y = self.predict(df), df[TARGET_COL]
        return {
            "shots": int(len(df)),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "log_loss": round(float(log_loss(y, p)), 5),
            "predicted_goals": round(float(p.sum()), 1),
            "actual_goals": int(y.sum()),
            "pred_over_actual": round(float(p.sum() / y.sum()), 4),
        }

    def calibration(self, df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        """Predicted vs actual goal rate in equal-size groups of shots."""
        cal = pd.DataFrame({"xg": self.predict(df), "goal": df[TARGET_COL].to_numpy()})
        cal["bin"] = pd.qcut(cal.xg, bins, labels=False, duplicates="drop")
        return cal.groupby("bin").agg(
            shots=("goal", "size"), predicted=("xg", "mean"), actual=("goal", "mean"))

    def feature_importance(self) -> pd.Series:
        """Share of total gain contributed by each feature."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        gain = pd.Series(self.model.booster_.feature_importance("gain"),
                         index=self.feature_cols)
        return (gain / gain.sum()).sort_values(ascending=False)

    # ---------------------------------------------------------- save / load
    def save(self, path: str | Path) -> Path:
        """Save the trained model and its settings to one file."""
        if self.model is None:
            raise RuntimeError("Nothing to save: model is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExpectedGoalsModel":
        """Load a model saved with save()."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain an ExpectedGoalsModel")
        if model.feature_cols != list(FEATURE_COLS):
            raise ValueError(
                "Saved model's features differ from xg_processing.FEATURE_COLS. "
                "Retrain with src/pipelines/train_xg.py.")
        return model