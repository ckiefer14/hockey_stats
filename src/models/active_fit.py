"""
active_fit.py
=============
Which saved Track B fit the live pipeline uses.

Home: src/models/active_fit.py

fit_hierarchical.py writes output/posteriors/active_fit.json after every successful fit:
the fit file's name, the model (v1 goals or v1.1 xG), the training window and the
seasons it covers. late_game.py, score_state.py and predict_game.py read it, so they
always use the same fit and the same games without anyone editing four files.

To go back to an older fit, run fit_hierarchical.py again with its settings, or edit
"fit_file" (and the window dates) in active_fit.json by hand.
"""

import json

import pandas as pd

from config.settings import OUTPUT_DIR

POSTERIOR_DIR = OUTPUT_DIR / "posteriors"
ACTIVE_PATH = POSTERIOR_DIR / "active_fit.json"

# Production defaults: the model Phase 7 adopted
DEFAULT_MODEL = "xg"          # Track B v1.1 (ratings from xG + goals)
DEFAULT_WINDOW_DAYS = 365     # 1-year window
DEFAULT_HALF_LIFE = None      # no recency weighting


def write_active(info: dict) -> None:
    """Record a fit as the active one (called by fit_hierarchical.py)."""
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_PATH, "w") as f:
        json.dump(info, f, indent=2)


def read_active() -> dict:
    """The active fit's record. Fails with a clear message if there is none yet."""
    if not ACTIVE_PATH.exists():
        raise SystemExit(f"No active fit: {ACTIVE_PATH} is missing. "
                         f"Run: py -m src.pipelines.fit_hierarchical")
    with open(ACTIVE_PATH) as f:
        return json.load(f)


def active_fit_path():
    """Path to the active fit's NetCDF file."""
    info = read_active()
    path = POSTERIOR_DIR / info["fit_file"]
    if not path.exists():
        raise SystemExit(f"active_fit.json names {info['fit_file']}, but that file is not in {POSTERIOR_DIR}.")
    return path


def window_games(conn, info: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The games the active fit was trained on (same rows and team order as the fit)."""
    from src.processing.game_table import build_game_table, select_window
    info = info or read_active()
    gt_all, _ = build_game_table(conn, info["seasons"])
    gt, teams = select_window(gt_all, info["start"], info["end"])
    if teams != info["teams"]:
        raise SystemExit("The games in the active window no longer match the fit's teams "
                         "(the database changed?). Refit: py -m src.pipelines.fit_hierarchical --overwrite")
    return gt, teams


def check_params_match(params: dict, label: str) -> str | None:
    """Warning text if a late_game / score_state file was measured on a different fit."""
    active = read_active()["fit_file"]
    used = params.get("fit_file")
    if used != active:
        return (f"WARNING  {label} was measured on {used}, but the active fit is {active}. "
                f"Rebuild: py -m src.processing.late_game, then py -m src.processing.score_state")
    return None