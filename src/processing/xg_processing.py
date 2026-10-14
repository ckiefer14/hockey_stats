"""
xg_processing.py
================
Turns the database into a training table, one row per shot.

Home: src/processing/xg_processing.py

- Pulls the training query (period <= 4, unblocked shots)
- Normalizes coordinates so every shot attacks the same net
- Adds distance and angle to the net
- Adds pre-shot movement from the previous event
- Decodes strength state into shooter-relative features
- Drops empty-net shots
"""

import sqlite3

import numpy as np
import pandas as pd

NET_X = 89                      # goal line, feet from center ice; net at y = 0
TARGET_COL = "is_goal"

# The NHL's labels for these pairs shifted between seasons (2025-26 moved many
# wrist -> snap and deflected -> tip-in), so the split isn't reliable. Group them.
SHOT_TYPE_GROUPS = {
    "wrist": "wrist-snap",
    "snap": "wrist-snap",
    "tip-in": "redirect",
    "deflected": "redirect",
}

FEATURE_COLS: list[str] = [
    "distance",
    "angle",
    "shot_type",
    "is_rebound",
    "is_rush",
    "last_event_type",
    "time_since_last_event",
    "skater_diff",
    "total_skaters",
    "own_goalie_pulled",
    "dist_from_prev",
    "speed_from_prev",
    "rebound_angle_change",
]

CATEGORICAL_COLS = ["shot_type", "last_event_type"]
ID_COLS = ["game_id", "event_id", "season"]

SHOT_QUERY = """
    WITH prev AS (
        SELECT
            game_id,
            event_id,
            LAG(x_coord) OVER w AS prev_x,
            LAG(y_coord) OVER w AS prev_y,
            game_seconds - LAG(game_seconds) OVER w AS prev_gap
        FROM game_events
        WINDOW w AS (PARTITION BY game_id, period ORDER BY sort_order)
    )
    SELECT
        se.game_id,
        se.event_id,
        s.season,
        se.period,
        se.team_id,
        s.home_team_id,
        se.x_coord,
        se.y_coord,
        se.shot_type,
        se.event_type,
        se.zone_code,
        sc.is_rebound,
        sc.is_rush,
        sc.last_event_type,
        sc.time_since_last_event,
        ge.situation_code,
        p.prev_x,
        p.prev_y,
        p.prev_gap
    FROM shot_events se
    JOIN shot_context sc
    ON sc.game_id = se.game_id AND sc.event_id = se.event_id
    JOIN game_events ge
    ON ge.game_id = se.game_id AND ge.event_id = se.event_id
    JOIN schedule s
    ON s.game_id = se.game_id
    LEFT JOIN prev p
    ON p.game_id = se.game_id AND p.event_id = se.event_id
    WHERE se.period <= 4
    AND se.event_type IN ('shot-on-goal', 'goal', 'missed-shot')
"""

def load_shots(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pull one row per eligible shot, with context and strength state."""
    return pd.read_sql(SHOT_QUERY, conn)

def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates so every shot attacks the net at x = +89."""
    df = df.copy()
    keys = ["game_id", "period", "team_id"]

    # 1. Direction from offensive-zone shots only (most reliable sign)
    df["_o_x"] = df["x_coord"].where(df["zone_code"] == "O")
    o_mean = df.groupby(keys)["_o_x"].transform("mean")

    # 2. Fallback: groups with no offensive-zone shots use all their shots
    all_mean = df.groupby(keys)["x_coord"].transform("mean")
    direction = o_mean.fillna(all_mean)

    # 3. Flip groups attacking toward -89 (180° turn: both x and y)
    sign = np.where(direction < 0,-1,1)
    for col in ["x_coord", "y_coord", "prev_x", "prev_y"]:
        df[col] = df[col] * sign

    return df.drop(columns="_o_x")  

def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance (feet) and angle (degrees) from each shot to the net."""
    df = df.copy()
    dx = NET_X - df["x_coord"]
    dy = df["y_coord"].abs()

    df["distance"]  = np.sqrt(dx ** 2 + dy ** 2)
    df["angle"]     = np.degrees(np.arctan2(dy,dx))
    return df


def add_prev_event(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-shot movement features from the event just before each shot."""
    df = df.copy()
    dx = df["x_coord"] - df["prev_x"]
    dy = df["y_coord"] - df["prev_y"]

    # How far the puck moved between the previous event and the shot
    df["dist_from_prev"] = np.sqrt(dx ** 2 + dy ** 2)

    # Feet per second; the feed logs whole seconds, so treat a 0-second gap as 1
    df["speed_from_prev"] = df["dist_from_prev"] / df["prev_gap"].clip(lower=1)

    # Rebounds only: how far the shooting angle swung from the first shot.
    # Signed angles (y not made absolute) so a cross-crease rebound counts fully.
    now  = np.degrees(np.arctan2(df["y_coord"], NET_X - df["x_coord"]))
    then = np.degrees(np.arctan2(df["prev_y"],  NET_X - df["prev_x"]))
    change = (now - then).abs()
    change = np.minimum(change, 360 - change)      # shots behind the net wrap past 180
    df["rebound_angle_change"] = change.where(df["is_rebound"] == 1)
    return df

def decode_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Turn situation_code into shooter-relative strength features.
    
    situation_code is 4 digits: away goalie, away skaters, home skaters, home goalie.
    1551 = 5v5; 1451 = home power play; 0651 = away goalie pulled.
    """
    df = df.copy()
    code = df["situation_code"].astype(str).str.zfill(4)

    away_g = code.str[0].astype(int)
    away_s = code.str[1].astype(int)
    home_s = code.str[2].astype(int)
    home_g = code.str[3].astype(int)

    is_home = df["team_id"] == df["home_team_id"]

    own_s = np.where(is_home, home_s, away_s)
    opp_s = np.where(is_home, away_s, home_s)
    own_g = np.where(is_home, home_g, away_g)
    opp_g = np.where(is_home, away_g, home_g)

    df["skater_diff"] = own_s - opp_s
    df["total_skaters"] = own_s + opp_s
    df["empty_net"] = opp_g == 0
    df["own_goalie_pulled"] = (own_g == 0).astype(int)
    return df

def build_xg_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Full pipeline: database -> one training row per shot."""
    df = load_shots(conn)
    df = normalize_coordinates(df)
    df = add_geometry(df)
    df = add_prev_event(df)
    df = decode_strength(df)

    df = df[~df["empty_net"]].copy()

    df[TARGET_COL] = (df["event_type"] == "goal").astype(int)
    df["shot_type"] = df["shot_type"].fillna("wrist").replace(SHOT_TYPE_GROUPS)
    df["time_since_last_event"] = df["time_since_last_event"].astype(float)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df[ID_COLS + FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)