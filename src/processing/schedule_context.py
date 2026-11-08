"""
schedule_context.py
===================
Per-game schedule context for both teams, from the schedule alone (so it is known before
puck drop and works for future games): rest days, back-to-backs, travel since the last
game, time zones crossed, and crowded stretches.

Home: src/processing/schedule_context.py

Columns (one row per regular-season game, <side> = home / away):
    rest_<side>    days since the team's previous game (NaN for its first game on record)
    b2b_<side>     1 if the team also played the day before
    km_<side>      great-circle km from the previous game's arena to this one (0 if unknown)
    trav_<side>    overnight travel: b2b * km / 1000 (thousands of km flown between nights)
    tz_<side>      time zones crossed since the previous game, counted only with rest <= 2
                   days (jet lag fades), absolute hours
    dense_<side>   1 if this is the team's 3rd game in 4 nights

A game is played at the home team's arena. Neutral-site games (Global Series, some outdoor
games) are found from games.venue when the game is loaded: a venue that is not the home
team's usual one that season. Their location is unknown, so travel into and out of them
counts as 0. Rest and density still count.

Locations are city-level (a few km does not matter). Time zones are standard-time offsets;
Arizona's lack of daylight time is ignored (one team, at most one hour, half the year).
"""

import sqlite3

import numpy as np
import pandas as pd

FRANCHISE = {"ARI": "UTA"}     # relocated franchises count as one team (ratings, rest)

# Arena (lat, lon, standard UTC offset). Keyed by the abbreviation the schedule uses, so
# ARI keeps its own (Arizona) location even though its games count toward UTA's ratings.
ARENAS = {
    "ANA": (33.808, -117.876, -8), "ARI": (33.428, -111.932, -7), "BOS": (42.366, -71.062, -5),
    "BUF": (42.875, -78.876, -5), "CAR": (35.803, -78.722, -5), "CBJ": (39.969, -83.006, -5),
    "CGY": (51.037, -114.052, -7), "CHI": (41.881, -87.674, -6), "COL": (39.749, -105.008, -7),
    "DAL": (32.790, -96.810, -6), "DET": (42.341, -83.055, -5), "EDM": (53.547, -113.498, -7),
    "FLA": (26.158, -80.326, -5), "LAK": (34.043, -118.267, -8), "MIN": (44.945, -93.101, -6),
    "MTL": (45.496, -73.569, -5), "NJD": (40.734, -74.171, -5), "NSH": (36.159, -86.779, -6),
    "NYI": (40.712, -73.726, -5), "NYR": (40.750, -73.993, -5), "OTT": (45.297, -75.927, -5),
    "PHI": (39.901, -75.172, -5), "PIT": (40.439, -79.989, -5), "SEA": (47.622, -122.354, -8),
    "SJS": (37.333, -121.901, -8), "STL": (38.627, -90.203, -6), "TBL": (27.943, -82.452, -5),
    "TOR": (43.643, -79.379, -5), "UTA": (40.768, -111.901, -7), "VAN": (49.278, -123.109, -8),
    "VGK": (36.103, -115.178, -8), "WSH": (38.898, -77.021, -5), "WPG": (49.893, -97.144, -6),
}
FEATURES = ["trav", "tz", "dense"]          # the v2.0f model terms (b2b is v2.0d)
CONTEXT_COLS = [f"{c}_{s}" for c in ["rest", "b2b", "km", "trav", "tz", "dense"] for s in ("home", "away")]


def _km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (arrays)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _neutral_games(conn: sqlite3.Connection, s: pd.DataFrame) -> set:
    """Loaded games played away from the home team's usual arena that season."""
    try:
        v = pd.read_sql("SELECT id AS game_id, venue FROM games WHERE game_type = 2", conn)
    except Exception:
        return set()
    v = v.dropna(subset=["venue"]).merge(s[["game_id", "season", "site"]], on="game_id")
    if v.empty:
        return set()
    usual = v.groupby(["season", "site"])["venue"].agg(lambda x: x.mode().iloc[0]).rename("usual")
    v = v.join(usual, on=["season", "site"])
    return set(v.loc[v["venue"] != v["usual"], "game_id"])


def schedule_context(conn: sqlite3.Connection) -> pd.DataFrame:
    """CONTEXT_COLS for every regular-season game in the schedule (played and upcoming)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(schedule)")}
    where = "WHERE game_type = 2" if "game_type" in cols else ""
    s = pd.read_sql(f"""SELECT game_id, season, game_date, home_team_abbrev AS home, away_team_abbrev AS away
                        FROM schedule {where}""", conn)
    s["game_date"] = pd.to_datetime(s["game_date"])
    s["site"] = s["home"]                                   # arena = home team's (schedule abbreviation)
    neutral = _neutral_games(conn, s)
    loc = s["site"].map(ARENAS)
    known = loc.notna() & ~s["game_id"].isin(neutral)
    s["lat"] = [a[0] if k else np.nan for a, k in zip(loc, known)]
    s["lon"] = [a[1] if k else np.nan for a, k in zip(loc, known)]
    s["utc"] = [a[2] if k else np.nan for a, k in zip(loc, known)]
    s[["home", "away"]] = s[["home", "away"]].replace(FRANCHISE)

    long = pd.concat([s.assign(team=s["home"], side="home"), s.assign(team=s["away"], side="away")])
    long = long.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    g = long.groupby("team")
    long["rest"] = g["game_date"].diff().dt.days
    long["b2b"] = (long["rest"] == 1).astype(int)
    prev_lat, prev_lon, prev_utc = g["lat"].shift(), g["lon"].shift(), g["utc"].shift()
    km = _km(prev_lat, prev_lon, long["lat"], long["lon"])
    long["km"] = np.where(np.isnan(km), 0.0, km)
    long["trav"] = long["b2b"] * long["km"] / 1000.0
    tz = (long["utc"] - prev_utc).abs()
    long["tz"] = np.where(tz.notna() & (long["rest"] <= 2), tz, 0.0)
    two_back = g["game_date"].shift(2)
    long["dense"] = ((long["game_date"] - two_back).dt.days <= 3).astype(int)   # 3 games within 4 nights

    vals = ["rest", "b2b", "km", "trav", "tz", "dense"]
    wide = long.pivot_table(index="game_id", columns="side", values=vals, aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    out = wide.reset_index()
    for c in CONTEXT_COLS:
        if c not in out:
            out[c] = np.nan if c.startswith("rest") else 0.0
        elif not c.startswith("rest"):
            out[c] = out[c].fillna(0.0)
    for c in [c for c in CONTEXT_COLS if c.startswith(("b2b", "dense"))]:
        out[c] = out[c].astype(int)
    return out[["game_id"] + CONTEXT_COLS]


def context_for_game(conn: sqlite3.Connection, game_id: int) -> dict:
    """Schedule context of one game as a dict (zeros / NaN rest if the game is not in the schedule)."""
    ctx = schedule_context(conn)
    row = ctx[ctx["game_id"] == int(game_id)]
    if row.empty:
        return {c: (float("nan") if c.startswith("rest") else 0) for c in CONTEXT_COLS}
    return row.iloc[0][CONTEXT_COLS].to_dict()