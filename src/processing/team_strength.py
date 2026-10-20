"""
team_strength.py
================
Rolling team offense and defense estimates from shot-level xG.

Home: src/processing/team_strength.py
"""

import sqlite3

import numpy as np
import pandas as pd

# Games in the rolling window; chosen in phase3_check.ipynb section 4
# (25 and 50 tied; 25 reacts faster)
TEAM_WINDOW = 25

def team_game_xg(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per team per game (two rows per game).

    Columns: game_id, game_date, season, team_id, team, opponent, is_home,
             xgf, xga, gf, ga, gf_xg, ga_xg, minutes

    xgf / xga cover unblocked shots with a goalie in net (the shot_xg table).
    gf / ga are all goals in periods 1-4, empty-net goals included (the
    scoreboard); shootout goals are excluded.
    gf_xg / ga_xg are goals on the same shots as xgf / xga, so empty-net goals
    are excluded. Use them when comparing goals with xG.
    """
    # 1. Start from schedule: two rows per game (home team, away team),
    #    so a team with zero shots still gets a row
    rows = pd.read_sql("""
        SELECT game_id, game_date, season,
               home_team_id AS team_id, home_team_abbrev AS team,
               away_team_abbrev AS opponent, 1 AS is_home
        FROM schedule
        WHERE game_id IN (SELECT DISTINCT game_id FROM shot_xg)

        UNION ALL

        SELECT game_id, game_date, season,
               away_team_id, away_team_abbrev,
               home_team_abbrev, 0
        FROM schedule
        WHERE game_id IN (SELECT DISTINCT game_id FROM shot_xg)

        ORDER BY game_date, game_id, is_home DESC
    """, conn)

    # 2. Sum shot_xg.xg per (game_id, team_id) -> xgf
    xgf = pd.read_sql("""
        SELECT x.game_id, se.team_id, SUM(x.xg) AS xgf
        FROM shot_xg x
        JOIN shot_events se
          ON se.game_id = x.game_id AND se.event_id = x.event_id
        GROUP BY x.game_id, se.team_id
    """, conn)

    # 3. Count goals per (game_id, team_id), period <= 4
    #    gf: all goals (matches the scoreboard)
    #    gf_xg: goals on shots in shot_xg, i.e. excluding empty-net goals,
    #           the same set of shots as xgf
    gf = pd.read_sql("""
        SELECT se.game_id, se.team_id,
               COUNT(*) AS gf,
               SUM(CASE WHEN x.event_id IS NOT NULL THEN 1 ELSE 0 END) AS gf_xg
        FROM shot_events se
        LEFT JOIN shot_xg x
          ON x.game_id = se.game_id AND x.event_id = se.event_id
        WHERE se.event_type = 'goal' AND se.period <= 4
        GROUP BY se.game_id, se.team_id
    """, conn)

    # 4. Game length in minutes from game_events (periods 1-4), at least 60
    minutes = pd.read_sql("""
        SELECT game_id, MAX(game_seconds) / 60.0 AS minutes
        FROM game_events
        WHERE period <= 4
        GROUP BY game_id
    """, conn)
    minutes["minutes"] = minutes["minutes"].clip(lower=60)

    # 5. Attach steps 2-4 to the step 1 rows; fill missing xgf / gf with 0
    df = (rows
          .merge(xgf, on=["game_id", "team_id"], how="left")
          .merge(gf, on=["game_id", "team_id"], how="left")
          .merge(minutes, on="game_id", how="left"))
    df["xgf"] = df["xgf"].fillna(0.0)
    df["gf"] = df["gf"].fillna(0).astype(int)
    df["gf_xg"] = df["gf_xg"].fillna(0).astype(int)
    df["minutes"] = df["minutes"].fillna(60.0)

    # 6. xga / ga = the opponent's xgf / gf in the same game
    #    (each game has exactly two rows, so opponent = game total - own)
    df["xga"] = df.groupby("game_id")["xgf"].transform("sum") - df["xgf"]
    df["ga"] = df.groupby("game_id")["gf"].transform("sum") - df["gf"]
    df["ga_xg"] = df.groupby("game_id")["gf_xg"].transform("sum") - df["gf_xg"]

    # 7. Return the columns listed above
    return df[["game_id", "game_date", "season", "team_id", "team", "opponent",
               "is_home", "xgf", "xga", "gf", "ga", "gf_xg", "ga_xg", "minutes"]]

def rolling_strength(tg: pd.DataFrame, window: int = TEAM_WINDOW) -> pd.DataFrame:
    """Each team's offense and defense estimate going INTO each game.

    Takes the output of team_game_xg() and adds, per row, estimates built only
    from that team's previous `window` games (never the game itself), across
    season boundaries:

        xgf60, xga60          xG for / against per 60 minutes
        xgf60_var, xga60_var  variance of those estimates (shrinks as games add up)
        games_in_window       how many past games went into them (0 to `window`)

    A team's first game has no history, so its estimates are NaN.
    """
    # 1. One row per team-game, in the order the team played them
    df = tg.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    # 2. Each game's own rates, used for the variance
    df["_xgf60_game"] = df["xgf"] / df["minutes"] * 60
    df["_xga60_game"] = df["xga"] / df["minutes"] * 60

    by_team = df.groupby("team")

    def past(col, how):
        """Rolling `how` over the team's previous games. shift(1) drops the
        current game, so an estimate never sees the game it is predicting."""
        return by_team[col].transform(
            lambda s: getattr(s.shift(1).rolling(window, min_periods=1), how)())

    # 3. Rates over the window: total xG / total minutes, so OT games count fairly
    minutes = past("minutes", "sum")
    df["xgf60"] = past("xgf", "sum") / minutes * 60
    df["xga60"] = past("xga", "sum") / minutes * 60

    # 4. Uncertainty: variance of the average = game-to-game variance / games
    df["games_in_window"] = past("minutes", "count").fillna(0).astype(int)
    n = df["games_in_window"].replace(0, np.nan)
    df["xgf60_var"] = past("_xgf60_game", "var") / n
    df["xga60_var"] = past("_xga60_game", "var") / n

    return df.drop(columns=["_xgf60_game", "_xga60_game"])


def strength_as_of(tg: pd.DataFrame, date, window: int = TEAM_WINDOW) -> pd.DataFrame:
    """Each team's offense and defense estimate as of `date`: one row per team.

    Uses each team's last `window` games played strictly before `date`, so it
    is what you would have known going into games on that date. Takes the
    output of team_game_xg(). Teams with no games before `date` are left out.

    Columns: team, games_in_window, last_game, xgf60, xga60, xgf60_var, xga60_var
    """
    # 1. Only games played before the date
    cutoff = pd.Timestamp(date)
    before = tg[pd.to_datetime(tg["game_date"]) < cutoff]

    # 2. Each team's most recent `window` of those games
    recent = (before.sort_values(["team", "game_date", "game_id"])
                    .groupby("team").tail(window)
                    .assign(_xgf60=lambda d: d.xgf / d.minutes * 60,
                            _xga60=lambda d: d.xga / d.minutes * 60))

    # 3. Same formulas as rolling_strength
    g = recent.groupby("team")
    n = g.size()
    out = pd.DataFrame({
        "games_in_window": n,
        "last_game": g["game_date"].max(),
        "xgf60": g["xgf"].sum() / g["minutes"].sum() * 60,
        "xga60": g["xga"].sum() / g["minutes"].sum() * 60,
        "xgf60_var": g["_xgf60"].var() / n,
        "xga60_var": g["_xga60"].var() / n,
    })
    return out.reset_index()