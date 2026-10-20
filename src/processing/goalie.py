
"""
goalie.py
=========
Per-game goalie workload and results from shot-level xG.
 
Home: src/processing/goalie.py
The Bayesian rating built on this lives in src/models/stochastic/goalie.py.
"""
 
import sqlite3
 
import pandas as pd
 

def goalie_game_xg(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per goalie per game.

    Columns: game_id, game_date, season, goalie_id, goalie_name, team,
             opponent, shots, xga, ga, gsax
    ...
    """
    # 1-3. Each scored shot with its goalie and a goal flag, shots with no
    #      goalie dropped; 4. summed per goalie per game; 5-6. date, season and
    #      the goalie's team (the team that did NOT shoot) from schedule
    df = pd.read_sql("""
        SELECT x.game_id,
               s.game_date,
               s.season,
               se.goalie_id,
               CASE WHEN se.team_id = s.home_team_id
                    THEN s.away_team_abbrev ELSE s.home_team_abbrev END AS team,
               CASE WHEN se.team_id = s.home_team_id
                    THEN s.home_team_abbrev ELSE s.away_team_abbrev END AS opponent,
               COUNT(*)                                             AS shots,
               SUM(x.xg)                                            AS xga,
               SUM(CASE WHEN se.event_type = 'goal' THEN 1 ELSE 0 END) AS ga
        FROM shot_xg x
        JOIN shot_events se
          ON se.game_id = x.game_id AND se.event_id = x.event_id
        JOIN schedule s
          ON s.game_id = x.game_id
        WHERE se.goalie_id IS NOT NULL
        GROUP BY x.game_id, se.goalie_id
        ORDER BY s.game_date, x.game_id
    """, conn)

    # Names from boxscores: covers every goalie who played, not just current rosters
    names = pd.read_sql("""
        SELECT player_id AS goalie_id, MAX(name) AS goalie_name
        FROM goalie_game_stats
        GROUP BY player_id
    """, conn)
    df = df.merge(names, on="goalie_id", how="left")

    # 7. Goals saved above expected
    df["gsax"] = df["xga"] - df["ga"]

    return df[["game_id", "game_date", "season", "goalie_id", "goalie_name",
               "team", "opponent", "shots", "xga", "ga", "gsax"]]
