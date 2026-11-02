"""
game_table.py
=============
One row per game for the Track B model: teams, goals (no empty-netters;
all periods 1-4 and regulation only 1-3), starting goalies and their fixed
gamma offsets.

Home: src/processing/game_table.py
"""

import sqlite3

import numpy as np
import pandas as pd
from src.models.stochastic.goalie import goalie_rating_as_of
from src.processing.team_strength import team_game_xg
from src.processing.goalie import goalie_game_xg


def starting_goalies(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per team per game: the goalie who started.

    Columns: game_id, team, goalie_id
    """
    return pd.read_sql("""
        SELECT  game_id,
                team_abbreviation AS team,
                player_id AS goalie_id
        FROM    goalie_game_stats
        WHERE   starter = 1
    """, conn)


def regulation_goals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Goals in periods 1-3 per game, empty-net goals excluded.

    Same shot set as gf_xg (shots in shot_xg), so the only difference from
    home_goals / away_goals is overtime.
    Columns: game_id, home_goals_reg, away_goals_reg
    """
    return pd.read_sql("""
        SELECT  se.game_id,
                SUM(CASE WHEN se.team_id = s.home_team_id THEN 1 ELSE 0 END) AS home_goals_reg,
                SUM(CASE WHEN se.team_id = s.away_team_id THEN 1 ELSE 0 END) AS away_goals_reg
        FROM    shot_events se
        JOIN    shot_xg x  ON x.game_id = se.game_id AND x.event_id = se.event_id
        JOIN    schedule s ON s.game_id = se.game_id
        WHERE   se.event_type = 'goal'
        AND     se.period <= 3
        GROUP BY se.game_id
    """, conn)


def regulation_xg(conn: sqlite3.Connection) -> pd.DataFrame:
    """Expected goals in periods 1-3 per game (shots in shot_xg, so no empty-net shots).

    Columns: game_id, home_xg_reg, away_xg_reg
    """
    return pd.read_sql("""
        SELECT  se.game_id,
                SUM(CASE WHEN se.team_id = s.home_team_id THEN x.xg ELSE 0 END) AS home_xg_reg,
                SUM(CASE WHEN se.team_id = s.away_team_id THEN x.xg ELSE 0 END) AS away_xg_reg
        FROM    shot_xg x
        JOIN    shot_events se ON se.game_id = x.game_id AND se.event_id = x.event_id
        JOIN    schedule s     ON s.game_id = se.game_id
        WHERE   se.period <= 3
        GROUP BY se.game_id
    """, conn)


def goalie_gamma(gg: pd.DataFrame, dates) -> pd.DataFrame:
    """Each goalie's fixed offset as of each game date.

    gamma = -log(theta_mean), Phase 3 rating using only games before the date.
    Positive = better than average.
    Columns: game_date, goalie_id, gamma
    """
    frames = []
    for date in sorted(set(dates)):
        pred = goalie_rating_as_of(gg, date)

        frames.append(pd.DataFrame({
            "game_date": date,
            "goalie_id": pred["goalie_id"],
            "gamma": -np.log(pred["theta_mean"]),
        }))

    return pd.concat(frames, ignore_index=True)


# Relocated franchises: one team for the models (same roster and staff). Arizona moved to
# Utah after 2023-24; without this, Utah starts from zero and ARI keeps a data-free rating.
FRANCHISE = {"ARI": "UTA"}


def _schedule_games(conn: sqlite3.Connection) -> pd.DataFrame:
    """Regular-season schedule (played and upcoming), team names franchise-mapped."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(schedule)")}
    where = "WHERE game_type = 2" if "game_type" in cols else ""
    s = pd.read_sql(f"""SELECT game_id, game_date, home_team_abbrev AS home, away_team_abbrev AS away
                        FROM schedule {where}""", conn)
    s["game_date"] = pd.to_datetime(s["game_date"])
    s[["home", "away"]] = s[["home", "away"]].replace(FRANCHISE)
    return s


def team_rest(conn: sqlite3.Connection) -> pd.DataFrame:
    """Days since each side's previous regular-season game (from the schedule).

    Columns: game_id, rest_home, rest_away (NaN for a team's first game on record),
             b2b_home, b2b_away (1 if that team also played the day before).
    Known before puck drop, so usable for predictions.
    """
    s = _schedule_games(conn)
    long = pd.concat([s[["game_id", "game_date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
                      s[["game_id", "game_date", "away"]].rename(columns={"away": "team"}).assign(side="away")])
    long = long.sort_values(["team", "game_date", "game_id"])
    long["rest"] = long.groupby("team")["game_date"].diff().dt.days
    wide = long.pivot_table(index="game_id", columns="side", values="rest", aggfunc="first")
    out = pd.DataFrame({"game_id": wide.index, "rest_home": wide.get("home"), "rest_away": wide.get("away")})
    out["b2b_home"] = (out["rest_home"] == 1).astype(int)
    out["b2b_away"] = (out["rest_away"] == 1).astype(int)
    return out.reset_index(drop=True)


def rest_on_date(conn: sqlite3.Connection, team: str, date) -> float:
    """Days since `team`'s previous scheduled regular-season game before `date` (NaN if none)."""
    s = _schedule_games(conn)
    team = FRANCHISE.get(team, team)
    d = pd.Timestamp(date).normalize()
    prev = s[((s["home"] == team) | (s["away"] == team)) & (s["game_date"] < d)]["game_date"]
    return float((d - prev.max()).days) if len(prev) else float("nan")


def build_game_table(conn: sqlite3.Connection, seasons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One row per game in `seasons` for the Track B model.

    Columns: game_id, game_date, season, home, away, home_idx, away_idx,
             home_goals, away_goals (periods 1-4), home_goals_reg, away_goals_reg
             (periods 1-3, what Track B fits), home_xg_reg, away_xg_reg (periods 1-3 xG),
             home_goalie, away_goalie,
             gamma_home, gamma_away, rest_home, rest_away, b2b_home, b2b_away
    Relocated franchises (FRANCHISE: ARI -> UTA) count as one team.
    Also returns `teams`: teams[i] is the abbreviation for index i.
    """
    # 1. Games: home rows of team_game_xg in these seasons, renamed
    #    (team -> home, opponent -> away, gf_xg -> home_goals, ga_xg -> away_goals)
    tg = team_game_xg(conn)

    keep = (tg["is_home"] == 1) & (tg["season"].astype(str).isin(seasons))
    games = tg[keep]

    games = games[["game_id", "game_date", "season", "team", "opponent","gf_xg","ga_xg"]]
    games = games.rename(columns={
        "team": "home",
        "opponent": "away",
        "gf_xg": "home_goals",
        "ga_xg": "away_goals",
    })
    # 2. Starters: merge starting_goalies twice (home, then away)
    starters = starting_goalies(conn)

    df = games.merge(
        starters.rename(columns={"team": "home", "goalie_id": "home_goalie"}),
        on=["game_id", "home"], how="left",
    )
    df = df.merge(
        starters.rename(columns={"team": "away", "goalie_id": "away_goalie"}),
        on=["game_id", "away"], how="left",
    )
    # 2b. Regulation goals (periods 1-3); games that were 0-0 after
    #     regulation have no goal rows, so fill with 0
    df = df.merge(regulation_goals(conn), on="game_id", how="left")
    df[["home_goals_reg", "away_goals_reg"]] = (
        df[["home_goals_reg", "away_goals_reg"]].fillna(0).astype(int))

    # 2c. Regulation xG (the information Track B v1.1 rates teams on)
    df = df.merge(regulation_xg(conn), on="game_id", how="left")
    df[["home_xg_reg", "away_xg_reg"]] = df[["home_xg_reg", "away_xg_reg"]].fillna(0.0)

    # 3. Gammas: goalie_gamma on the unique dates, merged twice; fill missing with 0
    gg = goalie_game_xg(conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    gammas = goalie_gamma(gg, df["game_date"])

    df = df.merge(
        gammas.rename(columns={"goalie_id": "home_goalie", "gamma": "gamma_home"}),
        on=["game_date", "home_goalie"], how="left",
    )

    df = df.merge(
        gammas.rename(columns={"goalie_id": "away_goalie", "gamma": "gamma_away"}),
        on=["game_date", "away_goalie"], how="left",
    )

    df[["gamma_home", "gamma_away"]] = df[["gamma_home", "gamma_away"]].fillna(0.0)

    # 3b. Rest: days since each team's previous game, back-to-back flags (from the schedule)
    df = df.merge(team_rest(conn), on="game_id", how="left")
    df[["b2b_home", "b2b_away"]] = df[["b2b_home", "b2b_away"]].fillna(0).astype(int)

    # 3c. Relocated franchises count as one team (after the goalie / starter merges,
    #     which use the abbreviation of the season)
    df[["home", "away"]] = df[["home", "away"]].replace(FRANCHISE)

    # 4. Team indices 0..n-1 from the sorted team list
    teams = sorted(set(df["home"]) | set(df["away"]))

    idx = {team: i for i, team in enumerate(teams)}

    df["home_idx"] = df["home"].map(idx)
    df["away_idx"] = df["away"].map(idx)
    # 5. Sort by date, reset index, return (df, teams)
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    cols = ["game_id", "game_date", "season", "home", "away", "home_idx", "away_idx",
            "home_goals", "away_goals", "home_goals_reg", "away_goals_reg",
            "home_xg_reg", "away_xg_reg", "home_goalie", "away_goalie",
            "gamma_home", "gamma_away", "rest_home", "rest_away", "b2b_home", "b2b_away"]
    return df[cols], teams

def select_window(gt: pd.DataFrame, start, end) -> tuple[pd.DataFrame, list[str]]:
    """Games with start <= game_date < end, with team indices renumbered for that window.

    Used by the Phase 7 walk-forward backtest: build the full game table once,
    then cut a training window for each refit. Returns (window_gt, teams).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    w = gt[(gt["game_date"] >= start) & (gt["game_date"] < end)].copy()
    teams = sorted(set(w["home"]) | set(w["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    w["home_idx"] = w["home"].map(idx)
    w["away_idx"] = w["away"].map(idx)
    return w.reset_index(drop=True), teams