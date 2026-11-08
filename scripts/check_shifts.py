"""
check_shifts.py
===============
v2.2 step 0: is the shift data good enough to build stints (who was on the ice when)?

Home: scripts/check_shifts.py

Usage (from the project root):
    py .\\scripts\\check_shifts.py                 # every season, on-ice check on 150 games per season
    py .\\scripts\\check_shifts.py --sample 0      # on-ice check on every game (slower)

Per regular season:
  games         loaded games (games table)
  with shifts   games with at least one real shift row
  goal rows     rows with no duration (the shift-chart feed lists goals as pseudo-shifts;
                stint building drops them)
  TOI match     team-games whose skater shift time is within 2% of the boxscore TOI
                (missing or partial shift charts show up here)
  dressed match team-games where the skaters in the shift chart are exactly the boxscore skaters
  on-ice ok     share of sampled seconds (periods 1-3) where each team has 3-6 skaters on the
                ice per the shift chart (gaps or overlapping shifts break this)
  goalie gaps   share of sampled seconds with no goalie on the ice for a team (pulled goalie
                is real but rare: about 1-2% of time)
Then the worst games, so they can be reloaded or skipped.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.db import get_connection

TOI_TOL = 0.02


def mmss(s: pd.Series) -> pd.Series:
    """'MM:SS' -> seconds (NaN for empty)."""
    parts = s.fillna("").astype(str).str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=s.index)
    m = pd.to_numeric(parts[0], errors="coerce")
    sec = pd.to_numeric(parts[1], errors="coerce")
    return m * 60 + sec


def load(conn):
    games = pd.read_sql("SELECT id AS game_id, season, home_team_abbrev AS home, away_team_abbrev AS away "
                        "FROM games WHERE game_type = 2", conn)
    sh = pd.read_sql("SELECT game_id, player_id, team_abbrev AS team, period, start_time, end_time, duration "
                     "FROM shifts", conn)
    sk = pd.read_sql("SELECT game_id, player_id, team_abbreviation AS team, position, toi FROM skater_game_stats", conn)
    gk = pd.read_sql("SELECT game_id, player_id, team_abbreviation AS team FROM goalie_game_stats", conn)
    sh = sh[sh["game_id"].isin(games["game_id"])]
    sh["start"] = mmss(sh["start_time"]) + (sh["period"] - 1) * 1200
    sh["end"] = mmss(sh["end_time"]) + (sh["period"] - 1) * 1200
    sh["dur"] = mmss(sh["duration"])
    sh["goal_row"] = sh["dur"].isna() | (sh["end"] <= sh["start"])
    sk["toi_s"] = mmss(sk["toi"])
    gk_ids = set(zip(gk["game_id"], gk["player_id"]))
    sh["is_goalie"] = [(g, p) in gk_ids for g, p in zip(sh["game_id"], sh["player_id"])]
    return games, sh, sk


def team_checks(games, sh, sk) -> pd.DataFrame:
    real = sh[~sh["goal_row"]]
    sk_toi = sk.groupby(["game_id", "team"])["toi_s"].sum().rename("box_toi")
    sh_toi = real[~real["is_goalie"]].groupby(["game_id", "team"])["dur"].sum().rename("shift_toi")
    t = pd.concat([sk_toi, sh_toi], axis=1).reset_index()
    t["shift_toi"] = t["shift_toi"].fillna(0)
    t["toi_ok"] = (t["box_toi"] > 0) & ((t["shift_toi"] - t["box_toi"]).abs() <= TOI_TOL * t["box_toi"])
    box_players = sk.groupby(["game_id", "team"])["player_id"].apply(frozenset).rename("box")
    sh_players = real[~real["is_goalie"]].groupby(["game_id", "team"])["player_id"].apply(frozenset).rename("chart")
    pm = pd.concat([box_players, sh_players], axis=1).reset_index()
    pm["dressed_ok"] = [isinstance(a, frozenset) and isinstance(b, frozenset) and a == b
                        for a, b in zip(pm["box"], pm["chart"])]
    t = t.merge(pm[["game_id", "team", "dressed_ok"]], on=["game_id", "team"], how="left")
    return t.merge(games[["game_id", "season"]], on="game_id")


def on_ice(sh_game: pd.DataFrame, step: int = 5) -> dict:
    """Share of sampled seconds in periods 1-3 with 3-6 skaters per team, and with no goalie."""
    real = sh_game[~sh_game["goal_row"]]
    t = np.arange(1, 3600, step)
    ok = gap = n = 0
    for team, g in real.groupby("team"):
        s, e = g["start"].to_numpy(), g["end"].to_numpy()
        on = (s[:, None] <= t[None, :]) & (t[None, :] < e[:, None])          # shift x second
        sk = on[~g["is_goalie"].to_numpy()].sum(axis=0)
        gl = on[g["is_goalie"].to_numpy()].sum(axis=0)
        ok += int(((sk >= 3) & (sk <= 6)).sum())
        gap += int((gl == 0).sum())
        n += len(t)
    return {"on_ice_ok": ok / n if n else np.nan, "goalie_gap": gap / n if n else np.nan}


def main():
    parser = argparse.ArgumentParser(description="Check shift-chart data quality for v2.2 stints.")
    parser.add_argument("--sample", type=int, default=150, help="games per season for the on-ice check (0 = all)")
    args = parser.parse_args()
    conn = get_connection()
    games, sh, sk = load(conn)
    conn.close()
    tc = team_checks(games, sh, sk)

    rng = np.random.default_rng(0)
    oi = []
    has_shifts = set(sh.loc[~sh["goal_row"], "game_id"])
    for season, g in games[games["game_id"].isin(has_shifts)].groupby("season"):
        ids = g["game_id"].to_numpy()
        if args.sample and len(ids) > args.sample:
            ids = rng.choice(ids, args.sample, replace=False)
        for gid in ids:
            oi.append({"game_id": gid, "season": season, **on_ice(sh[sh["game_id"] == gid])})
    oi = pd.DataFrame(oi)

    rows = []
    for season, g in games.groupby("season"):
        s = sh[sh["game_id"].isin(g["game_id"])]
        t = tc[tc["season"] == season]
        o = oi[oi["season"] == season] if len(oi) else pd.DataFrame()
        rows.append({
            "season": season, "games": len(g),
            "with shifts": int(s.loc[~s["goal_row"], "game_id"].nunique()),
            "goal rows": int(s["goal_row"].sum()),
            "TOI match": f"{t['toi_ok'].mean():.1%}" if len(t) else "-",
            "dressed match": f"{t['dressed_ok'].mean():.1%}" if len(t) else "-",
            "on-ice ok": f"{o['on_ice_ok'].mean():.1%}" if len(o) else "-",
            "goalie gaps": f"{o['goalie_gap'].mean():.1%}" if len(o) else "-",
            "on-ice games checked": len(o),
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))

    bad = tc[~tc["toi_ok"]].copy()
    bad["off by"] = (bad["shift_toi"] - bad["box_toi"]) / bad["box_toi"]
    print(f"\nTeam-games whose shift time misses the boxscore TOI by more than {TOI_TOL:.0%}: {len(bad):,} "
          f"of {len(tc):,}")
    if len(bad):
        no_chart = int((bad["shift_toi"] == 0).sum())
        print(f"  with no shift chart at all: {no_chart:,}; partial or off: {len(bad) - no_chart:,}")
        part = bad[bad["shift_toi"] > 0]
        show = part.loc[part["off by"].abs().sort_values(ascending=False).index].head(10)
        if len(show):
            print("  worst partial ones:")
            print(show[["game_id", "season", "team", "box_toi", "shift_toi", "off by"]].round(3).to_string(index=False))
    if len(oi):
        w = oi.sort_values("on_ice_ok").head(5)
        print("\nLowest on-ice ok (sampled games):")
        print(w.round(3).to_string(index=False))
    print("\nRead: stints need 'with shifts', 'TOI match' and 'on-ice ok' near 100%. A few percent bad games can be "
          "skipped; a whole season missing or a low on-ice share means reloading shifts first.")


if __name__ == "__main__":
    main()
