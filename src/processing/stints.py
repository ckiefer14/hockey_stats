"""
stints.py
=========
v2.2: cut each game into stints, stretches of play in which nobody on the ice changes.

Home: src/processing/stints.py

A stint (s, e] runs between two consecutive line changes (any change by either team ends
it). For each one: duration, the skaters and goalie of each side, strength, score at its
start, faceoff zone it started from (if it began with a faceoff), and each side's unblocked
shots, xG (shot_xg; no empty-net shots) and goals.

Conventions:
  - A shift (start, end) covers stint (s, e] if start <= s and end >= e.
  - A shot or goal at second t belongs to the stint with s < t <= e: the players whose shift
    ends at the goal (the usual stoppage) are on the ice for it.
  - A faceoff at second t sets the zone start of the stint beginning at t (zone from the
    home team's point of view: O = home offensive zone).
  - Score at a stint's start counts goals at or before s (shootout excluded).
  - Duplicate shift rows (same player, period, start, end: the feed sometimes repeats a
    whole game) and pseudo-shifts without duration (goal markers) are dropped first.

Quality: a game is skipped (status + reason in stint_games) if, after de-duplication, either
team's skater shift time is off the boxscore TOI by more than TOI_TOL, the skater count is
outside 3-6 for more than BAD_COUNT_TOL of the time, a team has no goalie for more than
NO_GOALIE_TOL of the time (pulled goalies are about 1-2%), or shots fall outside any stint.
"""

import numpy as np
import pandas as pd

TOI_TOL = 0.03
BAD_COUNT_TOL = 0.01
NO_GOALIE_TOL = 0.06
UNASSIGNED_TOL = 0.02

STINT_COLS = ["game_id", "stint", "period", "start", "end", "duration", "home_skaters", "away_skaters",
              "home_goalie", "away_goalie", "n_home", "n_away", "strength", "score_diff", "zone_start",
              "home_shots", "away_shots", "home_xg", "away_xg", "home_goals", "away_goals"]


def mmss(s: pd.Series) -> pd.Series:
    parts = s.fillna("").astype(str).str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=s.index)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


# ---------------------------------------------------------------------------
# Loading (a batch of games at once)
# ---------------------------------------------------------------------------

def load_batch(conn, game_ids: list[int]) -> dict:
    """Everything stint building needs for these games, as DataFrames keyed by game_id."""
    ids = ",".join(str(int(g)) for g in game_ids)
    sched = pd.read_sql(f"""SELECT game_id, home_team_id, away_team_id FROM schedule
                            WHERE game_id IN ({ids})""", conn)
    sh = pd.read_sql(f"""SELECT game_id, player_id, team_id, team_abbrev, period, start_time, end_time, duration
                         FROM shifts WHERE game_id IN ({ids})""", conn)
    gk = pd.read_sql(f"SELECT game_id, player_id FROM goalie_game_stats WHERE game_id IN ({ids})", conn)
    box = pd.read_sql(f"SELECT game_id, toi, team_abbreviation FROM skater_game_stats WHERE game_id IN ({ids})", conn)
    # boxscore teams come as abbreviations: map them to team ids with the shift chart's own pairs
    pairs = sh.drop_duplicates(["game_id", "team_abbrev"])[["game_id", "team_abbrev", "team_id"]]
    box = box.merge(pairs.rename(columns={"team_abbrev": "team_abbreviation"}), on=["game_id", "team_abbreviation"],
                    how="left")
    ev = pd.read_sql(f"""SELECT game_id, event_id, period, period_type, game_seconds, event_type, team_id, zone_code
                         FROM game_events WHERE game_id IN ({ids})
                         AND event_type IN ('goal', 'faceoff', 'shot-on-goal', 'missed-shot')""", conn)
    xg = pd.read_sql(f"SELECT game_id, event_id, xg FROM shot_xg WHERE game_id IN ({ids})", conn)
    return {"sched": sched, "shifts": sh, "goalies": gk, "box": box, "events": ev, "xg": xg}


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def clean_shifts(sh: pd.DataFrame, goalie_ids: set) -> pd.DataFrame:
    sh = sh.copy()
    sh["s"] = mmss(sh["start_time"]) + (sh["period"] - 1) * 1200
    sh["e"] = mmss(sh["end_time"]) + (sh["period"] - 1) * 1200
    sh = sh[sh["duration"].notna() & sh["s"].notna() & sh["e"].notna() & (sh["e"] > sh["s"])]
    sh = sh.drop_duplicates(["player_id", "period", "s", "e"])
    sh["is_goalie"] = sh["player_id"].isin(goalie_ids)
    return sh


def build_game(game_id: int, data: dict) -> tuple[pd.DataFrame, dict]:
    """Stints of one game and a quality record {status, reason, ...}."""
    q = {"game_id": game_id, "status": "ok", "reason": None, "n_stints": 0, "toi_off": np.nan,
         "bad_count_share": np.nan, "no_goalie_share": np.nan, "unassigned_shots": 0}
    sc = data["sched"][data["sched"]["game_id"] == game_id]
    if sc.empty:
        return _skip(q, "not in schedule")
    home, away = int(sc["home_team_id"].iloc[0]), int(sc["away_team_id"].iloc[0])
    goalie_ids = set(data["goalies"].loc[data["goalies"]["game_id"] == game_id, "player_id"])
    sh = clean_shifts(data["shifts"][data["shifts"]["game_id"] == game_id], goalie_ids)
    if sh.empty:
        return _skip(q, "no shift chart")

    # TOI vs boxscore, per team (skaters)
    box = data["box"][data["box"]["game_id"] == game_id].copy()
    box["toi_s"] = mmss(box["toi"])
    sk = sh[~sh["is_goalie"]]
    off = []
    for team in (home, away):
        chart = (sk.loc[sk["team_id"] == team, "e"] - sk.loc[sk["team_id"] == team, "s"]).sum()
        b = box.loc[box["team_id"] == team, "toi_s"].sum()
        off.append(abs(chart - b) / b if b and b > 0 else np.nan)
    q["toi_off"] = float(np.nanmax(off)) if not all(np.isnan(off)) else np.nan

    rows = []
    for period, ps in sh.groupby("period"):
        b = np.unique(np.concatenate([ps["s"].to_numpy(), ps["e"].to_numpy()]))
        if len(b) < 2:
            continue
        lo, hi = b[:-1], b[1:]
        s_, e_ = ps["s"].to_numpy(), ps["e"].to_numpy()
        cover = (s_[:, None] <= lo[None, :]) & (e_[:, None] >= hi[None, :])
        pid, tid, gl = ps["player_id"].to_numpy(), ps["team_id"].to_numpy(), ps["is_goalie"].to_numpy()
        prev = None                                    # never merge across a period boundary
        for k in range(len(lo)):
            on = cover[:, k]
            hs = tuple(sorted(pid[on & (tid == home) & ~gl]))
            as_ = tuple(sorted(pid[on & (tid == away) & ~gl]))
            hg = pid[on & (tid == home) & gl]
            ag = pid[on & (tid == away) & gl]
            sig = (hs, as_, int(hg[0]) if len(hg) else None, int(ag[0]) if len(ag) else None)
            if prev is not None and sig == prev[0] and rows[-1]["end"] == lo[k]:
                rows[-1]["end"] = float(hi[k])                     # same players: extend the stint
            else:
                rows.append({"game_id": game_id, "period": int(period), "start": float(lo[k]), "end": float(hi[k]),
                             "home_skaters": ",".join(map(str, hs)), "away_skaters": ",".join(map(str, as_)),
                             "home_goalie": sig[2], "away_goalie": sig[3], "n_home": len(hs), "n_away": len(as_)})
            prev = (sig,)
    if not rows:
        return _skip(q, "no stints")
    st = pd.DataFrame(rows)
    st["duration"] = st["end"] - st["start"]
    total = st["duration"].sum()
    bad = ~st["n_home"].between(3, 6) | ~st["n_away"].between(3, 6)
    q["bad_count_share"] = float(st.loc[bad, "duration"].sum() / total)
    nog = (st["home_goalie"].isna() * st["duration"]).sum() + (st["away_goalie"].isna() * st["duration"]).sum()
    q["no_goalie_share"] = float(nog / (2 * total))
    st["strength"] = st["n_home"].astype(str) + "v" + st["n_away"].astype(str)

    # Events
    ev = data["events"][(data["events"]["game_id"] == game_id) & (data["events"]["period_type"] != "SO")]
    starts, ends = st["start"].to_numpy(), st["end"].to_numpy()

    def stint_of(t):                       # s < t <= e
        i = np.searchsorted(starts, t, side="left") - 1
        ok = (i >= 0) & (t <= ends[np.clip(i, 0, None)])
        return np.where(ok, i, -1)

    goals = ev[ev["event_type"] == "goal"]
    gh = np.sort(goals.loc[goals["team_id"] == home, "game_seconds"].to_numpy())
    ga = np.sort(goals.loc[goals["team_id"] == away, "game_seconds"].to_numpy())
    st["score_diff"] = (np.searchsorted(gh, starts, side="right") - np.searchsorted(ga, starts, side="right"))

    fo = ev[ev["event_type"] == "faceoff"]
    st["zone_start"] = None
    if len(fo):
        i = np.searchsorted(starts, fo["game_seconds"].to_numpy(), side="left")
        hit = (i < len(starts)) & (starts[np.clip(i, 0, len(starts) - 1)] == fo["game_seconds"].to_numpy())
        z = fo["zone_code"].to_numpy()
        z_home = np.where(fo["team_id"].to_numpy() == home, z, np.select([z == "O", z == "D"], ["D", "O"], z))
        for k, zz in zip(i[hit], z_home[hit]):
            st.iat[k, st.columns.get_loc("zone_start")] = zz

    shots = ev[ev["event_type"].isin(["goal", "shot-on-goal", "missed-shot"])]
    shots = shots.merge(data["xg"][data["xg"]["game_id"] == game_id], on=["game_id", "event_id"], how="left")
    k = stint_of(shots["game_seconds"].to_numpy())
    q["unassigned_shots"] = int((k < 0).sum())
    for side, team in (("home", home), ("away", away)):
        m = (shots["team_id"].to_numpy() == team) & (k >= 0)
        st[f"{side}_shots"] = np.bincount(k[m], minlength=len(st))
        st[f"{side}_xg"] = np.bincount(k[m], weights=shots["xg"].fillna(0).to_numpy()[m], minlength=len(st))
        g = m & (shots["event_type"].to_numpy() == "goal")
        st[f"{side}_goals"] = np.bincount(k[g], minlength=len(st))
    st["stint"] = np.arange(len(st))
    q["n_stints"] = len(st)

    reasons = []
    if not np.isnan(q["toi_off"]) and q["toi_off"] > TOI_TOL:
        reasons.append(f"TOI off {q['toi_off']:.0%}")
    if q["bad_count_share"] > BAD_COUNT_TOL:
        reasons.append(f"skater count outside 3-6 for {q['bad_count_share']:.0%} of the time")
    if q["no_goalie_share"] > NO_GOALIE_TOL:
        reasons.append(f"no goalie for {q['no_goalie_share']:.0%} of the time")
    if len(shots) and q["unassigned_shots"] / len(shots) > UNASSIGNED_TOL:
        reasons.append(f"{q['unassigned_shots']} shots outside any stint")
    if reasons:
        q["status"], q["reason"], q["n_stints"] = "skipped", "; ".join(reasons), 0
        return st.iloc[0:0][STINT_COLS], q
    return st[STINT_COLS], q


def _skip(q, reason):
    q["status"], q["reason"] = "skipped", reason
    return pd.DataFrame(columns=STINT_COLS), q
