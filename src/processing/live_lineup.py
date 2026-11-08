"""
live_lineup.py
==============
v2.2: tonight's assumed lineup for each team, and the lineup shift (lin) the game model uses.

Home: src/processing/live_lineup.py

Assumed lineup (assume_lineup):
  - If the NHL game-day roster already lists the dressed lineup (18 skaters or fewer for the
    team), use it as is ("posted").
  - Otherwise start from the team's last dressed lineup ("assumed") and:
      1. drop players not on today's NHL roster (injured reserve, sent down, traded) and
         players ESPN lists as Out (or Injured Reserve, when the NHL roster is unavailable);
         day-to-day players stay in, noted (dtd_out=True drops them too);
      2. bring back regulars who are on today's roster, not injured, and dressed for at least
         RETURN_GAMES of the team's previous USAGE_GAMES games;
      3. apply --out / --in overrides;
      4. balance to 12 forwards and 6 defensemen: drop the least-used extras, fill open spots
         from today's roster by usage, or with an unrated "call-up" (the fit's 'other player'
         rating) if nobody is available.
Shift (lineup_shift): the same calculation as lineup_context.py for past games: expected-TOI
weights, strength vs the team's previous BASELINE_GAMES lineups with the same as-of ratings,
lin = (offense change - opponent's defense change) * share_5v5 / base_xg60.
"""

import unicodedata

import numpy as np
import pandas as pd

from src.processing.lineup_context import BASELINE_GAMES, DEFAULT_TOI, LOOKBACK_TOI, strengths
from src.processing.stints import mmss

USAGE_GAMES = 20
RETURN_GAMES = 10
TARGET = {"F": 12, "D": 6}
CALL_UP = -1                               # placeholder id: unrated, gets the 'other player' rating


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def group(pos: str) -> str:
    return "D" if pos == "D" else "F"


# ---------------------------------------------------------------------------
# History the assumption needs (one read for the whole run)
# ---------------------------------------------------------------------------

def history(conn) -> pd.DataFrame:
    """Every dressed skater: game_id, game_date, team, player_id, name, position, toi_s."""
    sk = pd.read_sql("""
        SELECT k.game_id, s.game_date, k.team_abbreviation AS team, k.player_id, k.name, k.position, k.toi
        FROM skater_game_stats k JOIN schedule s ON s.game_id = k.game_id
        JOIN games g ON g.id = k.game_id WHERE g.game_type = 2
    """, conn, parse_dates=["game_date"])
    sk["toi_s"] = mmss(sk["toi"])
    return sk.drop(columns="toi")


def team_games_before(hist: pd.DataFrame, team: str, day) -> list[int]:
    g = hist[(hist["team"] == team) & (hist["game_date"] < pd.Timestamp(day))]
    return g.drop_duplicates("game_id").sort_values(["game_date", "game_id"])["game_id"].tolist()


# ---------------------------------------------------------------------------
# The assumed lineup
# ---------------------------------------------------------------------------

def _match(names: pd.DataFrame, first: str, last: str):
    """player_id in `names` (columns player_id, first, last) for a first/last name, or None."""
    m = names[(names["last_n"] == norm(last)) & (names["first_n"] == norm(first))]
    if len(m) == 1:
        return int(m["player_id"].iloc[0])
    m = names[names["last_n"] == norm(last)]
    if len(m) == 1:
        return int(m["player_id"].iloc[0])
    m = names[(names["last_n"] == norm(last)) & (names["first_n"].str[:1] == norm(first)[:1])]
    return int(m["player_id"].iloc[0]) if len(m) == 1 else None


def _names_from_history(hist: pd.DataFrame, ids) -> pd.DataFrame:
    """Names for player ids from the box scores ('L. Draisaitl' style)."""
    h = hist[hist["player_id"].isin(ids)].drop_duplicates("player_id", keep="last")
    if h.empty:
        return pd.DataFrame(columns=["player_id", "first_n", "last_n"])
    parts = h["name"].fillna("").str.split(" ", n=1, expand=True)
    first = parts[0].str.rstrip(".") if parts.shape[1] else ""
    last = parts[1] if parts.shape[1] > 1 else ""
    return pd.DataFrame({"player_id": h["player_id"].to_numpy(), "first_n": first.map(norm).to_numpy(),
                         "last_n": pd.Series(last).map(norm).to_numpy() if len(h) else []})


def assume_lineup(team: str, day, hist: pd.DataFrame, spots: pd.DataFrame | None, injuries: pd.DataFrame | None,
                  outs: list[str] = (), ins: list[str] = (), dtd_out: bool = False) -> dict:
    """Assumed skaters for `team` on `day`. Returns {players, positions, notes, source, missing}."""
    notes = []
    tg = team_games_before(hist, team, day)
    usage_ids = tg[-USAGE_GAMES:]
    use = hist[hist["game_id"].isin(usage_ids) & (hist["team"] == team)]
    games_played = use.groupby("player_id").size()
    minutes = use.groupby("player_id")["toi_s"].mean()
    pos_of = hist.drop_duplicates("player_id", keep="last").set_index("player_id")["position"]

    sp = None
    if spots is not None and len(spots):
        sp = spots[(spots["team"] == team) & (spots["pos"] != "G")].copy()
        sp["first_n"], sp["last_n"] = sp["first"].map(norm), sp["last"].map(norm)
        for _, r in sp.iterrows():
            pos_of.loc[r["player_id"]] = r["pos"]
        if 0 < len(sp) <= 18:
            players = sp["player_id"].astype(int).tolist()
            return {"players": players, "positions": [pos_of.get(p, "C") for p in players],
                    "notes": ["NHL posted the dressed lineup"], "source": "posted", "missing": []}

    if not tg:
        return {"players": [], "positions": [], "notes": ["no earlier game"], "source": "none", "missing": []}
    lineup = hist[(hist["game_id"] == tg[-1]) & (hist["team"] == team)]["player_id"].astype(int).tolist()
    source = "assumed"

    # 1. who is unavailable
    inj = injuries[injuries["team"] == team] if injuries is not None and len(injuries) else pd.DataFrame(
        columns=["first", "last", "name", "status"])
    lookup = sp if sp is not None and len(sp) else _names_from_history(hist, set(lineup) | set(games_played.index))
    unavailable = {}
    for _, r in inj.iterrows():
        st = r["status"]
        if st == "Day-To-Day" and not dtd_out:
            continue
        pid = _match(lookup, r["first"], r["last"])
        if pid is not None:
            unavailable[pid] = f"{r['name']} ({st})"
    if sp is not None and len(sp):
        on_roster = set(sp["player_id"].astype(int))
        for p in lineup:
            if p not in on_roster and p not in unavailable:
                unavailable[p] = f"{_label(hist, p)} (not on today's roster)"
    for p in [p for p in lineup if p in unavailable]:
        lineup.remove(p)
        notes.append(f"out: {unavailable[p]}")
    dtd = [r for _, r in inj.iterrows() if r["status"] == "Day-To-Day"]
    for r in dtd:
        pid = _match(lookup, r["first"], r["last"])
        if pid in lineup:
            notes.append(f"day-to-day, assumed in: {r['name']}")

    # 2. regulars coming back
    if sp is not None and len(sp):
        for p in sp["player_id"].astype(int):
            if p not in lineup and p not in unavailable and games_played.get(p, 0) >= RETURN_GAMES:
                lineup.append(p)
                notes.append(f"back: {_label(hist, p)} ({games_played.get(p, 0)} of the last {len(usage_ids)} games)")

    # 3. overrides
    for who in outs:
        pid = _find(hist, lookup, lineup, who)
        if pid in lineup:
            lineup.remove(pid)
            notes.append(f"out (given): {_label(hist, pid)}")
        else:
            notes.append(f"--out {who}: not in the assumed lineup")
    forced = set()
    for who in ins:
        cand = sp if sp is not None and len(sp) else _names_from_history(hist, set(hist.loc[hist["team"] == team, "player_id"]))
        pid = _find(hist, cand, list(cand["player_id"].astype(int)), who)
        if pid is not None and pid not in lineup:
            lineup.append(pid)
            forced.add(pid)
            notes.append(f"in (given): {_label(hist, pid)}")
        elif pid is None:
            notes.append(f"--in {who}: no such player on {team}")

    # 4. balance to 12 F / 6 D
    missing = []
    for grp, target in TARGET.items():
        mine = [p for p in lineup if group(pos_of.get(p, "C")) == grp]
        extra = len(mine) - target
        if extra > 0:
            drop = sorted([p for p in mine if p not in forced], key=lambda p: (games_played.get(p, 0), minutes.get(p, 0)))[:extra]
            for p in drop:
                lineup.remove(p)
                notes.append(f"assumed scratched: {_label(hist, p)}")
        while len([p for p in lineup if group(pos_of.get(p, "C")) == grp]) < target:
            pool = []
            if sp is not None and len(sp):
                pool = [int(p) for p, ps in zip(sp["player_id"], sp["pos"])
                        if group(ps) == grp and int(p) not in lineup and int(p) not in unavailable]
            if pool:
                best = max(pool, key=lambda p: (games_played.get(p, 0), minutes.get(p, 0)))
                lineup.append(best)
                notes.append(f"fills in: {_label(hist, best)}")
            else:
                lineup.append(CALL_UP)
                pos_of.loc[CALL_UP] = "C" if grp == "F" else "D"
                missing.append(grp)
                notes.append(f"unknown call-up ({grp})")
    return {"players": lineup, "positions": [pos_of.get(p, "C") for p in lineup], "notes": notes,
            "source": source, "missing": missing}


def _label(hist, pid) -> str:
    if pid == CALL_UP:
        return "call-up"
    n = hist.loc[hist["player_id"] == pid, "name"]
    return n.iloc[-1] if len(n) else str(pid)


def _find(hist, names: pd.DataFrame, pool: list[int], who: str):
    """Player id for a --out / --in name (last name or part of it, or an id) among `pool`."""
    if str(who).isdigit():
        return int(who)
    w = norm(who)
    if "last_n" in names:
        m = names[names["last_n"].str.contains(w, regex=False) & names["player_id"].isin(pool)]
        if len(m) == 1:
            return int(m["player_id"].iloc[0])
    h = hist[hist["player_id"].isin(pool)].drop_duplicates("player_id")
    m = h[h["name"].map(norm).str.contains(w, regex=False)]
    return int(m["player_id"].iloc[0]) if len(m) == 1 else None


# ---------------------------------------------------------------------------
# The shift
# ---------------------------------------------------------------------------

def expected_toi(hist: pd.DataFrame, players: list[int], day) -> np.ndarray:
    """Average TOI of each player's previous LOOKBACK_TOI games before `day` (call-ups: lineup median)."""
    h = hist[(hist["player_id"].isin([p for p in players if p != CALL_UP])) & (hist["game_date"] < pd.Timestamp(day))]
    h = h.sort_values("game_date").groupby("player_id").tail(LOOKBACK_TOI)
    t = h.groupby("player_id")["toi_s"].mean()
    vals = np.array([t.get(p, np.nan) for p in players], float)
    med = np.nanmedian(vals) if np.isfinite(vals).any() else DEFAULT_TOI
    return np.where(np.isnan(vals), med, vals)


def lineup_strength(players, t, ratings, fit):
    off = ratings.set_index("player_id")["off"]
    dfn = ratings.set_index("player_id")["def"]
    f = 5 * t / t.sum()
    so, sd, nu = strengths(np.array(players), f, off, dfn, fit["other_off"], fit["other_def"])
    return so, sd


def baseline(hist, team, day, ratings, fit):
    """Mean strength of the team's previous BASELINE_GAMES lineups, rated with today's ratings."""
    tg = team_games_before(hist, team, day)[-BASELINE_GAMES:]
    so, sd = [], []
    for gid in tg:
        g = hist[(hist["game_id"] == gid) & (hist["team"] == team)]
        t = expected_toi(hist, g["player_id"].tolist(), g["game_date"].iloc[0])
        o, d = lineup_strength(g["player_id"].tolist(), t, ratings, fit)
        so.append(o); sd.append(d)
    return (float(np.mean(so)), float(np.mean(sd))) if so else (np.nan, np.nan)


def latest_fit(conn, day):
    fit = pd.read_sql("SELECT * FROM lineup_fits WHERE fit_date <= ? ORDER BY fit_date DESC LIMIT 1", conn,
                      params=(str(pd.Timestamp(day).date()),))
    if fit.empty:
        raise SystemExit("No player-rating fit on or before this date. Run: py -m src.pipelines.build_lineup_asof")
    fit = fit.iloc[0].to_dict()
    ratings = pd.read_sql("SELECT player_id, off, def FROM player_rating_asof WHERE fit_date = ?", conn,
                          params=(fit["fit_date"],))
    return fit, ratings


def lineup_shift(conn, hist, home, away, day, lineups: dict) -> dict:
    """lin_home / lin_away and the strength changes for assumed lineups {'home': ..., 'away': ...}."""
    fit, ratings = latest_fit(conn, day)
    out = {"lineup_fit": fit["fit_date"]}
    for side, team in (("home", home), ("away", away)):
        pl = lineups[side]["players"]
        if not pl:
            out[f"d_off_{side}"] = out[f"d_def_{side}"] = 0.0
            continue
        t = expected_toi(hist, pl, day)
        so, sd = lineup_strength(pl, t, ratings, fit)
        bo, bd = baseline(hist, team, day, ratings, fit)
        out[f"d_off_{side}"] = 0.0 if np.isnan(bo) else so - bo
        out[f"d_def_{side}"] = 0.0 if np.isnan(bd) else sd - bd
    scale = fit["share_5v5"] / fit["base_xg60"]
    out["lin_home"] = (out["d_off_home"] - out["d_def_away"]) * scale
    out["lin_away"] = (out["d_off_away"] - out["d_def_home"]) * scale
    return out
