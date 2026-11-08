"""
lineup_context.py
=================
v2.2: how much stronger or weaker each team's dressed lineup is than its usual one, from
as-of player ratings (player_rating_asof, built by build_lineup_asof.py).

Home: src/processing/lineup_context.py

For a team in a game:
    expected 5v5 share  f_p = 5 * t_p / sum(t)       t_p = the player's average TOI in his
                                                      previous LOOKBACK_TOI games (box score;
                                                      no history: the lineup's median)
    lineup strength      S_off = sum f_p * off_p,  S_def = sum f_p * def_p
                         (ratings as of the game date; unrated players get the fit's
                          'other player' rating, about a call-up's level)
    usual strength       mean of S over the team's previous BASELINE_GAMES lineups, rated with
                         the SAME as-of ratings (so only who dressed differs)
    change               d_off = S_off - usual, d_def = S_def - usual           (xG/60, 5v5)

Into the game model, on the log xG-rate scale:
    lin_home = (d_off_home - d_def_away) * share_5v5 / base_xg60
    lin_away = (d_off_away - d_def_home) * share_5v5 / base_xg60
share_5v5 = 5v5 share of all xG, base_xg60 = league 5v5 xG/60 per team (both from the fit
window). The game model multiplies these by a learned kappa (prior N(1, 0.5)).
A full-strength night gives about 0; a missing star gives a negative lin for his team.
"""

import numpy as np
import pandas as pd

from src.processing.stints import mmss

LOOKBACK_TOI = 10
BASELINE_GAMES = 20
DEFAULT_TOI = 15 * 60.0          # seconds, when a lineup has no TOI history at all

CONTEXT_COLS = ["s_off_home", "s_def_home", "s_off_away", "s_def_away", "d_off_home", "d_def_home",
                "d_off_away", "d_def_away", "lin_home", "lin_away", "n_unrated_home", "n_unrated_away"]


def dressed(conn) -> pd.DataFrame:
    """Every skater who dressed, with the game date, side and expected TOI from his previous
    LOOKBACK_TOI games (not the game itself). Columns: game_id, game_date, side, team, player_id, t_exp."""
    sk = pd.read_sql("""
        SELECT k.game_id, k.player_id, k.team_abbreviation AS team, k.toi, s.game_date,
               CASE WHEN k.team_abbreviation = s.home_team_abbrev THEN 'home'
                    WHEN k.team_abbreviation = s.away_team_abbrev THEN 'away' END AS side
        FROM skater_game_stats k JOIN schedule s ON s.game_id = k.game_id
        JOIN games g ON g.id = k.game_id
        WHERE g.game_type = 2
    """, conn, parse_dates=["game_date"])
    sk = sk.dropna(subset=["side"])
    sk["toi_s"] = mmss(sk["toi"])
    sk = sk.sort_values(["player_id", "game_date", "game_id"])
    sk["t_exp"] = (sk.groupby("player_id")["toi_s"]
                   .transform(lambda s: s.shift(1).rolling(LOOKBACK_TOI, min_periods=1).mean()))
    med = sk.groupby(["game_id", "side"])["t_exp"].transform("median")
    sk["t_exp"] = sk["t_exp"].fillna(med).fillna(DEFAULT_TOI)
    return sk[["game_id", "game_date", "side", "team", "player_id", "t_exp"]]


def team_lineups(dr: pd.DataFrame) -> dict:
    """{team: list of (game_id, game_date, players array, f array)} in date order."""
    out = {}
    for (team, gid), g in dr.groupby(["team", "game_id"]):
        f = 5 * g["t_exp"].to_numpy() / g["t_exp"].sum()
        out.setdefault(team, []).append((gid, g["game_date"].iloc[0], g["player_id"].to_numpy(), f))
    for team in out:
        out[team].sort(key=lambda r: (r[1], r[0]))
    return out


def strengths(players, f, off: pd.Series, dfn: pd.Series, other_off: float, other_def: float):
    o = off.reindex(players)
    d = dfn.reindex(players)
    unrated = int(o.isna().sum())
    return float(np.sum(f * o.fillna(other_off).to_numpy())), float(np.sum(f * d.fillna(other_def).to_numpy())), unrated


def context_for_period(games: pd.DataFrame, lineups: dict, ratings: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """Lineup context for the games (game_id, game_date, home, away) rated with one fit."""
    off = ratings.set_index("player_id")["off"]
    dfn = ratings.set_index("player_id")["def"]
    oo, od = fit["other_off"], fit["other_def"]
    pos = {team: {r[0]: i for i, r in enumerate(rows)} for team, rows in lineups.items()}
    scale = fit["share_5v5"] / fit["base_xg60"]
    out = []
    for g in games.itertuples():
        rec = {"game_id": g.game_id}
        ok = True
        for side, team in (("home", g.home), ("away", g.away)):
            rows = lineups.get(team)
            i = pos.get(team, {}).get(g.game_id)
            if rows is None or i is None:
                ok = False
                break
            _, _, pl, f = rows[i]
            so, sd, nu = strengths(pl, f, off, dfn, oo, od)
            prev = rows[max(0, i - BASELINE_GAMES):i]
            if prev:
                bo = np.mean([strengths(p, ff, off, dfn, oo, od)[0] for _, _, p, ff in prev])
                bd = np.mean([strengths(p, ff, off, dfn, oo, od)[1] for _, _, p, ff in prev])
            else:
                bo, bd = so, sd
            rec.update({f"s_off_{side}": so, f"s_def_{side}": sd, f"d_off_{side}": so - bo,
                        f"d_def_{side}": sd - bd, f"n_unrated_{side}": nu})
        if not ok:
            continue
        rec["lin_home"] = (rec["d_off_home"] - rec["d_def_away"]) * scale
        rec["lin_away"] = (rec["d_off_away"] - rec["d_def_home"]) * scale
        out.append(rec)
    return pd.DataFrame(out, columns=["game_id"] + CONTEXT_COLS)