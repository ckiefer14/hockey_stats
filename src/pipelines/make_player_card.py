"""
make_player_card.py
===================
Player cards (PNG): percentile bars with 90% ranges for 5v5 offense, defense, net impact and
finishing, plus usage and the net impact over time.

Home: src/pipelines/make_player_card.py

Usage (from the project root; after the weekly build_lineup_asof and fit_xg_layer --final):
    py -m src.pipelines.make_player_card McDavid            # a name (or part of one) or a player id
    py -m src.pipelines.make_player_card McDavid Makar 8478402
    py -m src.pipelines.make_player_card --team EDM         # everyone who dressed in EDM's last 10 games
    py -m src.pipelines.make_player_card --all              # every rated skater who played this season
                                                              (~800 cards, about 3-6 min)

Writes output/players/<rating date>/<Name>_<id>.png; with --team / --all, one folder per team:
output/players/<rating date>/<TEAM>/<Name>_<id>.png.

Sources:
- player_rating_asof (build_lineup_asof.py): 5v5 offense / defense in xG per 60 vs an average
  player, with posterior sds, every 14 days. Percentiles compare the player with the regulars at
  his position (forwards or defensemen with MIN_TOI 5v5 minutes in the 2-season window). The
  90% range: where the ends of his rating's 90% interval would rank. Net sd treats offense and
  defense as independent (the table keeps no covariance), slightly wide if they are correlated.
- Finishing: xg_layer_<season>_beta.csv from `fit_xg_layer --final` (all loaded seasons).
  Shooters with MIN_SHOTS shots at his position. If the file is older than the newest scored
  shots by more than STALE_DAYS, the card says so.
"""

import argparse
import json
import re

import matplotlib
import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.pipelines.fit_rapm import names

PLAYER_DIR = OUTPUT_DIR / "players"
POSTERIOR_DIR = OUTPUT_DIR / "posteriors"
MIN_TOI = 300.0          # 5v5 minutes in the 2-season window to count as a regular
MIN_SHOTS = 150          # shots (all loaded seasons) to count in the finishing comparison
STALE_DAYS = 10
Z90 = 1.645


def pct_rank(pool: np.ndarray, x) -> np.ndarray:
    """Percentile of x within pool (0-100, ties count half)."""
    pool = np.sort(np.asarray(pool, float))
    x = np.atleast_1d(np.asarray(x, float))
    lo = np.searchsorted(pool, x, side="left")
    hi = np.searchsorted(pool, x, side="right")
    return 100.0 * (lo + hi) / 2.0 / max(len(pool), 1)


def bar(label, value_text, mean, sd, pool, note=""):
    p, lo, hi = pct_rank(pool, [mean, mean - Z90 * sd, mean + Z90 * sd])
    return {"label": label, "value_text": value_text, "pct": float(p), "pct_lo": float(lo), "pct_hi": float(hi),
            "note": note}


def _signed(x, digits=2):
    return f"{x:+.{digits}f}".replace("-", "−")


def load_ratings(conn) -> pd.DataFrame:
    rt = pd.read_sql("SELECT * FROM player_rating_asof", conn)
    if rt.empty:
        raise SystemExit("player_rating_asof is empty. Run: py -m src.pipelines.build_lineup_asof")
    rt["net"] = rt["off"] + rt["def"]
    rt["net_sd"] = np.sqrt(rt["off_sd"] ** 2 + rt["def_sd"] ** 2)
    return rt


def load_finishing(conn):
    """(effects DataFrame level/mean/sd/shots/goals/xg, fitted_at, note) from the newest xg_layer run."""
    files = sorted(POSTERIOR_DIR.glob("xg_layer_*_beta.csv"))
    if not files:
        return None, None, "no finishing fit yet (run fit_xg_layer --final)"
    full = [f for f in files if re.fullmatch(r"xg_layer_\d{8}_beta\.csv", f.name)]
    f = (full or files)[-1]
    meta_path = f.with_name(f.name.replace("_beta.csv", ".json"))
    fitted = None
    if meta_path.exists():
        with open(meta_path) as fh:
            fitted = json.load(fh).get("fitted_at")
    eff = pd.read_csv(f).rename(columns={"level": "player_id"})
    eff["player_id"] = pd.to_numeric(eff["player_id"], errors="coerce")
    eff = eff.dropna(subset=["player_id"]).astype({"player_id": int})
    note = ""
    last_shot = conn.execute("""SELECT MAX(s.game_date) FROM shot_xg x JOIN schedule s ON s.game_id = x.game_id""").fetchone()[0]
    if fitted and last_shot:
        age = (pd.Timestamp(last_shot) - pd.Timestamp(fitted[:10])).days
        if age > STALE_DAYS:
            note = f"finishing fit {fitted[:10]} is {age} days behind the shots (run fit_xg_layer --final)"
    return eff, (fitted or "")[:10], note


def find_players(who: pd.DataFrame, query: str, rated: set) -> list[int]:
    if query.isdigit():
        return [int(query)] if int(query) in set(who["player_id"]) else []
    q = query.casefold()
    hit = who[who["name"].str.casefold().str.contains(q, regex=False)]
    if len(hit) > 1:
        rated_hit = hit[hit["player_id"].isin(rated)]
        hit = rated_hit if len(rated_hit) else hit
    return hit["player_id"].astype(int).tolist()


def usage(conn, pid: int):
    g = pd.read_sql("""SELECT k.team_abbreviation AS team, k.toi, s.season, s.game_date FROM skater_game_stats k
                       JOIN schedule s ON s.game_id = k.game_id WHERE k.player_id = ? ORDER BY s.game_date""",
                    conn, params=(pid,))
    if g.empty:
        return None, None
    season = g["season"].max()
    cur = g[g["season"] == season]

    def mins(t):
        try:
            m, s = str(t).split(":")
            return int(m) + int(s) / 60
        except ValueError:
            return np.nan
    return g["team"].iloc[-1], {"season": str(season), "gp": len(cur), "toi": cur["toi"].map(mins).mean(),
                                "last": str(g["game_date"].iloc[-1])[:10]}


def team_players(conn, team: str, n_games: int = 10) -> list[int]:
    g = pd.read_sql("""SELECT k.player_id, k.toi FROM skater_game_stats k JOIN schedule s ON s.game_id = k.game_id
                       WHERE k.team_abbreviation = ? AND k.game_id IN (
                           SELECT s2.game_id FROM schedule s2 JOIN skater_game_stats k2 ON k2.game_id = s2.game_id
                           WHERE k2.team_abbreviation = ? GROUP BY s2.game_id ORDER BY MAX(s2.game_date) DESC LIMIT ?)""",
                    conn, params=(team, team, n_games))
    return g["player_id"].value_counts().index.astype(int).tolist()


def card_data(conn, pid: int, who: pd.DataFrame, rt: pd.DataFrame, fin, fin_date, fin_note):
    from src.viz.player_card import PlayerCardData

    w = who.set_index("player_id")
    name = str(w.loc[pid, "name"]) if pid in w.index else str(pid)
    pos = str(w.loc[pid, "position"]) if pid in w.index else "C"
    grp = "D" if pos == "D" else "F"
    grp_label = "defensemen" if grp == "D" else "forwards"
    gpos = who.set_index("player_id")["position"].map(lambda p: "D" if p == "D" else "F")

    latest = rt["fit_date"].max()
    cur = rt[rt["fit_date"] == latest].copy()
    cur["grp"] = cur["player_id"].map(gpos)
    pool = cur[(cur["grp"] == grp) & (cur["toi_5v5"] >= MIN_TOI)]
    me = cur[cur["player_id"] == pid]
    notes, bars = [], []
    if me.empty:
        bars = [{"label": k, "value_text": "", "pct": None, "note": "not rated (under 100 5v5 minutes in the window)"}
                for k in ("Offense", "Defense", "Net impact")]
    else:
        r = me.iloc[0]
        small = r["toi_5v5"] < MIN_TOI
        n = "small sample: pulled toward average" if small else ""
        bars.append(bar("Offense", f"{_signed(r['off'])} xG/60 for", r["off"], r["off_sd"], pool["off"], n))
        bars.append(bar("Defense", f"{_signed(r['def'])} xG/60 kept out", r["def"], r["def_sd"], pool["def"], n))
        bars.append(bar("Net impact", f"{_signed(r['net'])} xG/60 net", r["net"], r["net_sd"], pool["net"], n))

    shots_tile = None
    if fin is None:
        bars.append({"label": "Finishing", "value_text": "", "pct": None, "note": fin_note})
    else:
        f = fin.copy()
        f["grp"] = f["player_id"].map(gpos)
        fpool = f[(f["grp"] == grp) & (f["shots"] >= MIN_SHOTS)]
        mine = f[f["player_id"] == pid]
        if mine.empty:
            bars.append({"label": "Finishing", "value_text": "", "pct": None, "note": "no shots in the finishing fit"})
        else:
            m = mine.iloc[0]
            txt = f"{np.expm1(m['mean']):+.0%} goals vs expected".replace("-", "−")
            nn = f"{int(m['shots'])} shots (under {MIN_SHOTS}): pulled toward average" if m["shots"] < MIN_SHOTS else ""
            bars.append(bar("Finishing", txt, m["mean"], m["sd"], fpool["mean"], nn))
            shots_tile = ("Goals vs expected (finishing fit)", f"{int(m['goals'])} G on {m['xg']:.1f} xG",
                          f"{int(m['shots'])} shots")
        if fin_note:
            notes.append(fin_note)

    team, use = usage(conn, pid)
    tiles = []
    if use:
        tiles.append((f"Games {use['season'][:4]}-{use['season'][6:]}", str(use["gp"]), f"last {use['last']}"))
        tiles.append(("Time on ice per game", f"{int(use['toi'])}:{int(round((use['toi'] % 1) * 60)):02d}"
                      if pd.notna(use["toi"]) else "–", "all situations"))
    if not me.empty:
        tiles.append(("5v5 minutes in the rating window", f"{me.iloc[0]['toi_5v5']:,.0f}", "2 seasons"))
    if shots_tile:
        tiles.append(shots_tile)

    # net impact over time, with the middle 80% of regulars at his position
    hist = rt[rt["player_id"] == pid].sort_values("fit_date")
    path = None
    if len(hist) >= 2:
        r2 = rt.assign(grp=rt["player_id"].map(gpos))
        r2 = r2[(r2["grp"] == grp) & (r2["toi_5v5"] >= MIN_TOI)]
        q = r2.groupby("fit_date")["net"].quantile([0.1, 0.9]).unstack()
        path = hist[["fit_date", "net", "net_sd"]].assign(
            net_lo=lambda d: d["net"] - Z90 * d["net_sd"], net_hi=lambda d: d["net"] + Z90 * d["net_sd"],
            grp_lo=lambda d: d["fit_date"].map(q[0.1]), grp_hi=lambda d: d["fit_date"].map(q[0.9]))

    ratings_note = f"{len(pool)} regular {grp_label} ({MIN_TOI:.0f}+ 5v5 min)"
    label = f"Ratings: v2.2 5v5 player ratings ({latest}) · {ratings_note}"
    if fin_date:
        label += f" · finishing: v2.1 xG layer ({fin_date})"
    return PlayerCardData(name=name, player_id=pid, team=team or "", position=pos, group_label=grp_label,
                          as_of=str(latest), bars=bars, tiles=tiles, path=path, notes=notes, model_label=label)


def season_players(conn, rated: set) -> list[int]:
    """Rated skaters who played in the latest season, most games first."""
    g = pd.read_sql("""SELECT k.player_id, COUNT(*) AS gp FROM skater_game_stats k JOIN schedule s ON s.game_id = k.game_id
                       WHERE s.season = (SELECT MAX(season) FROM schedule s2 JOIN skater_game_stats k2
                                         ON k2.game_id = s2.game_id)
                       GROUP BY k.player_id ORDER BY gp DESC""", conn)
    return [int(p) for p in g["player_id"] if int(p) in rated]


def make_cards(queries: list[str], team: str | None = None, everyone: bool = False) -> list:
    import matplotlib.pyplot as plt
    from src.viz.player_card import plot_player_card

    conn = get_connection()
    who = names(conn)
    rt = load_ratings(conn)
    fin, fin_date, fin_note = load_finishing(conn)
    rated = set(rt.loc[rt["fit_date"] == rt["fit_date"].max(), "player_id"])
    ids = []
    for q in queries:
        hit = find_players(who, q, rated)
        if not hit:
            print(f"No player matches '{q}'.")
        elif len(hit) > 1:
            opts = who[who["player_id"].isin(hit)][["player_id", "name", "position"]]
            print(f"'{q}' matches {len(hit)} players; use the id:\n{opts.to_string(index=False)}")
        else:
            ids += hit
    by_team = bool(team or everyone)
    if team:
        ids += team_players(conn, team.upper())
    if everyone:
        ids += season_players(conn, rated)
    ids = list(dict.fromkeys(ids))
    out_dir = PLAYER_DIR / str(rt["fit_date"].max())
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, failed = [], []
    if len(ids) > 20:
        print(f"{len(ids)} cards to draw...", flush=True)
    for i, pid in enumerate(ids):
        try:
            d = card_data(conn, pid, who, rt, fin, fin_date, fin_note)
            fig = plot_player_card(d)
            safe = re.sub(r"[^A-Za-z0-9]+", "_", d.name).strip("_")
            folder = out_dir / (d.team or "unknown") if by_team else out_dir
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{safe}_{pid}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            saved.append(path)
        except Exception as e:                                   # noqa: BLE001  one bad player must not stop --all
            plt.close("all")
            failed.append(pid)
            print(f"  FAILED for player {pid}: {type(e).__name__}: {e}")
            continue
        if len(ids) <= 20:
            print(f"Saved {path}")
        elif (i + 1) % 50 == 0 or i + 1 == len(ids):
            print(f"  {i + 1}/{len(ids)} done", flush=True)
    if len(ids) > 20:
        print(f"Saved {len(saved)} cards to {out_dir}" + (f"; {len(failed)} failed: {failed}" if failed else ""))
    conn.close()
    return saved


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Player cards.")
    parser.add_argument("players", nargs="*", help="names (or parts of names) or player ids")
    parser.add_argument("--team", help="every skater who dressed in this team's last 10 games")
    parser.add_argument("--all", action="store_true", help="every rated skater who played this season")
    args = parser.parse_args()
    if not args.players and not args.team and not args.all:
        parser.error("give player names / ids, --team or --all")
    make_cards(args.players, args.team, everyone=args.all)


if __name__ == "__main__":
    main()