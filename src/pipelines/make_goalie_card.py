"""
make_goalie_card.py
===================
Goalie cards (PNG), in the player card's layout: percentile bars with 90% ranges, usage tiles,
and the rating week by week.

Home: src/pipelines/make_goalie_card.py

Usage (from the project root; after load_all / score_xg, and fit_xg_layer --final weekly):
    py -m src.pipelines.make_goalie_card Swayman            # a name (or part of one) or a player id
    py -m src.pipelines.make_goalie_card --team BOS         # every goalie who started for BOS this season
    py -m src.pipelines.make_goalie_card --all              # every goalie who started this season (~1 min)

Writes output/goalies/<date>/<Name>_<id>.png (one folder per team with --team / --all).

Bars (percentile among regular goalies; dot = mean, whisker = where the ends of the 90% range rank):
1. Goals kept out vs expected: the Phase 3 goalie rating (goalie.py: Gamma-Poisson on GA vs xGA,
   2-season window, season-adjusted, pulled toward average). THIS is the rating the game model
   uses for the starter. Regulars: MIN_SHOTS shots faced in the window.
2. Shooter-adjusted skill: v2.1 xG layer goalie effect (xg_layer_<season>_gamma.csv, all loaded
   seasons, adjusts for who shot and the rink's recording). Display only.
3. This season only: 1 - GA / xGA this season (league-adjusted), no pull toward average, so noisy
   early on; range from Poisson noise. Regulars: MIN_SEASON_SHOTS shots this season.
"""

import argparse
import json
import re

import matplotlib
import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.stochastic.goalie import goalie_rating_as_of
from src.processing.goalie import goalie_game_xg

GOALIE_DIR = OUTPUT_DIR / "goalies"
POSTERIOR_DIR = OUTPUT_DIR / "posteriors"
MIN_SHOTS = 1500          # shots faced in the 2-season window to count as a regular
MIN_SEASON_SHOTS = 200    # shots this season for the this-season bar
STALE_DAYS = 10
Z90 = 1.645


def pct_rank(pool, x):
    pool = np.sort(np.asarray(pool, float))
    x = np.atleast_1d(np.asarray(x, float))
    return 100.0 * (np.searchsorted(pool, x, "left") + np.searchsorted(pool, x, "right")) / 2.0 / max(len(pool), 1)


def bar(label, text, mean, lo, hi, pool, note=""):
    p, plo, phi = pct_rank(pool, [mean, lo, hi])
    return {"label": label, "value_text": text, "pct": float(p), "pct_lo": float(plo), "pct_hi": float(phi),
            "note": note}


def _p(x, digits=1):
    return f"{100 * x:+.{digits}f}%".replace("-", "−")


def load_gamma(conn):
    files = sorted(POSTERIOR_DIR.glob("xg_layer_*_gamma.csv"))
    if not files:
        return None, None, "no shooter-adjusted fit yet (run fit_xg_layer --final)"
    full = [f for f in files if re.fullmatch(r"xg_layer_\d{8}_gamma\.csv", f.name)]
    f = (full or files)[-1]
    meta = f.with_name(f.name.replace("_gamma.csv", ".json"))
    fitted = json.load(open(meta)).get("fitted_at", "")[:10] if meta.exists() else ""
    eff = pd.read_csv(f).rename(columns={"level": "goalie_id"})
    eff["goalie_id"] = pd.to_numeric(eff["goalie_id"], errors="coerce")
    eff = eff.dropna(subset=["goalie_id"]).astype({"goalie_id": int})
    note = ""
    last = conn.execute("SELECT MAX(s.game_date) FROM shot_xg x JOIN schedule s ON s.game_id = x.game_id").fetchone()[0]
    if fitted and last and (pd.Timestamp(last) - pd.Timestamp(fitted)).days > STALE_DAYS:
        note = f"shooter-adjusted fit {fitted} is behind the shots (run fit_xg_layer --final)"
    return eff, fitted, note


def rating_history(gg: pd.DataFrame, end, days: int = 730, step: int = 7) -> pd.DataFrame:
    """Phase 3 ratings of every goalie, weekly over the last `days` days (in-season weeks only)."""
    dates = pd.to_datetime(gg["game_date"])
    out = []
    for d in pd.date_range(pd.Timestamp(end) - pd.Timedelta(days=days), end, freq=f"{step}D"):
        if not ((dates < d) & (dates >= d - pd.Timedelta(days=step))).any():
            continue                                           # off-season week
        r = goalie_rating_as_of(gg, d)
        out.append(r.assign(date=d))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def card_data(conn, gid: int, gg, cur, hist, gamma, gamma_date, gamma_note, season, starts, day,
              sv_counts: dict):
    from src.viz.player_card import PlayerCardData

    mine = gg[gg["goalie_id"] == gid].sort_values("game_date")
    name = mine["goalie_name"].iloc[-1] if len(mine) else str(gid)
    team = mine["team"].iloc[-1] if len(mine) else ""
    bars, notes, tiles = [], [], []

    # 1. Phase 3 rating (the game model's goalie input)
    reg = cur[cur["shots"] >= MIN_SHOTS]
    me = cur[cur["goalie_id"] == gid]
    if me.empty:
        bars.append({"label": "Goals kept out", "value_text": "", "pct": None, "note": "no games in the 2-season window"})
    else:
        r = me.iloc[0]
        e, elo, ehi = 1 - r["theta_mean"], 1 - r["theta_hi"], 1 - r["theta_lo"]
        n = f"{int(r['shots'])} shots in the window: pulled toward average" if r["shots"] < MIN_SHOTS else ""
        bars.append(bar("Goals kept out", f"{_p(e)} · GSAx {r['gsax']:+.1f}".replace("-", "−"),
                        e, elo, ehi, 1 - reg["theta_mean"], n))

    # 2. v2.1 shooter-adjusted skill
    if gamma is None:
        bars.append({"label": "Shooter-adjusted", "value_text": "", "pct": None, "note": gamma_note})
    else:
        g = gamma[gamma["goalie_id"] == gid]
        gpool = gamma[gamma["shots"] >= MIN_SHOTS]
        if g.empty:
            bars.append({"label": "Shooter-adjusted", "value_text": "", "pct": None, "note": "not in the fit"})
        else:
            m, sd = float(g["mean"].iloc[0]), float(g["sd"].iloc[0])
            f = lambda v: 1 - np.exp(-v)                          # fewer goals than expected, as a fraction
            n = f"{int(g['shots'].iloc[0])} shots: pulled toward average" if g["shots"].iloc[0] < MIN_SHOTS else ""
            bars.append(bar("Shooter-adjusted", f"{_p(f(m))} · shooters & rinks",
                            f(m), f(m - Z90 * sd), f(m + Z90 * sd), f(gpool["mean"]), n))
        if gamma_note:
            notes.append(gamma_note)

    # 3. this season only (no shrinkage)
    ss = gg[gg["season"].astype(str) == season]
    if len(ss):
        ratio = ss["ga"].sum() / ss["xga"].sum()                 # league GA / xGA this season -> 1.0
        tot = ss.groupby("goalie_id")[["shots", "xga", "ga"]].sum()
        tot["xga"] = tot["xga"] * ratio
        tot["e"] = 1 - tot["ga"] / tot["xga"]
        if gid in tot.index and tot.loc[gid, "xga"] > 0:
            t = tot.loc[gid]
            sd = 1 / np.sqrt(t["xga"])
            pool = tot[tot["shots"] >= MIN_SEASON_SHOTS]["e"]
            n = "few shots yet: very noisy" if t["shots"] < MIN_SEASON_SHOTS else ""
            bars.append(bar("This season only", f"{_p(t['e'])} · {int(t['ga'])} GA on {t['xga']:.1f} xGA",
                            t["e"], t["e"] - Z90 * sd, t["e"] + Z90 * sd, pool if len(pool) else tot["e"], n))
            sog = sv_counts.get(gid)
            st = starts[(starts["player_id"] == gid)]
            team_st = starts[starts["team_abbreviation"] == team]
            s_label = f"Starts {season[:4]}-{season[6:]}"
            tiles.append((s_label, str(len(st)), f"{len(st) / max(len(team_st), 1):.0%} of {team} starts"))
            if sog and sog[1]:
                tiles.append(("Save %", f"{1 - sog[0] / sog[1]:.3f}".lstrip("0"), f"{int(sog[1])} shots on goal"))
            tiles.append(("Goals saved above expected", f"{t['xga'] - t['ga']:+.1f}".replace("-", "−"), "this season"))
        else:
            bars.append({"label": "This season only", "value_text": "", "pct": None, "note": "no games this season"})
    if not me.empty:
        per_game = cur["xga"].sum() / max(cur["games"].sum(), 1)
        tiles.append(("Worth vs an average goalie", f"{(1 - me.iloc[0]['theta_mean']) * per_game:+.2f}".replace("-", "−"),
                      "goals per game (Phase 3)"))

    # trend: Phase 3 rating week by week, against the middle 80% of regulars
    path = None
    h = hist[hist["goalie_id"] == gid] if len(hist) else hist
    if len(h) >= 2:
        q = hist[hist["shots"] >= MIN_SHOTS].groupby("date")["theta_mean"].quantile([0.1, 0.9]).unstack()
        path = pd.DataFrame({"fit_date": h["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                             "net": (1 - h["theta_mean"]).to_numpy(), "net_lo": (1 - h["theta_hi"]).to_numpy(),
                             "net_hi": (1 - h["theta_lo"]).to_numpy(),
                             "grp_lo": (1 - h["date"].map(q[0.9])).to_numpy(),
                             "grp_hi": (1 - h["date"].map(q[0.1])).to_numpy()})

    label = f"Goals kept out: Phase 3 goalie model as of {day} (the game model's input) · {len(reg)} regular goalies " \
            f"({MIN_SHOTS:,}+ shots in 2 seasons)"
    if gamma_date:
        label += f" · shooter-adjusted: v2.1 xG layer ({gamma_date})"
    return PlayerCardData(name=name, player_id=gid, team=team, position="G", group_label="regular goalies",
                          as_of=str(day), bars=bars, tiles=tiles, path=path, notes=notes, model_label=label,
                          trend_title="Goals kept out over time",
                          trend_subtitle="% fewer goals than expected from the shots faced, weekly (2-season window)",
                          trend_pct=True, kind="GOALIE CARD")


def make_cards(queries: list[str], team: str | None = None, everyone: bool = False) -> list:
    import matplotlib.pyplot as plt
    from src.viz.player_card import plot_player_card

    conn = get_connection()
    gg = goalie_game_xg(conn)
    if gg.empty:
        raise SystemExit("No goalie games with xG. Run load_all and score_xg first.")
    day = (pd.to_datetime(gg["game_date"]).max() + pd.Timedelta(days=1)).date()
    season = str(gg["season"].max())
    starts = pd.read_sql("""SELECT g.player_id, g.team_abbreviation FROM goalie_game_stats g
                            JOIN schedule s ON s.game_id = g.game_id WHERE g.starter = 1 AND s.season = ?""",
                         conn, params=(int(season),))
    cur = goalie_rating_as_of(gg, day)
    sv = pd.read_sql("""SELECT se.goalie_id, SUM(se.event_type = 'goal') AS ga,
                               SUM(se.event_type IN ('goal', 'shot-on-goal')) AS sog
                        FROM shot_events se JOIN schedule s ON s.game_id = se.game_id
                        WHERE s.season = ? AND se.period <= 4 AND se.goalie_id IS NOT NULL
                        GROUP BY se.goalie_id""", conn, params=(int(season),))
    sv_counts = {int(r.goalie_id): (r.ga, r.sog) for r in sv.itertuples()}
    gamma, gamma_date, gamma_note = load_gamma(conn)
    names = gg.groupby("goalie_id")["goalie_name"].last()

    ids = []
    for q in queries:
        if q.isdigit():
            hit = [int(q)] if int(q) in names.index else []
        else:
            hit = names[names.str.casefold().str.contains(q.casefold(), regex=False)].index.astype(int).tolist()
        if not hit:
            print(f"No goalie matches '{q}'.")
        elif len(hit) > 1:
            print(f"'{q}' matches {len(hit)} goalies; use the id:\n"
                  + names.loc[hit].rename("name").reset_index().to_string(index=False))
        else:
            ids += hit
    if team:
        ids += starts[starts["team_abbreviation"] == team.upper()]["player_id"].value_counts().index.astype(int).tolist()
    if everyone:
        ids += starts["player_id"].value_counts().index.astype(int).tolist()
    ids = list(dict.fromkeys(ids))
    if not ids:
        conn.close()
        return []
    print(f"Weekly goalie ratings for the trend panels ({len(ids)} card{'s' if len(ids) != 1 else ''})...", flush=True)
    hist = rating_history(gg, day)
    by_team = bool(team or everyone)
    out_dir = GOALIE_DIR / str(day)
    saved, failed = [], []
    for pid in ids:
        try:
            d = card_data(conn, pid, gg, cur, hist, gamma, gamma_date, gamma_note, season, starts, day, sv_counts)
            fig = plot_player_card(d)
            folder = out_dir / (d.team or "unknown") if by_team else out_dir
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{re.sub(r'[^A-Za-z0-9]+', '_', d.name).strip('_')}_{pid}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            saved.append(path)
            print(f"Saved {path}")
        except Exception as e:                                   # noqa: BLE001
            plt.close("all")
            failed.append(pid)
            print(f"  FAILED for goalie {pid}: {type(e).__name__}: {e}")
    if failed:
        print(f"{len(failed)} failed: {failed}")
    conn.close()
    return saved


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Goalie cards.")
    parser.add_argument("goalies", nargs="*", help="names (or parts of names) or player ids")
    parser.add_argument("--team", help="every goalie who started for this team this season")
    parser.add_argument("--all", action="store_true", help="every goalie who started this season")
    args = parser.parse_args()
    if not args.goalies and not args.team and not args.all:
        parser.error("give goalie names / ids, --team or --all")
    make_cards(args.goalies, args.team, everyone=args.all)


if __name__ == "__main__":
    main()
