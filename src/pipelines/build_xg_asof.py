"""
build_xg_asof.py
================
v2.1 step 2: out-of-sample ("as-of") versions of the xG layer for the game model.

Home: src/pipelines/build_xg_asof.py

Usage (from the project root, after score_xg):
    py -m src.pipelines.build_xg_asof              # add the weeks not built yet (weekly use)
    py -m src.pipelines.build_xg_asof --rebuild    # everything from scratch (first time: about an hour)

The layer (shooter finishing + goalie skill, xg_layer.py) is refit every --step-days days,
each time on every shot BEFORE that date only. Shots in the following period are scored with
that fit, and each goalie's rating as of that date is saved. So no number in these tables
ever used the game it describes: the same rule shot_xg follows for LightGBM.
Track B learns from goals as well as xG; an in-sample shooter adjustment would let a game's
own goals leak into its xG and count twice.

Tables:
    shot_xg_layer(game_id, event_id, xg_shooter, fit_date, xg_base)
        xg_shooter = LightGBM xG adjusted for the shooter's finishing (logit + beta). Only the
        shooter: the goalie enters the game model once, through its goals equation.
    goalie_layer_asof(fit_date, goalie_id, gamma_logit, sd, shots, gamma)
        goalie skill from all shots before fit_date; gamma = gamma_logit * (1 - league goal
        rate per shot), the log-rate scale Track B's goalie offsets use (positive = better).
    xg_layer_fits(fit_date, n_shots, sigma_beta, sigma_gamma, seconds)

Shots before the first fit date (the first loaded season: nothing earlier to learn from) keep
their LightGBM xG and goalies get no layer rating there. Each run rebuilds the latest period
(new shots arrive in it), any new ones, and any period with a shot missing from shot_xg_layer
or whose LightGBM xG changed since (xg_base: e.g. after score_xg rescored a season).
"""

import argparse
import time

import numpy as np
import pandas as pd

from config.settings import SEASONS
from src.data.db import get_connection
from src.models.stochastic import xg_layer as xl

TERMS = ["beta", "gamma"]
STEP_DAYS = 7

DDL = [
    """CREATE TABLE IF NOT EXISTS shot_xg_layer (
           game_id INTEGER, event_id INTEGER, xg_shooter REAL, fit_date TEXT, xg_base REAL,
           PRIMARY KEY (game_id, event_id))""",
    """CREATE TABLE IF NOT EXISTS goalie_layer_asof (
           fit_date TEXT, goalie_id INTEGER, gamma_logit REAL, sd REAL, shots INTEGER, gamma REAL,
           PRIMARY KEY (fit_date, goalie_id))""",
    """CREATE TABLE IF NOT EXISTS xg_layer_fits (
           fit_date TEXT PRIMARY KEY, n_shots INTEGER, sigma_beta REAL, sigma_gamma REAL, seconds REAL)""",
]


def fit_dates(shots: pd.DataFrame, step_days: int) -> list[pd.Timestamp]:
    """Refit dates: every step_days from the start of the second loaded season, keeping only
    dates whose period [d, d + step) contains shots (skips off-seasons)."""
    seasons = sorted(shots["season"].unique())
    if len(seasons) < 2:
        return []
    first = shots.loc[shots["season"] == seasons[1], "game_date"].min().normalize()
    last = shots["game_date"].max().normalize()
    grid = pd.date_range(first, last, freq=f"{step_days}D")
    has = pd.Series(1, index=np.sort(shots["game_date"].dt.normalize().unique()))
    keep = [d for d in grid if has.loc[d:d + pd.Timedelta(days=step_days - 1)].any()]
    return keep


def main():
    parser = argparse.ArgumentParser(description="Build the as-of xG layer tables for the game model.")
    parser.add_argument("--rebuild", action="store_true", help="drop the tables and rebuild everything")
    parser.add_argument("--step-days", type=int, default=STEP_DAYS, help="days between refits (default 7)")
    args = parser.parse_args()

    conn = get_connection()
    if args.rebuild:
        for t in ("shot_xg_layer", "goalie_layer_asof", "xg_layer_fits"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    for ddl in DDL:
        conn.execute(ddl)
    conn.commit()

    have = {str(s) for (s,) in conn.execute("SELECT DISTINCT season FROM shot_xg")}
    seasons = [s for s in SEASONS if s in have]
    shots = xl.load_shots(conn, seasons)
    dates = fit_dates(shots, args.step_days)
    if not dates:
        raise SystemExit("Need at least two loaded seasons of shots (the first one is history only).")
    done = {pd.Timestamp(d) for (d,) in conn.execute("SELECT fit_date FROM xg_layer_fits")}
    todo = [d for d in dates if d not in done]
    if dates[-1] in done:
        todo.append(dates[-1])                       # the latest period: new shots may have arrived
    # periods with shots missing from shot_xg_layer, or whose LightGBM xG changed since
    old = pd.read_sql("SELECT game_id, event_id, xg_base FROM shot_xg_layer", conn)
    chk = shots[["game_id", "event_id", "xg", "game_date"]].merge(old, on=["game_id", "event_id"], how="left")
    bad = chk[chk["xg_base"].isna() | (np.abs(chk["xg"] - chk["xg_base"]) > 1e-9)]
    fits = np.array(dates, dtype="datetime64[ns]")
    if len(bad):
        k = np.searchsorted(fits, bad["game_date"].to_numpy(), side="right") - 1
        todo += [dates[i] for i in np.unique(k[k >= 0])]
    todo = sorted(set(todo))
    print(f"{len(shots):,} shots, {len(dates)} refit dates of {args.step_days} days "
          f"({dates[0].date()} to {dates[-1].date()}); {len(todo)} to (re)build")

    # Shots before the first fit date: LightGBM xG unchanged (nothing earlier to learn from)
    early = shots[shots["game_date"] < dates[0]]
    conn.executemany("INSERT OR REPLACE INTO shot_xg_layer VALUES (?,?,?,?,?)",
                     [(int(g), int(e), float(x), None, float(x)) for g, e, x in zip(early.game_id, early.event_id, early.xg)])
    conn.commit()

    rate = shots["goal"].mean()
    sigma0 = None
    prev = conn.execute("SELECT sigma_beta, sigma_gamma FROM xg_layer_fits ORDER BY fit_date DESC LIMIT 1").fetchone()
    if prev and prev[0]:
        sigma0 = {"beta": prev[0], "gamma": prev[1]}
    t_all = time.time()
    for i, d in enumerate(todo):
        nxt = next((x for x in dates if x > d), None)
        train = shots[shots["game_date"] < d]
        period = shots[(shots["game_date"] >= d) & ((shots["game_date"] < nxt) if nxt is not None else True)]
        t0 = time.time()
        fit = xl.fit_laplace(train, terms=TERMS, verbose=False, sigma0=sigma0, intervals=False)
        sigma0 = fit["sigma"]
        eta = xl.adjusted_logit(period, fit["effects"], fit["levels"], terms=["beta"])
        xg_s = 1 / (1 + np.exp(-eta))
        fd = str(d.date())
        conn.executemany("INSERT OR REPLACE INTO shot_xg_layer VALUES (?,?,?,?,?)",
                         [(int(g), int(e), float(x), fd, float(b))
                          for g, e, x, b in zip(period.game_id, period.event_id, xg_s, period.xg)])
        g = fit["effects"]["gamma"]
        g = g[g["level"] >= 0]
        conn.execute("DELETE FROM goalie_layer_asof WHERE fit_date = ?", (fd,))
        conn.executemany("INSERT INTO goalie_layer_asof VALUES (?,?,?,?,?,?)",
                         [(fd, int(r.level), float(r.mean), float(r.sd), int(r.shots), float(r.mean * (1 - rate)))
                          for r in g.itertuples()])
        sec = time.time() - t0
        conn.execute("INSERT OR REPLACE INTO xg_layer_fits VALUES (?,?,?,?,?)",
                     (fd, len(train), fit["sigma"]["beta"], fit["sigma"]["gamma"], sec))
        conn.commit()
        if i % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t_all
            print(f"  {fd}: trained on {len(train):,} shots, scored {len(period):,}; sigma_beta "
                  f"{fit['sigma']['beta']:.3f}, sigma_gamma {fit['sigma']['gamma']:.3f}; {sec:.0f}s "
                  f"[{i + 1}/{len(todo)}, {el / 60:.0f} min so far]")

    missing = conn.execute("""SELECT COUNT(*) FROM shot_xg x LEFT JOIN shot_xg_layer l
                              ON l.game_id = x.game_id AND l.event_id = x.event_id
                              WHERE l.game_id IS NULL AND x.xg IS NOT NULL""").fetchone()[0]
    n = conn.execute("SELECT COUNT(*) FROM shot_xg_layer").fetchone()[0]
    print(f"shot_xg_layer: {n:,} shots" + (f"; {missing:,} scored shots still missing" if missing else " (complete)"))
    conn.close()


if __name__ == "__main__":
    main()