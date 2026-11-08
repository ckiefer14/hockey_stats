"""
build_lineup_asof.py
====================
v2.2: as-of player ratings and each game's lineup context, for the game model.

Home: src/pipelines/build_lineup_asof.py

Usage (from the project root, after build_stints):
    py -m src.pipelines.build_lineup_asof              # add fits / games not built yet
    py -m src.pipelines.build_lineup_asof --rebuild    # everything again (first run: about 30-60 min)

1. Every --step-days days from --start, the Bayesian RAPM (lineup_rapm.py, no team effects)
   is fit on the --days days of 5v5 stints BEFORE that date only:
       player_rating_asof(fit_date, player_id, off, def, off_sd, def_sd, toi_5v5)
       lineup_fits(fit_date, n_stints, sigma, tau_*, other_off, other_def, base_xg60, share_5v5, seconds)
2. Every game from the first fit date on, with a box score, gets its lineup context
   (lineup_context.py): lineup strength vs the team's usual lineup, both rated with the latest
   fit made on or before the game date:
       lineup_context(game_id, fit_date, s_*, d_*, lin_home, lin_away, n_unrated_*)
Nothing in either table used the game it describes (ratings come from earlier stints; the
expected TOI from earlier games; the dressed lineup is known before puck drop).
"""

import argparse
import time

import numpy as np
import pandas as pd

from config.settings import SEASONS
from src.data.db import get_connection
from src.models.deterministic import rapm
from src.models.stochastic import lineup_rapm as lr
from src.pipelines.fit_rapm import names
from src.processing import lineup_context as lc

STEP_DAYS = 14
WINDOW_DAYS = 730
MIN_TOI = 100.0

DDL = [
    """CREATE TABLE IF NOT EXISTS player_rating_asof (
           fit_date TEXT, player_id INTEGER, off REAL, def REAL, off_sd REAL, def_sd REAL, toi_5v5 REAL,
           PRIMARY KEY (fit_date, player_id))""",
    """CREATE TABLE IF NOT EXISTS lineup_fits (
           fit_date TEXT PRIMARY KEY, n_stints INTEGER, sigma REAL, tau_f_off REAL, tau_d_off REAL,
           tau_f_def REAL, tau_d_def REAL, other_off REAL, other_def REAL, base_xg60 REAL, share_5v5 REAL,
           seconds REAL)""",
    f"""CREATE TABLE IF NOT EXISTS lineup_context (
           game_id INTEGER PRIMARY KEY, fit_date TEXT, {", ".join(c + (" INTEGER" if c.startswith("n_") else " REAL")
                                                                 for c in lc.CONTEXT_COLS)})""",
]


def load_all_stints(conn, start, end) -> pd.DataFrame:
    return rapm.load_stints(conn, start, end)


def share_5v5(conn, start, end) -> float:
    r = conn.execute("""SELECT SUM(CASE WHEN s.strength = '5v5' THEN s.home_xg + s.away_xg ELSE 0 END),
                               SUM(s.home_xg + s.away_xg)
                        FROM stints s JOIN schedule sc ON sc.game_id = s.game_id
                        WHERE sc.game_date >= ? AND sc.game_date < ?""",
                     (str(start.date()), str(end.date()))).fetchone()
    return float(r[0] / r[1]) if r and r[1] else 0.72


def main():
    parser = argparse.ArgumentParser(description="As-of player ratings and lineup context for the game model.")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--start", default=None,
                        help="first fit date (default: start of the third-to-last season in settings)")
    parser.add_argument("--step-days", type=int, default=STEP_DAYS)
    parser.add_argument("--days", type=int, default=WINDOW_DAYS)
    args = parser.parse_args()

    conn = get_connection()
    if args.rebuild:
        for t in ("player_rating_asof", "lineup_fits", "lineup_context"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    for ddl in DDL:
        conn.execute(ddl)
    conn.commit()

    if args.start:
        start = pd.Timestamp(args.start)
    else:
        s0 = SEASONS[-3] if len(SEASONS) >= 3 else SEASONS[0]
        r = conn.execute("SELECT MIN(game_date) FROM schedule WHERE season = ?", (int(s0),)).fetchone()[0]
        start = pd.Timestamp(r) if r else pd.Timestamp("2024-10-01")
    last = conn.execute("SELECT MAX(sc.game_date) FROM stints s JOIN schedule sc ON sc.game_id = s.game_id").fetchone()[0]
    last = pd.Timestamp(last) + pd.Timedelta(days=1)
    grid = list(pd.date_range(start, last, freq=f"{args.step_days}D"))
    if not grid or grid[-1] < last:
        grid.append(last)                                # the newest fit uses everything loaded
    done = {pd.Timestamp(d) for (d,) in conn.execute("SELECT fit_date FROM lineup_fits")}
    todo = [d for d in grid if d not in done]
    print(f"Fits every {args.step_days} days from {start.date()} to {last.date()}: {len(grid)} dates, "
          f"{len(todo)} to build", flush=True)

    # 1. Rating fits
    if todo:
        who = names(conn)
        pos = who.set_index("player_id")["position"]
        allst = load_all_stints(conn, min(todo) - pd.Timedelta(days=args.days), max(todo))
        print(f"  loaded {len(allst):,} 5v5 stints", flush=True)
        th0 = None
        t_all = time.time()
        for i, d in enumerate(todo):
            t0 = time.time()
            st = allst[(allst["game_date"] >= d - pd.Timedelta(days=args.days)) & (allst["game_date"] < d)]
            if st["game_id"].nunique() < 200:
                print(f"  {d.date()}: only {st['game_id'].nunique()} games of stints before it, skipped")
                continue
            X, y, w, games, info = lr.design(st, pos, min_toi=MIN_TOI, teams=False)
            A, c, yy, n, slw = lr.gram_parts(X, y, w)
            th = lr.fit_spreads(A, c, yy, n, slw, info, x0=th0)
            th0 = th
            m, cf, _ = lr.posterior(A, c, th, info["groups"])
            from scipy.linalg import cho_solve
            P = info["P"]
            # sds of the player effects only (diagonal of H^-1 for those columns)
            idx = np.r_[np.arange(P), np.arange(P + 1, 2 * P + 1)]
            E = np.zeros((len(m), len(idx)))
            E[idx, np.arange(len(idx))] = 1.0
            var = np.einsum("ij,ij->j", E, cho_solve(cf, E))
            fd = str(d.date())
            rows = [(fd, int(p), float(m[k]), float(-m[P + 1 + k]), float(np.sqrt(var[k])), float(np.sqrt(var[P + k])),
                     float(info["minutes"][k])) for k, p in enumerate(info["players"])]
            conn.execute("DELETE FROM player_rating_asof WHERE fit_date = ?", (fd,))
            conn.executemany("INSERT INTO player_rating_asof VALUES (?,?,?,?,?,?,?)", rows)
            names_ = info["names"]
            base = float(st["home_xg"].sum() + st["away_xg"].sum()) / float(st["duration"].sum()) * 1800.0
            conn.execute("INSERT OR REPLACE INTO lineup_fits VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (fd, len(st), th["sigma"], th["F_off"], th["D_off"], th["F_def"], th["D_def"],
                          float(m[names_.index("O_other")]), float(-m[names_.index("D_other")]), base,
                          share_5v5(conn, d - pd.Timedelta(days=args.days), d), time.time() - t0))
            conn.commit()
            el = time.time() - t_all
            print(f"  {fd}: {len(st):,} stints, {P} players; tau F/D off {th['F_off']:.3f}/{th['D_off']:.3f}, "
                  f"def {th['F_def']:.3f}/{th['D_def']:.3f}; {time.time() - t0:.0f}s "
                  f"[{i + 1}/{len(todo)}, about {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left]", flush=True)

    # 2. Lineup context for every game from the first fit date on (rebuilt each run: cheap)
    fits = pd.read_sql("SELECT * FROM lineup_fits ORDER BY fit_date", conn, parse_dates=["fit_date"])
    if fits.empty:
        raise SystemExit("No rating fits yet.")
    games = pd.read_sql("""SELECT s.game_id, s.game_date, s.home_team_abbrev AS home, s.away_team_abbrev AS away
                           FROM schedule s JOIN games g ON g.id = s.game_id
                           WHERE g.game_type = 2 AND s.game_date >= ?""", conn,
                        params=(str(fits["fit_date"].min().date()),), parse_dates=["game_date"])
    dr = lc.dressed(conn)
    lineups = lc.team_lineups(dr)
    fd = fits["fit_date"].to_numpy()
    games["fit_date"] = fd[np.searchsorted(fd, games["game_date"].to_numpy(), side="right") - 1]
    out = []
    for f_date, g in games.groupby("fit_date"):
        fit = fits[fits["fit_date"] == f_date].iloc[0].to_dict()
        rt = pd.read_sql("SELECT player_id, off, def FROM player_rating_asof WHERE fit_date = ?", conn,
                         params=(str(pd.Timestamp(f_date).date()),))
        ctx = lc.context_for_period(g, lineups, rt, fit)
        ctx["fit_date"] = str(pd.Timestamp(f_date).date())
        out.append(ctx)
    ctx = pd.concat(out, ignore_index=True)
    conn.execute("DELETE FROM lineup_context")
    cols = ["game_id", "fit_date"] + lc.CONTEXT_COLS
    conn.executemany(f"INSERT INTO lineup_context VALUES ({','.join('?' * len(cols))})",
                     [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
                      for r in ctx[cols].itertuples(index=False)])
    conn.commit()
    conn.close()

    lin = pd.concat([ctx["lin_home"], ctx["lin_away"]])
    print(f"\nlineup_context: {len(ctx):,} games. Change vs the usual lineup, as a log xG-rate shift per team-game:")
    print(f"  sd {lin.std():.4f}; 5th / 95th percentile {lin.quantile(0.05):+.4f} / {lin.quantile(0.95):+.4f}; "
          f"min {lin.min():+.4f} (about {np.expm1(lin.min()):+.1%} xG)")
    worst = ctx.assign(lin=ctx[["lin_home", "lin_away"]].min(axis=1)).nsmallest(5, "lin")
    print("  most weakened lineups (game_id, home/away lin):")
    print(worst[["game_id", "lin_home", "lin_away", "n_unrated_home", "n_unrated_away"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()