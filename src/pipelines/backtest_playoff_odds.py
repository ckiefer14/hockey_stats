"""
backtest_playoff_odds.py
========================
Walk-forward backtest of the playoff and Stanley Cup odds: were the past seasons' odds,
computed only from what was known on each date, well calibrated?

Home: src/pipelines/backtest_playoff_odds.py

Usage (from the project root):
    py -m src.pipelines.backtest_playoff_odds                      # 3 seasons x 5 dates (about 1-2 hours)
    py -m src.pipelines.backtest_playoff_odds --seasons 20242025 --dates 12-01 02-01
    py -m src.pipelines.backtest_playoff_odds --report             # only the report from saved results
    py -m src.pipelines.backtest_playoff_odds --tau-scales 1 1.25 1.5 2   # drift sweep from the saved fits
                                                                   # (no refitting; about 5-10 min per scale)
Stop any time: it resumes (one row block per season x date in the results file). Each date's fit
and inputs are saved to output/backtests/playoff_cache/ (about 1-3 MB each), so drift scales can
be compared later without refitting (--tau-scales). If results exist but fits were not saved yet
(runs before the sweep existed), a plain run refits only to save them.

For each season and checkpoint date (default Nov 15, Dec 15, Jan 15, Feb 15, Mar 15):
1. Fit the live model (v2.0 + rest + lineup) on the 730 days before the date, as the game backtest
   does (backtest.fit_window), and measure the OT / shootout parameters on those games only.
2. Standings as of that morning; every regular-season game from that date on is "remaining";
   goalie offsets from each team's starts before the date; rest shifts from the schedule.
3. Simulate the rest of the season and the playoffs (season_sim), as playoff_odds does.
4. Baseline, same everything but every team equally strong ("standings only"): shows what the
   ratings add over points already in the bank.
Truth: the real playoff teams and Cup winner from the NHL playoff-bracket feed; division winners
and final points from the final standings in the database. If the feed fails, playoff teams come
from the final standings (this project's tiebreakers) and the Cup is not scored.

Report: Brier score and log loss of P(playoffs) for the model and the baseline by checkpoint,
with the paired difference and a 90% range (bootstrap over team-seasons); calibration of
P(playoffs) by bin; how often final points fell inside the 90% projected range (should be about
90%); division winners; and the log score of the eventual Cup winner. Saved to
output/backtests/playoff_odds_backtest.csv.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.stochastic import season_sim as ss
from src.models.stochastic.hierarchical_tv import context_shift
from src.pipelines.playoff_odds import goalie_mix
from src.processing.game_table import build_game_table
from src.processing.goalie import goalie_game_xg
from src.processing.late_game import late_game_params
from src.processing.schedule_context import FRANCHISE, schedule_context
from src.validation.backtest import draws_from_idata, fit_window, with_new_teams

OUT = OUTPUT_DIR / "backtests" / "playoff_odds_backtest.csv"
CACHE_DIR = OUTPUT_DIR / "backtests" / "playoff_cache"           # prepared inputs per season x date (pickles)
TRUTH = OUTPUT_DIR / "backtests" / "playoff_odds_truth.csv"
BRACKET_URL = "https://api-web.nhle.com/v1/playoff-bracket/{year}"
DEFAULT_DATES = ["11-15", "12-15", "01-15", "02-15", "03-15"]
WINDOW = 730
EPS = 1e-4


# --------------------------------------------------------------------------
# truth
# --------------------------------------------------------------------------

def _walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def fetch_bracket(season: str):
    """(set of the 16 playoff teams, Cup winner) from the NHL feed, or (None, None)."""
    try:
        r = requests.get(BRACKET_URL.format(year=int(str(season)[4:])), timeout=20,
                         headers={"User-Agent": "hockey-stats"})
        r.raise_for_status()
        series = [s for s in _walk(r.json()) if "topSeedTeam" in s and "bottomSeedTeam" in s]
    except Exception as e:                                         # noqa: BLE001
        print(f"  playoff bracket feed unavailable for {season} ({type(e).__name__})")
        return None, None
    teams, winner = set(), None
    abbr = lambda t: FRANCHISE.get((t or {}).get("abbrev"), (t or {}).get("abbrev"))
    for s in series:
        top, bot = s.get("topSeedTeam") or {}, s.get("bottomSeedTeam") or {}
        if int(s.get("playoffRound", 0) or 0) == 1:
            teams |= {abbr(top), abbr(bot)}
        if int(s.get("playoffRound", 0) or 0) == 4 and s.get("winningTeamId"):
            winner = abbr(top) if top.get("id") == s["winningTeamId"] else abbr(bot)
    teams.discard(None)
    if len(teams) != 16:
        print(f"  playoff bracket feed for {season}: {len(teams)} round-1 teams found (expected 16); not used")
        return None, None
    return teams, winner


def final_truth(conn, season: str, info: pd.DataFrame) -> pd.DataFrame:
    """Final standings and the playoff / division outcomes computed from them."""
    st = ss.current_standings(conn, season, info, FRANCHISE)
    res = {k: st.set_index("team").reindex(info["team"])[k].to_numpy()[None, :].astype(float)
           for k in ("pts", "rw", "row", "w")}
    q = ss.qualify(res, info, np.random.default_rng(0))
    return pd.DataFrame({"team": info["team"].to_numpy(), "final_pts": res["pts"][0],
                         "made_calc": q["playoffs"][0], "div_win": q["div_win"][0]})


# --------------------------------------------------------------------------
# one checkpoint
# --------------------------------------------------------------------------

def prepare(conn, gt_all, gg, season: str, day: pd.Timestamp, info_all: pd.DataFrame, seed: int) -> dict:
    """Everything known on the morning of `day` that the simulation needs (the expensive part: one fit).
    Saved to CACHE_DIR so other drift scales can be tried without refitting."""
    sched = pd.read_sql("""SELECT game_id, season, game_date, home_team_abbrev AS home, away_team_abbrev AS away
                           FROM schedule WHERE season = ? AND game_type = 2 AND game_date >= ?""", conn,
                        params=(int(season), str(day.date())))
    sched[["home", "away"]] = sched[["home", "away"]].replace(FRANCHISE)
    teams_season = sorted(set(sched["home"]) | set(sched["away"]))
    info = info_all[info_all["team"].isin(teams_season)].sort_values(["conference", "division", "team"]) \
        .reset_index(drop=True)

    t0 = time.time()
    idata, w, teams = fit_window(gt_all, day, WINDOW, seed, "tv", rest=True, lineup=True)
    late = late_game_params(conn, w, idata.posterior)
    draws = draws_from_idata(idata)
    post = idata.posterior
    for k in ("tau_off", "tau_def"):
        if k in post:
            draws[k] = post[k].values.reshape(-1)
    draws, teams = with_new_teams(draws, teams, teams_season)
    fit_s = time.time() - t0

    standings = ss.current_standings(conn, season, info, FRANCHISE, before=day)
    ctx = sched[["game_id"]].merge(schedule_context(conn), on="game_id", how="left").fillna(0)
    for c in ("lin_home", "lin_away", "g2b_home", "g2b_away"):
        ctx[c] = 0.0
    sh, sa = context_shift(draws, ctx)
    sched["shift_home"], sched["shift_away"] = np.asarray(sh, float), np.asarray(sa, float)
    gamma = {t: goalie_mix(conn, gg, t, day.date()) for t in info["team"]}
    keep = {k: np.asarray(draws[k]) for k in ("mu_off", "mu_def", "alpha0", "alpha_home", "rho", "tau_off", "tau_def")
            if k in draws}
    return {"season": season, "day": day, "draws": keep, "teams": teams, "late": late, "sched": sched,
            "standings": standings, "info": info, "gamma": gamma, "fit_seconds": round(fit_s), "train_games": len(w)}


def simulate(inp: dict, sims: int, seed: int, tau_scale: float, baseline: bool = True) -> pd.DataFrame:
    """Playoff + Cup odds from prepared inputs: the model and (optionally) the standings-only baseline."""
    d0 = inp["draws"]
    runs = [("model", d0, inp["gamma"])]
    if baseline:                                                  # every team (and goalie) equal
        runs.append(("standings only", {**d0, "mu_off": np.zeros_like(d0["mu_off"]),
                                        "mu_def": np.zeros_like(d0["mu_def"])}, {}))
    out = []
    for label, d, gm in runs:
        res = ss.simulate_season(d, inp["teams"], inp["late"], inp["sched"], inp["standings"], inp["info"], gm,
                                 inp["day"], n_sims=sims, seed=seed, tau_scale=tau_scale)
        try:
            ss.simulate_playoffs(res, inp["info"], gm, inp["late"], np.random.default_rng(seed + 1))
        except ValueError as e:
            print(f"    Cup odds skipped: {e}")
        tab = ss.summarize(res, inp["info"], inp["standings"])
        tab.insert(0, "model", label)
        out.append(tab)
    tab = pd.concat(out, ignore_index=True)
    tab.insert(0, "date", str(inp["day"].date()))
    tab.insert(0, "season", inp["season"])
    tab["fit_seconds"], tab["train_games"], tab["tau_scale"] = inp["fit_seconds"], inp["train_games"], tau_scale
    return tab


def append_csv(tab: pd.DataFrame, path: Path) -> None:
    """Append rows; if the columns differ from the file's (older runs), rewrite it with both."""
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != tab.columns.tolist():
            old = pd.read_csv(path, dtype={"season": str, "date": str})
            pd.concat([old, tab], ignore_index=True).to_csv(path, index=False)
            return
    tab.to_csv(path, mode="a", index=False, header=not path.exists())


def cache_path(season: str, day) -> "Path":
    return CACHE_DIR / f"{season}_{pd.Timestamp(day).date()}.pkl"


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired(df, col="p_playoffs", n_boot=2000, seed=0):
    """Model minus baseline log loss, 90% range from resampling team-seasons."""
    m = df[df["model"] == "model"].set_index(["season", "date", "team"])
    b = df[df["model"] == "standings only"].set_index(["season", "date", "team"])
    j = m.join(b[[col]], rsuffix="_base", how="inner")
    y = j["made"].to_numpy(float)
    pm = np.clip(j[col].to_numpy(float), EPS, 1 - EPS)
    pb = np.clip(j[f"{col}_base"].to_numpy(float), EPS, 1 - EPS)
    d = -(y * np.log(pm) + (1 - y) * np.log(1 - pm)) + (y * np.log(pb) + (1 - y) * np.log(1 - pb))
    keys = j.reset_index()[["season", "team"]].astype(str).agg("|".join, axis=1).to_numpy()
    groups = pd.Series(d).groupby(keys)
    sums, counts = groups.sum().to_numpy(), groups.size().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (n_boot, len(sums)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    return float(d.mean()), float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


def _season_order(mmdd) -> list:
    """MM-DD labels in hockey-season order (August first)."""
    return sorted(set(mmdd), key=lambda x: ((int(x[:2]) - 8) % 12, x))


def report(df: pd.DataFrame):
    df = df.copy()
    order = _season_order(df["date"].str[5:])
    print(f"\nPlayoff odds backtest: {df['season'].nunique()} seasons, {df['date'].nunique()} dates, "
          f"{len(df[df['model'] == 'model']):,} team-dates\n")
    rows = []
    for (mm, md), g in df.groupby([df["date"].str[5:], "model"]):
        rows.append({"checkpoint": mm, "model": md, "Brier": brier(g["p_playoffs"], g["made"]),
                     "log loss": logloss(g["p_playoffs"], g["made"])})
    for md, g in df.groupby("model"):
        rows.append({"checkpoint": "all", "model": md, "Brier": brier(g["p_playoffs"], g["made"]),
                     "log loss": logloss(g["p_playoffs"], g["made"])})
    t = pd.DataFrame(rows).pivot(index="checkpoint", columns="model", values=["Brier", "log loss"]).round(4) \
        .reindex(order + ["all"])
    print("1. P(make playoffs): lower is better. 'standings only' = same simulation with every team equal.")
    print(t.to_string())
    d, lo, hi = paired(df)
    verdict = "ratings clearly help" if hi < 0 else "ratings clearly hurt" if lo > 0 else "within luck"
    print(f"   model minus standings-only log loss {d:+.4f} (90% {lo:+.4f} to {hi:+.4f}): {verdict}")

    m = df[df["model"] == "model"].copy()
    print("\n2. Calibration of P(make playoffs), model, all dates (about equal = calibrated):")
    bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
    m["bin"] = pd.cut(m["p_playoffs"], bins, right=False,
                      labels=["<5%", "5-20%", "20-40%", "40-60%", "60-80%", "80-95%", "95%+"])
    cal = m.groupby("bin", observed=True).agg(teams=("made", "size"), predicted=("p_playoffs", "mean"),
                                              made_it=("made", "mean"))
    print(cal.round(3).to_string())

    print("\n3. Final points vs the 90% projected range (should be inside about 90% of the time):")
    m["inside"] = (m["final_pts"] >= m["points_lo"]) & (m["final_pts"] <= m["points_hi"])
    m["abs_err"] = (m["final_pts"] - m["points_mean"]).abs()
    pts = m.groupby(m["date"].str[5:]).agg(inside=("inside", "mean"), mean_abs_error=("abs_err", "mean"),
                                           range_width=("points_hi", "mean"))
    pts["range_width"] = (m.groupby(m["date"].str[5:])["points_hi"].mean()
                          - m.groupby(m["date"].str[5:])["points_lo"].mean())
    print(pts.reindex(order).round(3).to_string())
    print(f"   all dates: inside {m['inside'].mean():.1%}; below the range {(m['final_pts'] < m['points_lo']).mean():.1%}, "
          f"above {(m['final_pts'] > m['points_hi']).mean():.1%}")

    print("\n4. Division winners (log loss of P(win division)):")
    dv = df.groupby("model").apply(lambda g: logloss(g["p_div_win"], g["div_win"]), include_groups=False)
    print(dv.round(4).to_string())

    if "p_cup" in df and df["cup"].notna().any():
        c = df[(df["cup"] == 1) & (df["model"] == "model")]
        print("\n5. Stanley Cup: the model's probability for the eventual champion on each date "
              "(every team equal: 1/32 = 3.1%):")
        if len(c):
            pv = c.pivot_table(index=["season", "team"], columns=c["date"].str[5:], values="p_cup", aggfunc="first")
            print(pv.reindex(columns=[o for o in order if o in pv.columns]).map(lambda v: f"{v:.1%}").to_string())
            for md, g in df[df["cup"] == 1].groupby("model"):
                print(f"   {md}: mean log score {np.log(np.clip(g['p_cup'], EPS, 1)).mean():.3f} "
                      f"(every team equal: {np.log(1 / 32):.3f}; higher is better)")
        print("   One champion per season: a sanity check, not a calibration test.")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def sweep_metrics(df: pd.DataFrame) -> dict:
    """Headline numbers of one drift scale (model rows only)."""
    m = df[df["model"] == "model"]
    inside = (m["final_pts"] >= m["points_lo"]) & (m["final_pts"] <= m["points_hi"])
    early = m["date"].str[5:7].isin(["10", "11", "12"])
    low = m["p_playoffs"] < 0.05
    high = m["p_playoffs"] >= 0.95
    champ = m[m["cup"] == 1]
    return {"log loss": logloss(m["p_playoffs"], m["made"]), "Brier": brier(m["p_playoffs"], m["made"]),
            "points inside 90%": inside.mean(), "inside Nov-Dec": inside[early].mean(),
            "<5%: predicted": m.loc[low, "p_playoffs"].mean(), "<5%: made": m.loc[low, "made"].mean(),
            "95%+: predicted": m.loc[high, "p_playoffs"].mean(), "95%+: made": m.loc[high, "made"].mean(),
            "division log loss": logloss(m["p_div_win"], m["div_win"]),
            "champion log score": float(np.log(np.clip(champ["p_cup"], EPS, 1)).mean()) if len(champ) else np.nan}


def run_sweep(scales, sims, seed, truth: pd.DataFrame):
    files = sorted(CACHE_DIR.glob("*.pkl"))
    if not files:
        raise SystemExit("No saved checkpoints: run the backtest once first (it saves them to "
                         f"{CACHE_DIR}).")
    print(f"Drift-scale sweep over {len(files)} saved checkpoints, {sims:,} seasons each:")
    rows, all_tabs = [], []
    for k in scales:
        t0 = time.time()
        tabs = []
        for f in files:
            with open(f, "rb") as fh:
                inp = pickle.load(fh)
            tab = simulate(inp, sims, seed, k, baseline=False)
            tabs.append(tab.merge(truth[truth["season"] == inp["season"]][["team", "final_pts", "made", "div_win", "cup"]],
                                  on="team", how="left"))
        df = pd.concat(tabs, ignore_index=True)
        all_tabs.append(df)
        rows.append({"drift x": k, **sweep_metrics(df)})
        print(f"  x{k:g} done in {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows).set_index("drift x")
    pd.concat(all_tabs, ignore_index=True).to_csv(OUTPUT_DIR / "backtests" / "playoff_odds_tau_sweep.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nlower log loss / Brier is better; 'inside' should be about 0.90; predicted should match made:")
    print(res.T.round(4).to_string())
    best = res["log loss"].idxmin()
    print(f"\nLowest log loss: drift x{best:g} (points inside the 90% range {res.loc[best, 'points inside 90%']:.1%}). "
          f"Differences under about 0.003 in log loss are within luck on 96 team-seasons; prefer the scale "
          f"whose coverage is closest to 90% among those.")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the playoff and Cup odds.")
    parser.add_argument("--seasons", nargs="+", default=None, help="default: the 3 latest complete seasons")
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES, help="checkpoints as MM-DD (default 5 dates)")
    parser.add_argument("--sims", type=int, default=5000, help="simulated seasons per checkpoint (default 5,000)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tau-scale", type=float, default=1.0,
                        help="drift scale for the main results (default 1.0 = the fit's own weekly step)")
    parser.add_argument("--tau-scales", type=float, nargs="+", default=None,
                        help="sweep these drift scales from the saved checkpoints (no refitting), e.g. 1 1.25 1.5 2")
    parser.add_argument("--report", action="store_true", help="only print the report from saved results")
    parser.add_argument("--restart", action="store_true", help="delete saved results (not the saved fits) and start over")
    args = parser.parse_args()

    if args.tau_scales:
        if not TRUTH.exists():
            raise SystemExit("No saved outcomes: run the backtest once first.")
        truth = pd.read_csv(TRUTH, dtype={"season": str})
        run_sweep(args.tau_scales, args.sims, args.seed, truth)
        return

    conn = get_connection()
    info_all = pd.read_sql("SELECT abbreviation AS team, division, conference FROM teams", conn)
    info_all["team"] = info_all["team"].replace(FRANCHISE)
    info_all = info_all.drop_duplicates("team")

    if args.restart and OUT.exists():
        OUT.unlink()
    if not args.report:
        if args.seasons is None:
            full = pd.read_sql("""SELECT s.season, COUNT(*) AS n, SUM(g.home_score IS NOT NULL) AS played
                                  FROM schedule s LEFT JOIN games g ON g.id = s.game_id
                                  WHERE s.game_type = 2 GROUP BY s.season""", conn)
            complete = sorted(str(s) for s, n, p in full.itertuples(index=False) if n >= 1000 and p == n)
            args.seasons = complete[-3:]
        print(f"Seasons: {', '.join(args.seasons)}; dates: {', '.join(args.dates)}")
        done = set()
        if OUT.exists():
            prev = pd.read_csv(OUT, usecols=["season", "date"]).astype(str)
            done = set(zip(prev["season"], prev["date"]))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gg = goalie_game_xg(conn)
        truths = []
        for season in args.seasons:
            truth = final_truth(conn, season, info_all[info_all["team"].isin(
                set(pd.read_sql("SELECT home_team_abbrev AS t FROM schedule WHERE season = ? AND game_type = 2",
                                conn, params=(int(season),))["t"].replace(FRANCHISE)))]
                .sort_values(["conference", "division", "team"]).reset_index(drop=True))
            teams16, champ = fetch_bracket(season)
            if teams16 is not None:
                calc = set(truth.loc[truth["made_calc"], "team"])
                if calc != teams16:
                    print(f"  {season}: computed playoff teams differ from the real ones "
                          f"(real only {sorted(teams16 - calc)}, computed only {sorted(calc - teams16)}): using the real ones")
                truth["made"] = truth["team"].isin(teams16)
            else:
                truth["made"] = truth["made_calc"]
            truth["cup"] = (truth["team"] == champ).astype(float) if champ else np.nan
            truth.insert(0, "season", season)
            truths.append(truth)
            gt_all = None
            y0 = int(season[:4])
            for md in args.dates:
                mm = int(md[:2])
                day = pd.Timestamp(f"{y0 if mm >= 8 else y0 + 1}-{md}")
                have_result = (season, str(day.date())) in done
                cp = cache_path(season, day)
                if have_result and cp.exists():
                    continue
                t0 = time.time()
                if cp.exists():
                    with open(cp, "rb") as fh:
                        inp = pickle.load(fh)
                else:
                    print(f"{season} as of {day.date()}: fitting...", flush=True)
                    if gt_all is None:
                        seasons_upto = sorted(s for s in pd.read_sql("SELECT DISTINCT season FROM schedule", conn)
                                              ["season"].astype(str) if s <= season)
                        gt_all, _ = build_game_table(conn, seasons_upto, lineup=True)
                    inp = prepare(conn, gt_all, gg, season, day, info_all, args.seed)
                    with open(cp, "wb") as fh:
                        pickle.dump(inp, fh)
                if have_result:
                    print(f"  saved the fit for {season} {day.date()} ({time.time() - t0:.0f}s); results already in the file")
                    continue
                tab = simulate(inp, args.sims, args.seed, args.tau_scale)
                tab = tab.merge(truth[["team", "final_pts", "made", "div_win", "cup"]], on="team", how="left")
                append_csv(tab, OUT)
                mm_ = tab[tab["model"] == "model"]
                print(f"  {season} {day.date()} done in {time.time() - t0:.0f}s (fit {int(mm_['fit_seconds'].iloc[0])}s): "
                      f"Brier {brier(mm_['p_playoffs'], mm_['made']):.3f}, "
                      f"standings only {brier(tab[tab['model'] != 'model']['p_playoffs'], mm_['made'].to_numpy()):.3f}",
                      flush=True)
        pd.concat(truths, ignore_index=True).to_csv(TRUTH, index=False)
    conn.close()
    if not OUT.exists():
        raise SystemExit("No results yet.")
    df = pd.read_csv(OUT, dtype={"season": str, "date": str})
    report(df)


if __name__ == "__main__":
    main()