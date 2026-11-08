"""
playoff_odds.py
===============
Playoff odds: simulate the rest of the regular season from the active fit (season_sim.py).

Home: src/pipelines/playoff_odds.py

Usage (from the project root, after load_all; any morning, e.g. weekly after the refit):
    py -m src.pipelines.playoff_odds                  # 10,000 simulated seasons from today
    py -m src.pipelines.playoff_odds --sims 20000
    py -m src.pipelines.playoff_odds --date 2026-12-01 --no-check

1. Current standings from the database (played regular-season games), checked against the
   NHL's own standings feed (points, games played) unless --no-check.
2. Remaining games: back-to-back / rest shifts from the schedule (as predict_day), each
   team's goalie offset = its goalies' ratings weighted by their share of the last 20 starts,
   lineups at full strength.
3. 10,000 seasons, each with one posterior draw for all its games plus weekly rating drift.
Prints each division with points now, projected points (mean, 90% range) and the chances to
make the playoffs, win the division, take a wild card, finish first in the conference, and
win the Presidents' Trophy. Saves output/playoffs/playoff_odds_<date>.csv and appends to
playoff_odds_history_<season>.csv (one row per team per run, for charts over the season), then
draws playoff_board_<date>.png and playoff_trend_<season>.png (make_playoff_charts.py; --no-charts).
"""

import argparse
from datetime import date as date_cls

import arviz as az
import numpy as np
import pandas as pd
import requests

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.active_fit import active_fit_path, read_active
from src.models.game_model import load_track_b
from src.models.stochastic import season_sim as ss
from src.models.stochastic.goalie import goalie_rating_as_of
from src.models.stochastic.hierarchical_tv import context_shift
from src.processing.goalie import goalie_game_xg
from src.processing.late_game import load_params
from src.processing.schedule_context import FRANCHISE, schedule_context

PLAYOFF_DIR = OUTPUT_DIR / "playoffs"
LATE_GAME_PATH = OUTPUT_DIR / "models" / "late_game.json"
STANDINGS_URL = "https://api-web.nhle.com/v1/standings/{date}"


def taus(fit_path, n_draws: int) -> dict:
    """tau_off / tau_def per draw from the fit file (load_track_b does not return them)."""
    try:
        post = az.from_netcdf(fit_path).posterior
        out = {}
        for k in ("tau_off", "tau_def"):
            if k in post:
                v = post[k].values.reshape(-1)
                out[k] = v if len(v) == n_draws else np.resize(v, n_draws)
        return out
    except Exception:
        return {}


def goalie_mix(conn, gg, team: str, day, n: int = 20) -> float:
    """Team goalie offset: its goalies' Phase 3 gamma weighted by share of the last n starts."""
    st = pd.read_sql("""SELECT g.player_id FROM goalie_game_stats g JOIN schedule s ON s.game_id = g.game_id
                        WHERE g.starter = 1 AND g.team_abbreviation = ? AND s.game_date < ?
                        ORDER BY s.game_date DESC LIMIT ?""", conn, params=(team, str(day), n))
    if st.empty:
        return 0.0
    r = goalie_rating_as_of(gg, str(day)).set_index("goalie_id")["theta_mean"]
    share = st["player_id"].value_counts(normalize=True)
    th = r.reindex(share.index)
    gam = -np.log(th.fillna(1.0))
    return float((share * gam).sum())


def nhl_check(standings: pd.DataFrame, day) -> None:
    try:
        r = requests.get(STANDINGS_URL.format(date=day), timeout=20)
        r.raise_for_status()
        rows = [{"team": FRANCHISE.get(e["teamAbbrev"]["default"], e["teamAbbrev"]["default"]),
                 "nhl_pts": e.get("points"), "nhl_gp": e.get("gamesPlayed")} for e in r.json().get("standings", [])]
    except Exception as e:                                    # noqa: BLE001
        print(f"NHL standings check skipped ({type(e).__name__})")
        return
    m = standings.merge(pd.DataFrame(rows), on="team", how="left")
    bad = m[(m["pts"] != m["nhl_pts"]) | (m["gp"] != m["nhl_gp"])]
    if bad.empty:
        print(f"Standings match the NHL's for all {len(m)} teams.")
    else:
        print(f"Standings differ from the NHL's for {len(bad)} teams (missing games? run load_all):")
        print(bad[["team", "gp", "nhl_gp", "pts", "nhl_pts"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Playoff odds from the active fit.")
    parser.add_argument("--date", default=str(date_cls.today()), help="simulate from this date (default today)")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-drift", action="store_true", help="no rating drift over the rest of the season")
    parser.add_argument("--no-check", action="store_true", help="skip the check against the NHL standings feed")
    parser.add_argument("--no-charts", action="store_true", help="skip the PNG charts (make_playoff_charts)")
    args = parser.parse_args()
    day = pd.Timestamp(args.date).date()

    active = read_active()
    fit_path = active_fit_path()
    draws, teams = load_track_b(fit_path)
    draws.update(taus(fit_path, len(draws["rho"])))
    late = load_params(LATE_GAME_PATH)
    conn = get_connection()

    sched = pd.read_sql("""SELECT s.game_id, s.season, s.game_date, s.home_team_abbrev AS home, s.away_team_abbrev AS away
                           FROM schedule s WHERE s.game_date >= ?""", conn, params=(str(day),))
    played = {g for (g,) in conn.execute("SELECT id FROM games WHERE home_score IS NOT NULL")}
    sched = sched[~sched["game_id"].isin(played)]
    gtype = {g: t for g, t in conn.execute("SELECT game_id, game_type FROM schedule")} \
        if "game_type" in {r[1] for r in conn.execute("PRAGMA table_info(schedule)")} else {}
    if gtype:
        sched = sched[sched["game_id"].map(gtype).eq(2)]
    if sched.empty:
        raise SystemExit(f"No regular-season games left on or after {day}.")
    season = str(sched["season"].min())
    sched = sched[sched["season"].astype(str) == season].copy()
    sched[["home", "away"]] = sched[["home", "away"]].replace(FRANCHISE)

    info = pd.read_sql("SELECT abbreviation AS team, division, conference FROM teams", conn)
    info["team"] = info["team"].replace(FRANCHISE)
    in_season = set(sched["home"]) | set(sched["away"])
    info = info[info["team"].isin(in_season)].drop_duplicates("team").sort_values(["conference", "division", "team"])
    missing = in_season - set(teams)
    if missing:
        raise SystemExit(f"Teams not in the active fit: {sorted(missing)}")
    standings = ss.current_standings(conn, season, info, FRANCHISE)
    print(f"Season {season}, from {day}: {int(standings['gp'].sum() / 2)} games played, {len(sched)} left "
          f"(fit {fit_path.name})")
    if not args.no_check and day == date_cls.today():
        nhl_check(standings, day)

    # known per-game shifts: rest / back-to-backs (lineups at full strength, no goalie back-to-back info)
    ctx = sched[["game_id"]].merge(schedule_context(conn), on="game_id", how="left").fillna(0)
    for c in ("lin_home", "lin_away", "g2b_home", "g2b_away"):
        ctx[c] = 0.0
    sh, sa = context_shift(draws, ctx)
    sched["shift_home"], sched["shift_away"] = np.asarray(sh, float), np.asarray(sa, float)
    gg = goalie_game_xg(conn)
    gamma = {t: goalie_mix(conn, gg, t, day) for t in info["team"]}
    conn.close()

    print(f"Simulating {args.sims:,} seasons{' (no rating drift)' if args.no_drift else ''}...", flush=True)
    res = ss.simulate_season(draws, teams, late, sched, standings, info.reset_index(drop=True), gamma, day,
                             n_sims=args.sims, seed=args.seed, drift=not args.no_drift)
    tab = ss.summarize(res, info.reset_index(drop=True), standings)
    spots = sum(min(3, n) for n in info.groupby("division").size()) + sum(
        min(2, max(0, len(g) - sum(min(3, n) for n in g.groupby("division").size()))) for _, g in info.groupby("conference"))
    assert (res["playoffs"].sum(axis=1) == spots).all(), f"every simulated season must have exactly {spots} playoff teams"

    pd.set_option("display.width", 200)
    show = tab.assign(points=lambda d: d["points_mean"].round(0).astype(int).astype(str) + " ("
                      + d["points_lo"].astype(int).astype(str) + "-" + d["points_hi"].astype(int).astype(str) + ")")
    for (cf, dv), g in show.groupby(["conference", "division"], sort=True):
        g = g.sort_values(["p_playoffs", "points_mean"], ascending=False)
        print(f"\n{cf} / {dv}")
        print(g[["team", "gp_now", "points_now", "points", "p_playoffs", "p_div_win", "p_wildcard", "p_conf_first",
                 "p_presidents"]].rename(columns={"gp_now": "GP", "points_now": "PTS", "points": "projected (90%)",
                                                  "p_playoffs": "playoffs", "p_div_win": "division", "p_wildcard": "wild card",
                                                  "p_conf_first": "1st conf", "p_presidents": "Presidents'"})
              .to_string(index=False, formatters={c: "{:.1%}".format for c in
                                                  ["playoffs", "division", "wild card", "1st conf", "Presidents'"]}))

    PLAYOFF_DIR.mkdir(parents=True, exist_ok=True)
    tab.insert(0, "date", str(day))
    tab["fit_file"], tab["sims"] = fit_path.name, args.sims
    tab.to_csv(PLAYOFF_DIR / f"playoff_odds_{day}.csv", index=False)
    hist = PLAYOFF_DIR / f"playoff_odds_history_{season}.csv"
    if hist.exists():
        old = pd.read_csv(hist)
        tab = pd.concat([old[old["date"] != str(day)], tab], ignore_index=True)
    tab.to_csv(hist, index=False)
    print(f"\nSaved playoff_odds_{day}.csv and {hist.name} to {PLAYOFF_DIR}")
    if not args.no_charts:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from src.pipelines.make_playoff_charts import make_charts
            for p in make_charts(str(day)):
                print(f"Saved {p.name}")
        except Exception as e:                                # noqa: BLE001
            print(f"Charts FAILED ({type(e).__name__}: {e}); the odds are saved. Retry: py -m src.pipelines.make_playoff_charts")


if __name__ == "__main__":
    main()