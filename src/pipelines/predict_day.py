"""
predict_day.py
==============
Forward test: predict every regular-season game on a date, before it is played, and
log the predictions. score_forward.py scores them once the games are loaded.

Home: src/pipelines/predict_day.py

Usage (from the project root; run BEFORE the first game of the day starts):
    py -m src.pipelines.predict_day                         # today's games
    py -m src.pipelines.predict_day --date 2026-10-25
    py -m src.pipelines.predict_day --days 3                # today and the next 2 days
    py -m src.pipelines.predict_day --goalie COL=Blackwood --goalie DAL=Oettinger --replace
                                                            # confirmed starters (last name or id): re-predict
    py -m src.pipelines.predict_day --list-goalies COL DAL  # each team's goalies with their ids
    py -m src.pipelines.predict_day --no-cards              # log only, no pregame cards

Appends to output/forward/predictions_<season>.csv. Every run adds rows (nothing is
overwritten), each stamped with when it was made. A game already in the log is skipped
unless --replace is given; score_forward.py uses the last prediction made on or before
the game date.

Leak-free by construction:
  * refuses games that are already loaded as played (they are in the `games` table),
  * refuses a game whose date is before the active fit's window end (the fit saw it),
  * default starters come from games before the date; goalie ratings are as of the date.
Same-day runs after puck drop can't be detected (the schedule has dates, not times), so
run it in the morning.

Uses the active fit and late_game / score_state files (the rebuild order).

Pregame cards: one PNG per logged game, output/game_summaries/<date>/<away>_at_<home>_pregame.png,
drawn from the same numbers that were logged (src/pipelines/make_cards.py). A --replace run
redraws the card. A card that fails to draw never stops the logging.
"""

import argparse
import json
from datetime import date as date_cls, datetime

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.processing.goalie import goalie_game_xg
from src.processing.late_game import load_params
from src.models.active_fit import read_active, active_fit_path, check_params_match
from src.models.game_model import load_track_b, default_starter, predict_matchup
from src.models.stochastic.goalie import goalie_rating_as_of
from src.models.stochastic.montecarlo import simulate_matchup_score_state
from src.analysis.aggregation import fair_american_odds

FORWARD_DIR = OUTPUT_DIR / "forward"
LATE_GAME_PATH = OUTPUT_DIR / "models" / "late_game.json"
SCORE_STATE_PATH = OUTPUT_DIR / "models" / "score_state.json"
TOTAL_MAX = 14                      # p_total_0..14, 15+ lumped into 14 (same as the backtest)


def scheduled_games(conn, day: str) -> pd.DataFrame:
    """Regular-season games on `day` from the schedule table."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(schedule)")}
    where = "game_date = ?" + (" AND game_type = 2" if "game_type" in cols else "")
    return pd.read_sql(f"""
        SELECT game_id, season, game_date, home_team_abbrev AS home, away_team_abbrev AS away
        FROM schedule WHERE {where} ORDER BY game_id
    """, conn, params=(day,))


def played_ids(conn) -> set:
    return set(pd.read_sql("SELECT id FROM games", conn)["id"])


def parse_goalies(items) -> dict:
    """--goalie TEAM=ID or TEAM=NAME (last name, or part of it) -> {team: text}."""
    out = {}
    for it in items or []:
        team, _, who = it.partition("=")
        if not team or not who:
            raise SystemExit(f"--goalie expects TEAM=ID or TEAM=NAME, got {it!r}")
        out[team.upper()] = who.strip()
    return out


def team_goalies(conn, team: str, days: int = 400) -> pd.DataFrame:
    """Goalies who played for `team` recently: id, name, starts, last game (most recent first)."""
    return pd.read_sql("""
        SELECT gs.player_id AS goalie_id, MAX(gs.name) AS name,
               SUM(CASE WHEN gs.starter = 1 THEN 1 ELSE 0 END) AS starts,
               COUNT(*) AS games, MAX(s.game_date) AS last_game
        FROM goalie_game_stats gs JOIN schedule s ON s.game_id = gs.game_id
        WHERE gs.team_abbreviation = ? AND s.game_date >= date('now', ?)
        GROUP BY gs.player_id ORDER BY last_game DESC, starts DESC
    """, conn, params=(team, f"-{days} days"))


def resolve_goalie(conn, team: str, who: str) -> int:
    """A player id from an id or a (partial) name; prefers the team's recent goalies."""
    if who.isdigit():
        return int(who)
    key = who.lower().replace(".", "").strip()
    recent = team_goalies(conn, team)
    hits = recent[recent["name"].str.lower().str.replace(".", "", regex=False).str.contains(key, regex=False)]
    if hits.empty:                                   # not with this team lately (e.g. just traded): search everyone
        allg = pd.read_sql("""SELECT player_id AS goalie_id, MAX(name) AS name, MAX(team_abbreviation) AS team
                              FROM goalie_game_stats GROUP BY player_id""", conn)
        hits = allg[allg["name"].str.lower().str.replace(".", "", regex=False).str.contains(key, regex=False)]
    if len(hits) == 1:
        return int(hits["goalie_id"].iloc[0])
    if hits.empty:
        raise SystemExit(f"No goalie matching {who!r}. Try: py -m src.pipelines.predict_day --list-goalies {team}")
    raise SystemExit(f"{who!r} matches several goalies; use the id:\n"
                     + hits[["goalie_id", "name"]].to_string(index=False))


def print_goalies(conn, teams_wanted) -> None:
    for team in teams_wanted:
        g = team_goalies(conn, team.upper())
        print(f"\n{team.upper()} goalies (last 400 days, most recent first):")
        print(g.to_string(index=False) if len(g) else "  none found")


def predict_one(row, draws, teams, late, ss, ratings, conn, goalies, score_sims, seed) -> dict:
    home, away, day = row["home"], row["away"], row["game_date"]
    picks, sources = {}, {}
    for side, team in (("home", home), ("away", away)):
        if team in goalies:
            picks[side], sources[side] = goalies[team], "given"
        else:
            picks[side], sources[side] = default_starter(conn, team, day), "default"
    gamma = {}
    for side, gid in picks.items():
        r = ratings[ratings["goalie_id"] == gid] if gid is not None else ratings.iloc[0:0]
        gamma[side] = float(-np.log(r["theta_mean"].iloc[0])) if len(r) else 0.0

    res = predict_matchup(draws, teams, late, home, away, gamma["home"], gamma["away"])
    s = res["summary"]
    sims = simulate_matchup_score_state(draws, teams, late, ss, home, away, gamma["home"], gamma["away"],
                                        n_sims=score_sims, seed=seed)
    fh, fa = sims["final_home"].to_numpy(), sims["final_away"].to_numpy()
    total, margin = fh + fa, fh - fa
    out = {
        "game_id": int(row["game_id"]), "season": str(row["season"]), "game_date": day,
        "home": home, "away": away,
        "home_goalie": picks["home"], "away_goalie": picks["away"],
        "home_goalie_source": sources["home"], "away_goalie_source": sources["away"],
        "gamma_home": gamma["home"], "gamma_away": gamma["away"],
        "p_home": s["home_win"]["mean"], "p_home_lo": s["home_win"]["lo"], "p_home_hi": s["home_win"]["hi"],
        "p_reg_home": s["reg_home"]["mean"], "p_reg_away": s["reg_away"]["mean"], "p_tie_reg": s["tie_reg"]["mean"],
        "lam_home": res["rates"]["home"], "lam_away": res["rates"]["away"],
        "fair_home": fair_american_odds(s["home_win"]["mean"]),
        "fair_away": fair_american_odds(1 - s["home_win"]["mean"]),
        "sim_p_home": float(sims["home_win"].mean()),
        "p_over_5_5": float((total > 5.5).mean()), "p_over_6_5": float((total > 6.5).mean()),
        "p_home_m1_5": float((margin >= 2).mean()), "p_away_m1_5": float((margin <= -2).mean()),
        "exp_total": float(total.mean()),
        "p_home_rate": float(late["home_win_share"]),     # baseline: historical home-win share
    }
    capped = np.minimum(total, TOTAL_MAX)
    for k in range(TOTAL_MAX + 1):
        out[f"p_total_{k}"] = float((capped == k).mean())
    return out, res, sims


def main():
    parser = argparse.ArgumentParser(description="Forward test: predict and log one day's games.")
    parser.add_argument("--date", default=str(date_cls.today()), help="first date, YYYY-MM-DD (default today)")
    parser.add_argument("--days", type=int, default=1, help="number of days from --date (default 1)")
    parser.add_argument("--goalie", action="append", metavar="TEAM=ID",
                        help="confirmed starter by last name or id, e.g. --goalie COL=Blackwood (repeatable)")
    parser.add_argument("--replace", action="store_true",
                        help="re-predict games already in the log (e.g. after starters are confirmed)")
    parser.add_argument("--score-sims", type=int, default=50_000, help="score-state simulations per game")
    parser.add_argument("--no-cards", action="store_true", help="don't draw pregame cards")
    parser.add_argument("--list-goalies", nargs="+", metavar="TEAM",
                        help="print these teams' goalies with their ids, then stop")
    args = parser.parse_args()
    if args.list_goalies:
        conn = get_connection()
        print_goalies(conn, args.list_goalies)
        conn.close()
        return
    goalies = parse_goalies(args.goalie)
    if not args.no_cards:
        import matplotlib
        matplotlib.use("Agg")                     # save cards to files, never open windows

    # 1. Model inputs (active fit + rebuild-order files)
    active = read_active()
    fit_path = active_fit_path()
    draws, teams = load_track_b(fit_path)
    late = load_params(LATE_GAME_PATH)
    with open(SCORE_STATE_PATH) as f:
        ss = json.load(f)
    for params, label in [(late, "late_game.json"), (ss, "score_state.json")]:
        warning = check_params_match(params, label)
        if warning:
            raise SystemExit(warning)
    fit_end = pd.Timestamp(active["end"])

    conn = get_connection()
    goalies = {team: resolve_goalie(conn, team, who) for team, who in goalies.items()}
    for team, gid in goalies.items():
        name = pd.read_sql("SELECT MAX(name) AS n FROM goalie_game_stats WHERE player_id = ?",
                           conn, params=(gid,))["n"].iloc[0]
        print(f"Starter given: {team} {name or 'unknown name'} ({gid})")
    gg = goalie_game_xg(conn)
    done = played_ids(conn)
    made_at = datetime.now().isoformat(timespec="seconds")
    FORWARD_DIR.mkdir(parents=True, exist_ok=True)
    total_new = 0
    teams_seen = set()

    for offset in range(args.days):
        day = str((pd.Timestamp(args.date) + pd.Timedelta(days=offset)).date())
        games = scheduled_games(conn, day)
        teams_seen |= set(games["home"]) | set(games["away"])
        if games.empty:
            print(f"{day}: no regular-season games in the schedule")
            continue
        season = str(games["season"].iloc[0])
        log_path = FORWARD_DIR / f"predictions_{season}.csv"
        logged = set(pd.read_csv(log_path, usecols=["game_id"])["game_id"]) if log_path.exists() else set()
        if pd.Timestamp(day) < fit_end:
            print(f"{day}: SKIPPED, the active fit was trained on games up to {fit_end - pd.Timedelta(days=1):%Y-%m-%d}, "
                  f"so this date's games are not out of sample. Earliest date you can predict: {active['end']} "
                  f"(py -m src.pipelines.predict_day --date {active['end']})")
            continue
        ratings = goalie_rating_as_of(gg, day)

        rows, for_cards = [], []
        print(f"\n{day}: {len(games)} games  (fit {fit_path.name})")
        for i, g in games.iterrows():
            tag = f"  {g['away']:>3s} @ {g['home']:<3s}"
            if g["game_id"] in done:
                print(f"{tag}  skipped: already played and loaded")
                continue
            if g["game_id"] in logged and not args.replace:
                print(f"{tag}  skipped: already predicted (use --replace to re-predict)")
                continue
            if g["home"] not in teams or g["away"] not in teams:
                print(f"{tag}  skipped: team not in the fit")
                continue
            r, res, sims = predict_one(g, draws, teams, late, ss, ratings, conn, goalies, args.score_sims,
                                       seed=int(g["game_id"]) % 2**31)
            r.update({"predicted_at": made_at, "fit_file": fit_path.name, "model": active["label"]})
            rows.append(r)
            for_cards.append((r, res, sims))
            gs = "" if r["home_goalie_source"] == r["away_goalie_source"] == "default" else "  [starters given]"
            print(f"{tag}  {g['home']} {r['p_home']:.1%} ({r['p_home_lo']:.0%}-{r['p_home_hi']:.0%})  "
                  f"fair {r['fair_home']} / {r['fair_away']}   total {r['exp_total']:.2f}, "
                  f"{g['home']} -1.5 {r['p_home_m1_5']:.1%}{gs}")
        if rows:
            df = pd.DataFrame(rows)
            if log_path.exists():                                   # keep the file's column order
                df = df.reindex(columns=pd.read_csv(log_path, nrows=0).columns)
            df.to_csv(log_path, mode="a", index=False, header=not log_path.exists())
            total_new += len(rows)
        if for_cards and not args.no_cards:                         # after logging, so a card error can't lose rows
            from src.pipelines.make_cards import pregame_card
            for r, res, sims in for_cards:
                try:
                    path = pregame_card(r, res, sims, conn)
                    print(f"    card: {path.parent.name}/{path.name}")
                except Exception as e:                              # noqa: BLE001
                    print(f"    card FAILED for {r['away']} @ {r['home']}: {type(e).__name__}: {e}")
    conn.close()
    unused = sorted(set(goalies) - teams_seen)
    if unused:
        print(f"\nWARNING  --goalie given for {', '.join(unused)}, but no game on the date(s) predicted: ignored.")
    if goalies and not args.replace and total_new == 0:
        print("Note: games already in the log are skipped, so --goalie had no effect. Add --replace.")
    print(f"\nLogged {total_new} new predictions to {FORWARD_DIR}")


if __name__ == "__main__":
    main()