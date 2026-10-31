"""
predict_game.py
===============
Predict one game from the saved Track B fit and late-game parameters.

Home: src/pipelines/predict_game.py

Usage (from the project root; HOME team first):
    py -m src.pipelines.predict_game COL DAL
    py -m src.pipelines.predict_game COL DAL --date 2026-10-25
    py -m src.pipelines.predict_game COL DAL --home-goalie 8478406 --away-goalie 8479979
    py -m src.pipelines.predict_game COL DAL --check-sims 200000 --save

Needs, in this order (each built from the one before):
       the active fit, output/posteriors/active_fit.json (py -m src.pipelines.fit_hierarchical),
       output/models/late_game.json (py -m src.processing.late_game) and
       output/models/score_state.json (py -m src.processing.score_state).
A warning is printed if late_game / score_state were measured on a different fit.

Win probabilities and how the game ends come from the exact model (game_model).
Score outputs (likely final scores, totals, puck lines) come from the score-state
simulator, which reproduces real score shapes better (Phase 6, section 9).

Goalies: pass the starters' player ids if known. Otherwise each team's default is the
goalie who started most of its last 10 games before the date (printed in the output).
"""

import argparse
import json
from datetime import date as date_cls

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.processing.goalie import goalie_game_xg
from src.processing.late_game import load_params
from src.models.active_fit import read_active, active_fit_path, check_params_match
from src.models.game_model import load_track_b, default_starter, goalie_offset, predict_matchup
from src.models.stochastic.montecarlo import simulate_matchup, simulate_matchup_score_state, summarize_sims
from src.analysis.aggregation import format_prediction, result_to_json, final_from_sims

LATE_GAME_PATH = OUTPUT_DIR / "models" / "late_game.json"
SCORE_STATE_PATH = OUTPUT_DIR / "models" / "score_state.json"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


def goalie_name(conn, goalie_id) -> str:
    if goalie_id is None:
        return "unknown (league average)"
    row = pd.read_sql("SELECT MAX(name) AS name FROM goalie_game_stats WHERE player_id = ?",
                      conn, params=(int(goalie_id),))
    name = row["name"].iloc[0] if len(row) else None
    return f"{name} ({goalie_id})" if name else str(goalie_id)


def main():
    parser = argparse.ArgumentParser(description="Predict one NHL game (home team first).")
    parser.add_argument("home", help="home team abbreviation, e.g. COL")
    parser.add_argument("away", help="away team abbreviation, e.g. DAL")
    parser.add_argument("--date", default=str(date_cls.today()), help="game date, YYYY-MM-DD (default today)")
    parser.add_argument("--home-goalie", type=int, default=None, help="home starter player id")
    parser.add_argument("--away-goalie", type=int, default=None, help="away starter player id")
    parser.add_argument("--score-sims", type=int, default=100_000,
                        help="score-state simulations for the score outputs (default 100,000)")
    parser.add_argument("--check-sims", type=int, default=0,
                        help="also simulate this many games and compare with the exact numbers")
    parser.add_argument("--save", action="store_true", help="save the prediction as JSON")
    args = parser.parse_args()
    home, away = args.home.upper(), args.away.upper()

    # 1. Model inputs
    active = read_active()
    fit_path = active_fit_path()
    draws, teams = load_track_b(fit_path)
    late = load_params(LATE_GAME_PATH)
    with open(SCORE_STATE_PATH) as f:
        ss = json.load(f)
    for params, label in [(late, "late_game.json"), (ss, "score_state.json")]:
        warning = check_params_match(params, label)
        if warning:
            print(warning + "\n")
    for t in (home, away):
        if t not in teams:
            raise SystemExit(f"{t} is not in the fit. Teams: {', '.join(teams)}")

    # 2. Goalies: given, or each team's most frequent recent starter
    conn = get_connection()
    home_g = args.home_goalie if args.home_goalie is not None else default_starter(conn, home, args.date)
    away_g = args.away_goalie if args.away_goalie is not None else default_starter(conn, away, args.date)
    gg = goalie_game_xg(conn)
    gamma_home = goalie_offset(gg, home_g, args.date)
    gamma_away = goalie_offset(gg, away_g, args.date)
    info = {
        "date": args.date,
        "home goalie": f"{goalie_name(conn, home_g)}, offset {gamma_home:+.3f}"
                       + ("" if args.home_goalie is not None else " [default: most starts in last 10]"),
        "away goalie": f"{goalie_name(conn, away_g)}, offset {gamma_away:+.3f}"
                       + ("" if args.away_goalie is not None else " [default: most starts in last 10]"),
        "ratings": (f"{fit_path.name} ({'Track B v1.1, xG + goals' if active['model'] == 'xg' else 'Track B v1, goals'}; "
                    f"games {active['start']} to {active['end']}; {draws['alpha0'].size:,} posterior draws)"),
    }
    conn.close()

    # 3. Win probabilities: exact model. Score outputs: score-state simulation.
    result = predict_matchup(draws, teams, late, home, away, gamma_home, gamma_away)
    score_sims = simulate_matchup_score_state(draws, teams, late, ss, home, away,
                                              gamma_home, gamma_away, n_sims=args.score_sims)
    result["final"] = final_from_sims(score_sims)
    p_ss = score_sims["home_win"].mean()
    info["score outputs"] = (f"score-state simulation, {args.score_sims:,} games "
                             f"(its {home} win share {p_ss:.1%} vs exact {result['summary']['home_win']['mean']:.1%})")
    print(format_prediction(result, info))

    # 4. Optional: Monte Carlo cross-check
    if args.check_sims > 0:
        sims = simulate_matchup(draws, teams, late, home, away, gamma_home, gamma_away, n_sims=args.check_sims)
        sm = summarize_sims(sims)
        tol = 3 * np.sqrt(0.25 / args.check_sims)
        print(f"\nMonte Carlo check ({args.check_sims:,} simulated games, tolerance ±{tol:.4f}):")
        for key in ["home_win", "reg_home", "tie_reg", "ot_home", "so_home"]:
            ex = result["summary"][key]["mean"]
            flag = "ok" if abs(sm[key] - ex) < tol else "CHECK"
            print(f"  {key:9s} exact {ex:.4f}  simulated {sm[key]:.4f}  {flag}")

    # 5. Optional: save
    if args.save:
        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
        out = PREDICTION_DIR / f"{args.date}_{away}_at_{home}.json"
        with open(out, "w") as f:
            json.dump(result_to_json(result, info), f, indent=2)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()