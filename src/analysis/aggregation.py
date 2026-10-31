"""
aggregation.py
==============
Turns Phase 6 model output into the numbers people read: win probabilities with
intervals, how the game ends, likely final scores, totals and puck lines, and fair odds.

Home: src/analysis/aggregation.py

Input is the dict returned by game_model.predict_matchup (exact, per posterior draw).
Win probabilities come from that exact model; in predict_game.py the score outputs
(result["final"]) are replaced by the score-state simulation via final_from_sims.
"""

import numpy as np
import pandas as pd

OUTCOME_LABELS = {
    "home_win": "{home} wins",
    "reg_home": "{home} wins in regulation",
    "ot_home": "{home} wins in OT",
    "so_home": "{home} wins in shootout",
    "reg_away": "{away} wins in regulation",
    "ot_away": "{away} wins in OT",
    "so_away": "{away} wins in shootout",
    "tie_reg": "Tied after regulation",
}


def final_from_sims(sims: pd.DataFrame, max_goals: int = 15) -> dict:
    """Final-score block (table, totals, margins) from simulated games.

    Same structure as result["final"] from game_model.predict_matchup, so every
    function below works on either. Used for score outputs, which come from the
    score-state simulator.
    """
    h = sims["final_home"].clip(upper=max_goals).to_numpy()
    a = sims["final_away"].clip(upper=max_goals).to_numpy()
    size = max_goals + 1
    table = np.bincount(h * size + a, minlength=size * size).reshape(size, size) / len(sims)
    totals = np.bincount(h + a, minlength=2 * size - 1) / len(sims)
    margin_vals = np.arange(-(size - 1), size)
    margins = pd.Series(np.bincount(h - a + size - 1, minlength=2 * size - 1) / len(sims), index=margin_vals)
    return {"table": table, "totals": totals, "margins": margins, "source": f"score-state simulation, {len(sims):,} games"}


def fair_american_odds(p: float) -> str:
    """Fair (no bookmaker margin) American odds for probability p, e.g. 0.6 -> '-150'."""
    if not 0 < p < 1:
        return "n/a"
    if p >= 0.5:
        return f"-{round(100 * p / (1 - p))}"
    return f"+{round(100 * (1 - p) / p)}"


def outcome_table(result: dict) -> pd.DataFrame:
    """One row per outcome: probability (mean over draws), 90% interval, fair odds."""
    rows = []
    for key, label in OUTCOME_LABELS.items():
        s = result["summary"][key]
        rows.append({
            "outcome": label.format(home=result["home"], away=result["away"]),
            "probability": s["mean"],
            "90% interval": f"{s['lo']:.1%} - {s['hi']:.1%}",
            "fair odds": fair_american_odds(s["mean"]) if key in ("home_win",) else "",
        })
    away = 1.0 - result["summary"]["home_win"]["mean"]
    rows.insert(1, {"outcome": f"{result['away']} wins", "probability": away,
                    "90% interval": f"{1 - result['summary']['home_win']['hi']:.1%} - "
                                    f"{1 - result['summary']['home_win']['lo']:.1%}",
                    "fair odds": fair_american_odds(away)})
    return pd.DataFrame(rows)


def likely_scores(result: dict, top: int = 8) -> pd.DataFrame:
    """Most likely official final scores (empty-net goals and OT/SO +1 included)."""
    table = result["final"]["table"]
    h, a = np.unravel_index(np.argsort(table, axis=None)[::-1][:top], table.shape)
    return pd.DataFrame({"final score": [f"{result['home']} {i} - {j} {result['away']}" for i, j in zip(h, a)],
                         "probability": table[h, a]})


def totals_and_lines(result: dict, total_lines=(5.5, 6.5), puck_line: float = 1.5) -> pd.DataFrame:
    """Over/under on total goals and the puck line, from the final-score distribution."""
    totals = result["final"]["totals"]
    goals = np.arange(len(totals))
    margins = result["final"]["margins"]                  # home - away, indexed by margin
    home, away = result["home"], result["away"]
    rows = []
    for line in total_lines:
        p_over = float(totals[goals > line].sum())
        rows.append({"market": f"Total over {line}", "probability": p_over, "fair odds": fair_american_odds(p_over)})
        rows.append({"market": f"Total under {line}", "probability": 1 - p_over,
                     "fair odds": fair_american_odds(1 - p_over)})
    p_home_cover = float(margins[margins.index > puck_line].sum())
    p_away_cover = float(margins[margins.index < -puck_line].sum())
    rows += [
        {"market": f"{home} -{puck_line}", "probability": p_home_cover, "fair odds": fair_american_odds(p_home_cover)},
        {"market": f"{away} +{puck_line}", "probability": 1 - p_home_cover, "fair odds": fair_american_odds(1 - p_home_cover)},
        {"market": f"{away} -{puck_line}", "probability": p_away_cover, "fair odds": fair_american_odds(p_away_cover)},
        {"market": f"{home} +{puck_line}", "probability": 1 - p_away_cover, "fair odds": fair_american_odds(1 - p_away_cover)},
    ]
    return pd.DataFrame(rows)


def expected_goals_line(result: dict) -> str:
    """Expected final goals per side (mean of the final-score distribution)."""
    table = result["final"]["table"]
    k = np.arange(table.shape[0])
    eh = float((table.sum(axis=1) * k).sum())
    ea = float((table.sum(axis=0) * k).sum())
    return (f"Expected final score: {result['home']} {eh:.2f} - {ea:.2f} {result['away']} "
            f"(regulation rates {result['rates']['home']:.2f} - {result['rates']['away']:.2f})")


def format_prediction(result: dict, info: dict | None = None) -> str:
    """A readable report for one game. info: optional context lines (date, goalies, fit)."""
    home, away = result["home"], result["away"]
    hw = result["summary"]["home_win"]
    lines = [f"{away} @ {home}"]
    for key, val in (info or {}).items():
        lines.append(f"  {key}: {val}")
    lines += [
        "",
        f"{home} win probability {hw['mean']:.1%} (90% interval {hw['lo']:.1%} - {hw['hi']:.1%})   "
        f"fair odds {home} {fair_american_odds(hw['mean'])} / {away} {fair_american_odds(1 - hw['mean'])}",
        "The interval is how unsure the model is about the two teams' strength; "
        "game-to-game luck is already inside the probability.",
        "",
        expected_goals_line(result),
        "",
        "How the game ends:",
        outcome_table(result).to_string(index=False, formatters={"probability": "{:.1%}".format}),
        "",
        f"Most likely final scores ({result['final'].get('source', 'exact model')}):",
        likely_scores(result).to_string(index=False, formatters={"probability": "{:.1%}".format}),
        "",
        f"Totals and puck line (final score, empty-net goals included; {result['final'].get('source', 'exact model')}):",
        totals_and_lines(result).to_string(index=False, formatters={"probability": "{:.1%}".format}),
    ]
    return "\n".join(lines)


def result_to_json(result: dict, info: dict | None = None) -> dict:
    """JSON-friendly version of a prediction (no per-draw arrays)."""
    return {
        "home": result["home"], "away": result["away"],
        "info": info or {},
        "gamma_home": result["gamma_home"], "gamma_away": result["gamma_away"],
        "regulation_rates": result["rates"],
        "summary": result["summary"],
        "likely_scores": likely_scores(result).to_dict(orient="records"),
        "totals_and_lines": totals_and_lines(result).to_dict(orient="records"),
    }