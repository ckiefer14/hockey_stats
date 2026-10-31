"""
score_state.py
==============
Measures how scoring changes with the score, from play-by-play, for the Phase 6
score-state simulator.

Home: src/processing/score_state.py

Run (after fit_hierarchical.py), from the project root:
    py -m src.processing.score_state
Uses the active fit (output/posteriors/active_fit.json) and the games it was trained on.
Writes output/models/score_state.json (including the simulator's calibrated scale)
and prints a summary.

What it measures (regulation only, periods 1-3)
-----------------------------------------------
Normal play (both goalies in net), by score state from each team's point of view:
    trailing 2+, trailing 1, tied, tied late (last 5 min of period 3), leading 1, leading 2+
    factor = goals scored / goals expected from the team's own Track B rate for that time.
Pulled goalie (a trailing team in the last 6 minutes of period 3, goalie out):
    pull hazard   chance per second of pulling, by margin (1, 2, 3+) and time left (30 s bins)
    trailer       6-on-5 goals relative to its normal rate, by margin
    leader        empty-net goals per minute, by margin
Goalie pulls outside that situation (delayed penalties, rare early pulls) count as normal play.

The goalie-in-net state comes from situation_code (digits: away goalie, away skaters,
home skaters, home goalie); a 0 goalie digit means the net is empty. The state at an
event holds until the next event. After a goal both goalies are treated as back in net
until an event says otherwise.
"""

import json
import sqlite3

import numpy as np
import pandas as pd

REG_END = 3 * 1200            # seconds in regulation
PULL_WINDOW = 360             # last 6 minutes of period 3: pulls modeled here
TIED_LATE = 300               # last 5 minutes of period 3: "tied late" state
BIN = 30                      # seconds per pull-hazard bin
MARGINS = [1, 2, 3]           # 3 means 3 or more
STATES = ["trailing 2+", "trailing 1", "tied", "tied late", "leading 1", "leading 2+"]


def state_of(diff: int, t: float) -> str:
    """Score state for a team `diff` goals ahead at game second t (normal play)."""
    if diff <= -2:
        return "trailing 2+"
    if diff == -1:
        return "trailing 1"
    if diff == 0:
        return "tied late" if t >= REG_END - TIED_LATE else "tied"
    if diff == 1:
        return "leading 1"
    return "leading 2+"


def load_events(conn: sqlite3.Connection, game_ids) -> pd.DataFrame:
    """Every regulation event for the given games, in order, with goal and goalie flags."""
    ev = pd.read_sql("""
        SELECT  ge.game_id, ge.sort_order, ge.game_seconds, ge.event_type, ge.situation_code,
                CASE WHEN ge.team_id = s.home_team_id THEN 1
                     WHEN ge.team_id = s.away_team_id THEN 0 END AS is_home,
                CASE WHEN ge.event_type = 'goal' AND x.event_id IS NULL THEN 1 ELSE 0 END AS empty_net
        FROM    game_events ge
        JOIN    schedule s ON s.game_id = ge.game_id
        LEFT JOIN shot_xg x ON x.game_id = ge.game_id AND x.event_id = ge.event_id
        WHERE   ge.period <= 3
    """, conn)
    ev = ev[ev["game_id"].isin(set(game_ids))].sort_values(["game_id", "sort_order"]).reset_index(drop=True)

    code = ev["situation_code"].astype("string").str.zfill(4)
    ev["away_pulled"] = (code.str[0] == "0").astype("float").where(code.notna())
    ev["home_pulled"] = (code.str[3] == "0").astype("float").where(code.notna())
    # Missing codes: carry the last known state within the game
    ev[["away_pulled", "home_pulled"]] = ev.groupby("game_id")[["away_pulled", "home_pulled"]].ffill().fillna(0)
    return ev


def measure(ev: pd.DataFrame, rates: pd.DataFrame) -> dict:
    """Walk every game and accumulate time, goals and expected goals by situation.

    rates: game_id, lam_home, lam_away (Track B regulation rates, goals per 60 minutes).
    """
    n_bins = PULL_WINDOW // BIN
    acc = {
        "state": {s: {"secs": 0.0, "goals": 0, "expected": 0.0} for s in STATES},
        "pulled": {m: {"secs": 0.0, "trailer_goals": 0, "trailer_expected": 0.0, "leader_en": 0}
                   for m in MARGINS},
        "risk_secs": {m: np.zeros(n_bins) for m in MARGINS},
        "pulls": {m: np.zeros(n_bins) for m in MARGINS},
        "other_en": 0, "games": 0,
    }
    lam = rates.set_index("game_id")[["lam_home", "lam_away"]]

    def is_pulled_trailer(pulled, diff, t):
        return pulled and diff < 0 and t >= REG_END - PULL_WINDOW

    def bin_of(t):
        return int(min(n_bins - 1, max(0, (REG_END - t) // BIN)))     # 0 = last 30 s

    for gid, g in ev.groupby("game_id", sort=False):
        if gid not in lam.index:
            continue
        acc["games"] += 1
        rate = {1: lam.at[gid, "lam_home"] / REG_END, 0: lam.at[gid, "lam_away"] / REG_END}   # goals per second
        score = {1: 0, 0: 0}
        pulled = {1: False, 0: False}
        t_prev = 0.0

        times = g["game_seconds"].to_numpy(float)
        types = g["event_type"].to_numpy()
        homes = g["is_home"].to_numpy()
        ens = g["empty_net"].to_numpy()
        hp, ap = g["home_pulled"].to_numpy(), g["away_pulled"].to_numpy()

        for i in range(len(g) + 1):
            t = times[i] if i < len(g) else REG_END
            t = min(max(t, t_prev), REG_END)

            # 1. Time from t_prev to t in the current state, for both teams
            dt = t - t_prev
            if dt > 0:
                for team in (1, 0):
                    opp = 1 - team
                    diff = score[team] - score[opp]
                    if is_pulled_trailer(pulled[team], diff, t_prev):
                        m = min(-diff, 3)
                        acc["pulled"][m]["secs"] += dt
                        acc["pulled"][m]["trailer_expected"] += rate[team] * dt
                    elif not is_pulled_trailer(pulled[opp], -diff, t_prev):
                        s = state_of(diff, t_prev)
                        acc["state"][s]["secs"] += dt
                        acc["state"][s]["expected"] += rate[team] * dt
                    if diff < 0 and not pulled[team] and t_prev >= REG_END - PULL_WINDOW:
                        acc["risk_secs"][min(-diff, 3)][bin_of(t_prev)] += dt
            if i == len(g):
                break

            # 2. The event itself
            if types[i] == "goal" and homes[i] in (0, 1):
                team = int(homes[i]); opp = 1 - team
                diff = score[team] - score[opp]
                if ens[i]:
                    if is_pulled_trailer(pulled[opp], -diff, t):
                        acc["pulled"][min(diff, 3)]["leader_en"] += 1
                    else:
                        acc["other_en"] += 1
                elif is_pulled_trailer(pulled[team], diff, t):
                    acc["pulled"][min(-diff, 3)]["trailer_goals"] += 1
                elif not is_pulled_trailer(pulled[opp], -diff, t):
                    acc["state"][state_of(diff, t)]["goals"] += 1
                score[team] += 1
                pulled = {1: False, 0: False}            # goalies back in after a goal
            else:
                new = {1: bool(hp[i]), 0: bool(ap[i])}
                for team in (1, 0):
                    diff = score[team] - score[1 - team]
                    if new[team] and not pulled[team] and diff < 0 and t >= REG_END - PULL_WINDOW:
                        acc["pulls"][min(-diff, 3)][bin_of(t)] += 1
                pulled = new
            t_prev = t
    return acc


def summarize(acc: dict) -> dict:
    """Turn accumulated totals into the simulator's parameters."""
    out = {"games": acc["games"], "window_seconds": PULL_WINDOW, "bin_seconds": BIN,
           "tied_late_seconds": TIED_LATE, "state_factor": {}, "state_detail": {},
           "pulled": {}, "pull_hazard": {}, "other_empty_net_goals": acc["other_en"]}
    for s, a in acc["state"].items():
        f = a["goals"] / a["expected"] if a["expected"] > 0 else 1.0
        se = np.sqrt(a["goals"]) / a["expected"] if a["expected"] > 0 else np.nan
        out["state_factor"][s] = float(f)
        out["state_detail"][s] = {"minutes": a["secs"] / 60, "goals": a["goals"], "factor_se": float(se)}
    for m in MARGINS:
        p = acc["pulled"][m]
        minutes = p["secs"] / 60
        out["pulled"][str(m)] = {
            "minutes": minutes,
            "trailer_goals": p["trailer_goals"],
            "trailer_factor": float(p["trailer_goals"] / p["trailer_expected"]) if p["trailer_expected"] > 0 else 1.0,
            "leader_en_goals": p["leader_en"],
            "leader_en_per_min": float(p["leader_en"] / minutes) if minutes > 0 else 0.0,
        }
        risk, pulls = acc["risk_secs"][m], acc["pulls"][m]
        # per-second hazard by bin (index 0 = last 30 s); bins with little time borrow the margin average
        overall = pulls.sum() / risk.sum() if risk.sum() > 0 else 0.0
        hz = np.where(risk >= 60, pulls / np.maximum(risk, 1e-9), overall)
        out["pull_hazard"][str(m)] = [float(h) for h in hz]
    return out


def score_state_params(conn: sqlite3.Connection, gt: pd.DataFrame, post) -> dict:
    """Measure everything for the fit-window games in gt."""
    from src.processing.late_game import mean_rates
    rates = mean_rates(post, gt)
    ev = load_events(conn, gt["game_id"])
    return summarize(measure(ev, rates))


def main():
    import arviz as az
    from config.settings import OUTPUT_DIR
    from src.data.db import get_connection
    from src.models.active_fit import read_active, active_fit_path, window_games

    # The active fit (fit_hierarchical.py) and the games it was trained on
    info = read_active()
    fit_path = active_fit_path()
    idata = az.from_netcdf(fit_path).load()
    if hasattr(idata, "close"):
        idata.close()

    conn = get_connection()
    gt, _ = window_games(conn, info)
    print(f"Active fit {fit_path.name}: {len(gt):,} games, {info['start']} to {info['end']} (end excluded)")
    params = score_state_params(conn, gt, idata.posterior)
    conn.close()
    params["fit_file"] = fit_path.name
    params["window"] = [info["start"], info["end"]]

    # Calibrate the simulator's overall scale on the same games
    from src.models.game_model import load_track_b
    from src.models.stochastic.montecarlo import calibrate_scale
    draws, _ = load_track_b(fit_path)
    params["scale"] = calibrate_scale(draws, None, params, gt)

    out = OUTPUT_DIR / "models" / "score_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(params, f, indent=2)

    print(f"{params['games']:,} games\n\nNormal play (both goalies in), goals vs each team's own pace:")
    for s in STATES:
        d = params["state_detail"][s]
        print(f"  {s:12s} {params['state_factor'][s]:.3f} ±{1.645 * d['factor_se']:.3f}   "
              f"({d['minutes']:,.0f} min, {d['goals']:,} goals)")
    print("\nGoalie pulled (trailing team, last 6 min of period 3):")
    for m, p in params["pulled"].items():
        label = f"{m}+" if int(m) == MARGINS[-1] else m
        print(f"  down {label:2s}: {p['minutes']:6,.0f} min   trailer 6-on-5 factor {p['trailer_factor']:.2f} "
              f"({p['trailer_goals']} goals)   leader empty-net {p['leader_en_per_min']:.3f}/min ({p['leader_en_goals']} goals)")
    print("\nPull timing: share of trailing teams that have pulled by each time left (no goals in between):")
    for m, hz in params["pull_hazard"].items():
        hz = np.array(hz)
        survive = np.exp(-np.cumsum((hz * BIN)[::-1]))                 # from 6:00 down to 0:00
        marks = {"4:00": survive[3], "3:00": survive[5], "2:00": survive[7], "1:00": survive[9]}
        label = f"{m}+" if int(m) == MARGINS[-1] else m
        print(f"  down {label:2s}: " + "   ".join(f"by {k}: {1 - v:.0%}" for k, v in marks.items()))
    print(f"\nEmpty-net goals outside a late trailing pull: {params['other_empty_net_goals']}")
    print(f"Simulator scale (keeps average goals equal to Track B's rates): {params['scale']:.4f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()