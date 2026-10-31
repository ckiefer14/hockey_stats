"""
late_game.py
============
Measures, from play-by-play, the parts of a game the Track B model doesn't cover:
overtime, the shootout and empty-net goals. Phase 6 uses these to simulate a full game.

Home: src/processing/late_game.py

Run (after fit_hierarchical.py), from the project root:
    py -m src.processing.late_game
Uses the active fit (output/posteriors/active_fit.json) and the games it was trained on.
Writes output/models/late_game.json and prints a summary.

What it estimates
-----------------
OT factor   Overtime is 5 minutes of 3-on-3, sudden death. Each team's OT scoring rate is
            its regulation rate (Track B, per 60 min) times ot_factor, so
                P(OT goal)        = 1 - exp(-ot_factor * (lam_home + lam_away) * 5 / 60)
                P(home scores it) = lam_home / (lam_home + lam_away)
            ot_factor is fit by maximum likelihood to which tied games ended in OT.
Shootout    Share of shootouts won by the home team (Phase 6 default stays 50/50
            unless the data clearly says otherwise).
Empty net   For games decided in regulation, by regulation margin (no empty-netters)
            of 1, 2, 3+: the share of games where the leader added 0, 1 or 2+
            empty-net goals. Only used for final-score outputs, never for P(win).
"""

import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import optimize, stats

OT_MINUTES = 5.0
EN_MARGINS = [1, 2, 3]            # 3 means "3 or more"
EN_MAX_GOALS = 2                  # 2 means "2 or more"


def game_results(conn: sqlite3.Connection, gt: pd.DataFrame) -> pd.DataFrame:
    """How each game in gt (output of build_game_table) actually ended.

    Columns: game_id, home_reg, away_reg (regulation goals, no empty-netters),
             decided ("REG", "OT", "SO"), home_won, ot_home (1/0 if decided by an
             OT goal), so_home (1/0 if decided by shootout), home_en, away_en
             (empty-net goals in regulation)
    """
    # 1. Final result and how the game ended (games table)
    final = pd.read_sql("""
        SELECT id AS game_id, home_score, away_score, period_type AS decided
        FROM   games
        WHERE  game_type = 2
    """, conn)

    # 2. Who scored the OT goal (period 4)
    ot = pd.read_sql("""
        SELECT  se.game_id,
                MAX(CASE WHEN se.team_id = s.home_team_id THEN 1 ELSE 0 END) AS ot_home
        FROM    shot_events se
        JOIN    schedule s ON s.game_id = se.game_id
        WHERE   se.event_type = 'goal' AND se.period = 4
        GROUP BY se.game_id
    """, conn)

    # 3. Empty-net goals in regulation (a goal not in shot_xg = net was empty)
    en = pd.read_sql("""
        SELECT  se.game_id,
                SUM(CASE WHEN se.team_id = s.home_team_id THEN 1 ELSE 0 END) AS home_en,
                SUM(CASE WHEN se.team_id = s.away_team_id THEN 1 ELSE 0 END) AS away_en
        FROM    shot_events se
        JOIN    schedule s     ON s.game_id = se.game_id
        LEFT JOIN shot_xg x    ON x.game_id = se.game_id AND x.event_id = se.event_id
        WHERE   se.event_type = 'goal' AND se.period <= 3 AND x.event_id IS NULL
        GROUP BY se.game_id
    """, conn)

    df = (gt[["game_id", "home_goals_reg", "away_goals_reg"]]
          .rename(columns={"home_goals_reg": "home_reg", "away_goals_reg": "away_reg"})
          .merge(final, on="game_id", how="left")
          .merge(ot, on="game_id", how="left")
          .merge(en, on="game_id", how="left"))
    df[["home_en", "away_en"]] = df[["home_en", "away_en"]].fillna(0).astype(int)
    df["home_won"] = (df["home_score"] > df["away_score"]).astype(int)
    df["ot_home"] = df["ot_home"].where(df["decided"] == "OT")
    df["so_home"] = df["home_won"].where(df["decided"] == "SO")
    return df.drop(columns=["home_score", "away_score"])


def mean_rates(post, gt: pd.DataFrame) -> pd.DataFrame:
    """Each game's regulation scoring rates from Track B posterior means.

    post: the posterior group of a saved Track B fit (needs alpha0, alpha_home, mu_off, mu_def).
    Returns game_id, lam_home, lam_away (expected regulation goals per game).
    """
    m = {n: post[n].values.reshape(-1, *post[n].values.shape[2:]).mean(axis=0)
         for n in ["alpha0", "alpha_home", "mu_off", "mu_def"]}
    hi, ai = gt["home_idx"].to_numpy(), gt["away_idx"].to_numpy()
    lam_home = np.exp(m["alpha0"] + m["alpha_home"] + m["mu_off"][hi] - m["mu_def"][ai] - gt["gamma_away"].to_numpy())
    lam_away = np.exp(m["alpha0"] + m["mu_off"][ai] - m["mu_def"][hi] - gt["gamma_home"].to_numpy())
    return pd.DataFrame({"game_id": gt["game_id"].to_numpy(), "lam_home": lam_home, "lam_away": lam_away})


def fit_ot_factor(results: pd.DataFrame, rates: pd.DataFrame) -> dict:
    """Maximum-likelihood OT scoring factor from games that went past regulation.

    Each OT game contributes log P(OT goal) if it was decided in OT, else
    log(1 - P(OT goal)), with P(OT goal) = 1 - exp(-k * (lam_home + lam_away) * 5/60).
    Also checks who scored: P(home) = lam_home / (lam_home + lam_away).
    """
    d = results[results["decided"].isin(["OT", "SO"])].merge(rates, on="game_id")
    total = (d["lam_home"] + d["lam_away"]).to_numpy() * OT_MINUTES / 60.0
    scored = (d["decided"] == "OT").to_numpy()

    def neg_loglik(log_k):
        p = 1.0 - np.exp(-np.exp(log_k) * total)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(np.where(scored, np.log(p), np.log1p(-p)))

    res = optimize.minimize_scalar(neg_loglik, bounds=(np.log(0.1), np.log(10.0)), method="bounded")
    k = float(np.exp(res.x))

    # Standard error from the curvature of the log-likelihood (on the log scale)
    h = 1e-3
    curv = (neg_loglik(res.x + h) - 2 * neg_loglik(res.x) + neg_loglik(res.x - h)) / h ** 2
    se_log_k = float(1.0 / np.sqrt(curv)) if curv > 0 else float("nan")

    p_fit = 1.0 - np.exp(-k * total)
    ot_games = d[scored]
    share_home = ot_games["lam_home"] / (ot_games["lam_home"] + ot_games["lam_away"])
    return {
        "ot_factor": k,
        "ot_factor_90": [float(np.exp(res.x - 1.645 * se_log_k)), float(np.exp(res.x + 1.645 * se_log_k))],
        "games_past_regulation": int(len(d)),
        "share_decided_in_ot": float(scored.mean()),
        "share_decided_in_ot_fitted": float(p_fit.mean()),
        "ot_goals": int(len(ot_games)),
        "ot_home_share": float(ot_games["ot_home"].mean()) if len(ot_games) else float("nan"),
        "ot_home_share_expected": float(share_home.mean()) if len(ot_games) else float("nan"),
    }


def shootout_home(results: pd.DataFrame) -> dict:
    """Share of shootouts won by the home team, with a test against 50/50."""
    so = results["so_home"].dropna()
    n, wins = int(len(so)), int(so.sum())
    if n == 0:
        return {"shootouts": 0, "home_share": float("nan"), "home_share_90": [float("nan")] * 2,
                "p_value_vs_half": float("nan"), "use_home_prob": 0.5}
    lo, hi = stats.beta.ppf([0.05, 0.95], wins + 0.5, n - wins + 0.5)     # Jeffreys interval
    p_value = float(stats.binomtest(wins, n, 0.5).pvalue)
    return {"shootouts": n, "home_share": wins / n, "home_share_90": [float(lo), float(hi)],
            "p_value_vs_half": p_value,
            # Keep the 50/50 default unless the data clearly disagree
            "use_home_prob": float(wins / n) if p_value < 0.01 else 0.5}


def empty_net_table(results: pd.DataFrame) -> dict:
    """Empty-net goals by the leader, for games decided in regulation.

    Margin is the regulation margin WITHOUT empty-net goals (what Track B simulates),
    capped at 3. Returns, per margin, the probabilities of the leader adding
    0, 1, 2+ empty-net goals, plus counts of rarer cases for transparency.
    """
    reg = results[results["decided"] == "REG"].copy()
    diff = reg["home_reg"] - reg["away_reg"]
    reg["margin"] = diff.abs().clip(upper=EN_MARGINS[-1])
    reg["leader_en"] = np.where(diff > 0, reg["home_en"], reg["away_en"])
    reg["trailer_en"] = np.where(diff > 0, reg["away_en"], reg["home_en"])

    table = {}
    for m in EN_MARGINS:
        sub = reg[reg["margin"] == m]
        k = sub["leader_en"].clip(upper=EN_MAX_GOALS)
        table[str(m)] = {
            "games": int(len(sub)),
            "probs": [float((k == j).mean()) for j in range(EN_MAX_GOALS + 1)],
        }
    all_goals = results["home_reg"].sum() + results["away_reg"].sum()
    en_goals = results["home_en"].sum() + results["away_en"].sum()
    return {
        "by_margin": table,
        # rarer cases the Phase 6 step ignores
        "games_tied_without_en_but_decided_in_reg": int((reg["margin"] == 0).sum()),
        "games_with_trailer_en_goal": int((reg["trailer_en"] > 0).sum()),
        "en_share_of_regulation_goals": float(en_goals / (all_goals + en_goals)),
    }


def late_game_params(conn: sqlite3.Connection, gt: pd.DataFrame, post) -> dict:
    """All late-game parameters for Phase 6, from the fit-window games."""
    results = game_results(conn, gt)
    rates = mean_rates(post, gt)
    return {
        "games": int(len(results)),
        "shares": {k: float((results["decided"] == k).mean()) for k in ["REG", "OT", "SO"]},
        "home_win_share": float(results["home_won"].mean()),
        "ot": fit_ot_factor(results, rates),
        "shootout": shootout_home(results),
        "empty_net": empty_net_table(results),
    }


def save_params(params: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_params(path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    import arviz as az
    from config.settings import OUTPUT_DIR
    from src.data.db import get_connection
    from src.models.active_fit import read_active, active_fit_path, window_games

    # The active fit (fit_hierarchical.py) and the games it was trained on
    info = read_active()
    fit_path = active_fit_path()
    idata = az.from_netcdf(fit_path)
    idata = idata.load()
    if hasattr(idata, "close"):
        idata.close()

    conn = get_connection()
    gt, _ = window_games(conn, info)
    params = late_game_params(conn, gt, idata.posterior)
    conn.close()
    params["fit_file"] = fit_path.name
    params["seasons"] = info["seasons"]
    params["window"] = [info["start"], info["end"]]
    print(f"Active fit {fit_path.name}: {len(gt):,} games, {info['start']} to {info['end']} (end excluded)\n")

    out = OUTPUT_DIR / "models" / "late_game.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_params(params, out)

    s, ot, so, en = params["shares"], params["ot"], params["shootout"], params["empty_net"]
    print(f"{params['games']:,} games: regulation {s['REG']:.1%}, OT {s['OT']:.1%}, shootout {s['SO']:.1%}; "
          f"home won {params['home_win_share']:.1%}")
    print(f"\nOvertime: {ot['games_past_regulation']:,} games past regulation, "
          f"{ot['share_decided_in_ot']:.1%} decided in OT (fitted {ot['share_decided_in_ot_fitted']:.1%})")
    print(f"  OT factor {ot['ot_factor']:.2f} (90% {ot['ot_factor_90'][0]:.2f}-{ot['ot_factor_90'][1]:.2f}): "
          f"3-on-3 scoring runs at about {ot['ot_factor']:.1f}x the regulation rate")
    print(f"  home scored the OT goal {ot['ot_home_share']:.1%} (expected from ratings {ot['ot_home_share_expected']:.1%})")
    print(f"\nShootout: {so['shootouts']} games, home won {so['home_share']:.1%} "
          f"(90% {so['home_share_90'][0]:.1%}-{so['home_share_90'][1]:.1%}, p = {so['p_value_vs_half']:.2f} vs 50/50) "
          f"-> using {so['use_home_prob']:.2f}")
    print(f"\nEmpty net ({en['en_share_of_regulation_goals']:.1%} of regulation goals), leader's empty-net goals "
          f"by regulation margin:")
    for m, row in en["by_margin"].items():
        label = f"{m}+" if int(m) == EN_MARGINS[-1] else m
        p = row["probs"]
        print(f"  margin {label:2s}: {row['games']:5,} games   0: {p[0]:.1%}   1: {p[1]:.1%}   2+: {p[2]:.1%}")
    print(f"  ignored rare cases: {en['games_tied_without_en_but_decided_in_reg']} games decided in regulation by an "
          f"empty-net goal from a tie, {en['games_with_trailer_en_goal']} with a trailer empty-net goal")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()