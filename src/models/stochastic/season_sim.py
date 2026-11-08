"""
season_sim.py
=============
Simulate the rest of the regular season from the game model's posterior, for playoff odds.

Home: src/models/stochastic/season_sim.py

One simulated season = one posterior draw used for EVERY remaining game (a team that is
better than its average rating in that draw is better all season), plus rating drift: each
team's offense and defense keep moving week by week at the fit's own weekly step size
(tau_off, tau_def of that draw; zero-sum across teams), so far-off games are more
uncertain than tomorrow's. Each game's outcome (regulation / overtime / shootout, either
side) is drawn from the exact probabilities of game_model.outcome_probs for that draw's
scoring rates.

Standings rules (NHL): win 2 points, overtime or shootout loss 1. Ranking within a group:
points, then regulation wins (RW), then regulation + overtime wins (ROW), then wins, then a
coin flip (head-to-head and goal differential are not simulated). Playoffs: top 3 of each
division + the 2 best remaining teams of each conference (wild cards).
"""

import numpy as np
import pandas as pd

from src.models.game_model import outcome_probs

OUTCOMES = ["reg_home", "ot_home", "so_home", "reg_away", "ot_away", "so_away"]
DEFAULT_TAU = 0.012             # weekly rating step if the fit has none


def current_standings(conn, season: str, teams_info: pd.DataFrame, franchise: dict | None = None) -> pd.DataFrame:
    """Standings from the season's played regular-season games in the database.
    Columns: team, gp, w, l, otl, pts, rw, row, gf, ga."""
    g = pd.read_sql("""SELECT home_team_abbrev AS home, away_team_abbrev AS away, home_score, away_score,
                              period_type FROM games WHERE game_type = 2 AND season = ?
                              AND home_score IS NOT NULL AND away_score IS NOT NULL""", conn, params=(int(season),))
    franchise = franchise or {}
    g[["home", "away"]] = g[["home", "away"]].replace(franchise)
    rows = {t: dict(team=t, gp=0, w=0, l=0, otl=0, pts=0, rw=0, row=0, gf=0, ga=0) for t in teams_info["team"]}
    for r in g.itertuples():
        hw = r.home_score > r.away_score
        for team, won, gf, ga in ((r.home, hw, r.home_score, r.away_score), (r.away, not hw, r.away_score, r.home_score)):
            if team not in rows:
                continue
            x = rows[team]
            x["gp"] += 1; x["gf"] += gf; x["ga"] += ga
            if won:
                x["w"] += 1; x["pts"] += 2
                x["rw"] += r.period_type == "REG"
                x["row"] += r.period_type in ("REG", "OT")
            elif r.period_type in ("OT", "SO"):
                x["otl"] += 1; x["pts"] += 1
            else:
                x["l"] += 1
    return pd.DataFrame(rows.values())


def rank_key(pts, rw, row, w, tie):
    """One sortable number per team: points, then RW, ROW, wins, coin flip (higher = better)."""
    return (((pts * 100 + rw) * 100 + row) * 100 + w) + tie


def simulate_season(draws: dict, teams: list[str], late: dict, remaining: pd.DataFrame,
                    standings: pd.DataFrame, info: pd.DataFrame, gamma: dict, today,
                    n_sims: int = 10_000, chunk: int = 1000, seed: int = 0, drift: bool = True) -> dict:
    """Simulate the remaining games n_sims times.

    remaining: game_id, game_date, home, away, shift_home, shift_away (known log-rate shifts:
               rest / back-to-backs from context_shift; 0 if none)
    standings: current_standings(...); info: team, division, conference; gamma: team -> goalie offset
    Returns {"teams": [...], "pts": (n_sims, T), "playoffs", "div_win", "div_top3", "wildcard",
             "conf_first", "presidents": boolean (n_sims, T) arrays}.
    """
    rng = np.random.default_rng(seed)
    T_list = list(info["team"])
    T = len(T_list)
    tix = {t: i for i, t in enumerate(T_list)}
    mi = np.array([teams.index(t) for t in T_list])                 # model index of each team
    n_draws = len(draws["rho"])
    h_ix = remaining["home"].map(tix).to_numpy()
    a_ix = remaining["away"].map(tix).to_numpy()
    week = ((pd.to_datetime(remaining["game_date"]) - pd.Timestamp(today)).dt.days // 7).clip(lower=0).to_numpy()
    n_weeks = int(week.max()) + 1 if len(week) else 1
    g_h = np.array([gamma.get(t, 0.0) for t in remaining["home"]])
    g_a = np.array([gamma.get(t, 0.0) for t in remaining["away"]])
    sh_h = remaining.get("shift_home", pd.Series(0.0, index=remaining.index)).to_numpy(float)
    sh_a = remaining.get("shift_away", pd.Series(0.0, index=remaining.index)).to_numpy(float)
    tau_o = np.asarray(draws.get("tau_off", np.full(n_draws, DEFAULT_TAU)), float).reshape(-1)
    tau_d = np.asarray(draws.get("tau_def", np.full(n_draws, DEFAULT_TAU)), float).reshape(-1)
    base = {k: standings.set_index("team").reindex(T_list)[k].fillna(0).to_numpy() for k in ("pts", "rw", "row", "w", "gp")}

    out = {k: [] for k in ("pts", "rw", "row", "w")}
    for start in range(0, n_sims, chunk):
        C = min(chunk, n_sims - start)
        d = rng.integers(0, n_draws, C)                                # one posterior draw per simulated season
        mu_o = draws["mu_off"][d][:, mi]                               # (C, T)
        mu_d = draws["mu_def"][d][:, mi]
        a0, ah, rho = draws["alpha0"][d], draws["alpha_home"][d], draws["rho"][d]
        if drift:                                                      # weekly random-walk steps, zero-sum
            so = rng.standard_normal((C, n_weeks, T)) * tau_o[d][:, None, None]
            sd = rng.standard_normal((C, n_weeks, T)) * tau_d[d][:, None, None]
            so -= so.mean(axis=2, keepdims=True); sd -= sd.mean(axis=2, keepdims=True)
            do, dd = np.cumsum(so, axis=1), np.cumsum(sd, axis=1)      # drift by week, (C, W, T)
            do = np.concatenate([np.zeros((C, 1, T)), do[:, :-1]], axis=1)   # week 0: today's ratings
            dd = np.concatenate([np.zeros((C, 1, T)), dd[:, :-1]], axis=1)
        pts, rw, row, w = (np.tile(base[k], (C, 1)).astype(float) for k in ("pts", "rw", "row", "w"))
        for gi in range(len(remaining)):
            h, a, k = h_ix[gi], a_ix[gi], week[gi]
            xo_h = mu_o[:, h] - mu_d[:, a]
            xo_a = mu_o[:, a] - mu_d[:, h]
            if drift:
                xo_h = xo_h + do[:, k, h] - dd[:, k, a]
                xo_a = xo_a + do[:, k, a] - dd[:, k, h]
            lam_h = np.exp(a0 + ah + xo_h - g_a[gi] + sh_h[gi])
            lam_a = np.exp(a0 + xo_a - g_h[gi] + sh_a[gi])
            p = outcome_probs(lam_h, lam_a, rho, late)
            cum = np.cumsum(np.stack([p[o] for o in OUTCOMES], axis=1), axis=1)
            cum /= cum[:, -1:]
            o = (rng.random((C, 1)) > cum).sum(axis=1)                 # 0..5 = OUTCOMES index
            home_won = o < 3
            win_t = np.where(home_won, h, a)
            lose_t = np.where(home_won, a, h)
            r_ = np.arange(C)
            pts[r_, win_t] += 2
            w[r_, win_t] += 1
            reg = (o == 0) | (o == 3)
            ot = (o == 1) | (o == 4)
            rw[r_[reg], win_t[reg]] += 1
            row[r_[reg | ot], win_t[reg | ot]] += 1
            pts[r_[~reg], lose_t[~reg]] += 1                          # OT / SO loser point
        for k, v in (("pts", pts), ("rw", rw), ("row", row), ("w", w)):
            out[k].append(v)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update(qualify(res, info, rng))
    res["teams"] = T_list
    res["current"] = base
    return res


def qualify(res: dict, info: pd.DataFrame, rng) -> dict:
    """Playoff spots per simulated season (top 3 per division + 2 wild cards per conference)."""
    pts, rw, row, w = res["pts"], res["rw"], res["row"], res["w"]
    n, T = pts.shape
    key = rank_key(pts, rw, row, w, rng.random((n, T)) * 0.5)
    div = info["division"].to_numpy()
    conf = info["conference"].to_numpy()
    out = {k: np.zeros((n, T), bool) for k in ("playoffs", "div_win", "div_top3", "wildcard", "conf_first", "presidents")}
    for dv in np.unique(div):
        cols = np.nonzero(div == dv)[0]
        order = np.argsort(-key[:, cols], axis=1)                      # best first
        ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(len(cols))[None, :], axis=1)
        out["div_top3"][:, cols] = ranks < 3
        out["div_win"][:, cols] = ranks == 0
    for cf in np.unique(conf):
        cols = np.nonzero(conf == cf)[0]
        k2 = np.where(out["div_top3"][:, cols], -np.inf, key[:, cols])  # wild cards: best of the rest
        order = np.argsort(-k2, axis=1)
        ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(len(cols))[None, :], axis=1)
        out["wildcard"][:, cols] = (ranks < 2) & ~out["div_top3"][:, cols]
        best = np.argmax(key[:, cols], axis=1)
        out["conf_first"][np.arange(n), cols[best]] = True
    out["presidents"][np.arange(n), np.argmax(key, axis=1)] = True
    out["playoffs"] = out["div_top3"] | out["wildcard"]
    return out


def summarize(res: dict, info: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    T = res["teams"]
    pts = res["pts"]
    tab = pd.DataFrame({"team": T, "division": info["division"].to_numpy(), "conference": info["conference"].to_numpy(),
                        "points_now": res["current"]["pts"], "gp_now": res["current"]["gp"],
                        "points_mean": pts.mean(axis=0), "points_lo": np.quantile(pts, 0.05, axis=0),
                        "points_hi": np.quantile(pts, 0.95, axis=0)})
    for k in ("playoffs", "div_win", "div_top3", "wildcard", "conf_first", "presidents"):
        tab[f"p_{k}"] = res[k].mean(axis=0)
    return tab
