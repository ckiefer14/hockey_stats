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
TAU_SCALE = 2.0                 # multiplies the fit's weekly step for season simulations (set from
                                # backtest_playoff_odds --tau-scales; see README v2.8)


def current_standings(conn, season: str, teams_info: pd.DataFrame, franchise: dict | None = None,
                      before=None) -> pd.DataFrame:
    """Standings from the season's played regular-season games in the database (only games
    before the date `before`, if given: standings as of that morning).
    Columns: team, gp, w, l, otl, pts, rw, row, gf, ga."""
    q = """SELECT home_team_abbrev AS home, away_team_abbrev AS away, home_score, away_score,
                  period_type FROM games WHERE game_type = 2 AND season = ?
                  AND home_score IS NOT NULL AND away_score IS NOT NULL"""
    params = [int(season)]
    if before is not None:
        q += " AND game_date < ?"
        params.append(str(pd.Timestamp(before).date()))
    g = pd.read_sql(q, conn, params=params)
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
                    n_sims: int = 10_000, chunk: int = 1000, seed: int = 0, drift: bool = True,
                    tau_scale: float | None = None) -> dict:
    """Simulate the remaining games n_sims times.

    tau_scale: multiplies the weekly drift (default TAU_SCALE): the fit's step size describes
               week-to-week rating changes within its window; over a season, injuries, trades and
               coaching changes add more than that (backtest_playoff_odds measures how much).
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
    k = TAU_SCALE if tau_scale is None else tau_scale
    tau_o = k * np.asarray(draws.get("tau_off", np.full(n_draws, DEFAULT_TAU)), float).reshape(-1)
    tau_d = k * np.asarray(draws.get("tau_def", np.full(n_draws, DEFAULT_TAU)), float).reshape(-1)
    base = {k: standings.set_index("team").reindex(T_list)[k].fillna(0).to_numpy() for k in ("pts", "rw", "row", "w", "gp")}

    out = {k: [] for k in ("pts", "rw", "row", "w", "off_end", "def_end", "a0", "ah", "rho", "tau_o", "tau_d")}
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
        # ratings at the end of the regular season (for the playoff bracket)
        out["off_end"].append(mu_o + (do[:, -1] + so[:, -1] if drift else 0.0))
        out["def_end"].append(mu_d + (dd[:, -1] + sd[:, -1] if drift else 0.0))
        for k, v in (("a0", a0), ("ah", ah), ("rho", rho), ("tau_o", tau_o[d]), ("tau_d", tau_d[d])):
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
    out["key"] = key
    out["div_rank"] = np.zeros((n, T), int)                            # 0 = division winner
    out["wc_rank"] = np.full((n, T), 99, int)                          # 0 = first wild card
    for dv in np.unique(div):
        cols = np.nonzero(div == dv)[0]
        order = np.argsort(-key[:, cols], axis=1)                      # best first
        ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(len(cols))[None, :], axis=1)
        out["div_top3"][:, cols] = ranks < 3
        out["div_win"][:, cols] = ranks == 0
        out["div_rank"][:, cols] = ranks
    for cf in np.unique(conf):
        cols = np.nonzero(conf == cf)[0]
        k2 = np.where(out["div_top3"][:, cols], -np.inf, key[:, cols])  # wild cards: best of the rest
        order = np.argsort(-k2, axis=1)
        ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(len(cols))[None, :], axis=1)
        out["wildcard"][:, cols] = (ranks < 2) & ~out["div_top3"][:, cols]
        out["wc_rank"][:, cols] = np.where(out["div_top3"][:, cols], 99, ranks)
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
    for k in ("round2", "conf_final", "final", "cup"):
        if k in res:
            tab[f"p_{k}"] = res[k].mean(axis=0)
    return tab


# ---------------------------------------------------------------------------
# Stanley Cup playoffs
# ---------------------------------------------------------------------------
# NHL bracket (since 2014): in each conference the division winner with the better record plays
# the second wild card, the other division winner the first wild card; 2nd vs 3rd in each
# division. Round 2 stays inside each division's half of the bracket, then the conference final
# and the Cup Final. Home ice: the higher seed in rounds 1-2 (division winner, then 2nd, then 3rd,
# then wild card), the better regular-season record in the conference final and the Final.
# Games: best of 7, 2-2-1-1-1; a game tied after regulation goes to sudden-death overtime with
# no shootout, won by the team that scores first (P = its share of the combined scoring rate).
# Ratings: each team's end-of-season rating in that simulated season, drifting ROUND_WEEKS weeks
# per round at the fit's weekly step size.

ROUND_WEEKS = 2.0
HOME_PATTERN = np.array([1, 1, 0, 0, 1, 0, 1], bool)                 # games at the home-ice team


def playoff_game_prob(lam_home, lam_away, rho) -> np.ndarray:
    """P(home team wins) with sudden-death overtime (no shootout)."""
    from src.models.game_model import regulation_table
    t = regulation_table(lam_home, lam_away, rho)
    reg_home = (np.tril(np.ones(t.shape[1:]), -1)[None] * t).sum(axis=(1, 2))
    tie = np.einsum("nkk->n", t)
    return reg_home + tie * lam_home / (lam_home + lam_away)


def series_prob(p_home: np.ndarray, p_road: np.ndarray) -> np.ndarray:
    """P(the home-ice team wins a best-of-7, 2-2-1-1-1). p_home / p_road: its single-game win
    probabilities at home and on the road (arrays)."""
    # dp[w, l] = P(reaching w wins, l losses)
    dp = {(0, 0): np.ones_like(p_home)}
    win = np.zeros_like(p_home)
    for g in range(7):
        p = p_home if HOME_PATTERN[g] else p_road
        nxt = {}
        for (w, l), v in dp.items():
            if w + l != g:
                continue
            for dw, pr in ((1, p), (0, 1 - p)):
                w2, l2 = w + dw, l + (1 - dw)
                if w2 == 4:
                    win = win + v * pr
                elif l2 < 4:
                    nxt[(w2, l2)] = nxt.get((w2, l2), 0) + v * pr
        dp.update(nxt)
    return win


def simulate_playoffs(res: dict, info: pd.DataFrame, gamma: dict, late: dict, rng, drift: bool = True) -> dict:
    """Play the bracket once per simulated season. Adds boolean (n, T) arrays round2 (won
    round 1), conf_final (won round 2), final (won the conference), cup to res."""
    n, T = res["pts"].shape
    off = res["off_end"].copy()
    dfn = res["def_end"].copy()
    a0, ah, rho = res["a0"], res["ah"], res["rho"]
    g = np.array([gamma.get(t, 0.0) for t in res["teams"]])
    div = info["division"].to_numpy()
    conf = info["conference"].to_numpy()
    seed = np.where(res["div_rank"] < 3, res["div_rank"] + 1, 4)       # 1 = division winner ... 4 = wild card
    key = res["key"]
    rows = np.arange(n)
    out = {k: np.zeros((n, T), bool) for k in ("round2", "conf_final", "final", "cup")}

    def step():
        if not drift:
            return
        s = np.sqrt(ROUND_WEEKS)
        for arr, tau in ((off, res["tau_o"]), (dfn, res["tau_d"])):
            z = rng.standard_normal((n, T))
            z -= z.mean(axis=1, keepdims=True)
            arr += z * tau[:, None] * s

    def play(a, b, by_seed: bool):
        """Winner of a vs b per simulation (team index arrays)."""
        if by_seed:
            a_home = (seed[rows, a] < seed[rows, b]) | ((seed[rows, a] == seed[rows, b]) & (key[rows, a] > key[rows, b]))
        else:
            a_home = key[rows, a] > key[rows, b]
        h = np.where(a_home, a, b)
        r = np.where(a_home, b, a)
        oh, orr, dh, dr = off[rows, h], off[rows, r], dfn[rows, h], dfn[rows, r]
        lam_hh = np.exp(a0 + ah + oh - dr - g[r])                        # h at home
        lam_ha = np.exp(a0 + orr - dh - g[h])
        lam_rh = np.exp(a0 + ah + orr - dh - g[h])                       # h on the road
        lam_ra = np.exp(a0 + oh - dr - g[r])
        p_home = playoff_game_prob(lam_hh, lam_ha, rho)
        p_road = 1 - playoff_game_prob(lam_rh, lam_ra, rho)
        h_wins = rng.random(n) < series_prob(p_home, p_road)
        return np.where(h_wins, h, r)

    def team_at(mask):
        """Index of the one team per simulation where mask (n, T) is True."""
        return np.argmax(mask, axis=1)

    champs = []
    for cf in np.unique(conf):
        divs = list(np.unique(div[conf == cf]))
        if len(divs) != 2:
            raise ValueError(f"Bracket needs 2 divisions per conference; {cf} has {len(divs)}")
        in_cf = conf == cf
        if (res["wildcard"][:, in_cf].sum(axis=1) != 2).any() or \
                any((res["div_rank"][:, div == dv] <= 2).sum(axis=1).min() < 3 for dv in divs):
            raise ValueError(f"Bracket needs 3 teams per division and 2 wild cards in {cf}")
        wc1 = team_at(in_cf[None] & (res["wc_rank"] == 0) & res["wildcard"])
        wc2 = team_at(in_cf[None] & (res["wc_rank"] == 1) & res["wildcard"])
        d1 = {dv: team_at((div == dv)[None] & (res["div_rank"] == 0)) for dv in divs}
        d2 = {dv: team_at((div == dv)[None] & (res["div_rank"] == 1)) for dv in divs}
        d3 = {dv: team_at((div == dv)[None] & (res["div_rank"] == 2)) for dv in divs}
        a_better = key[rows, d1[divs[0]]] > key[rows, d1[divs[1]]]
        wc_vs = {divs[0]: np.where(a_better, wc2, wc1), divs[1]: np.where(a_better, wc1, wc2)}
        step()
        r1 = {dv: (play(d1[dv], wc_vs[dv], True), play(d2[dv], d3[dv], True)) for dv in divs}
        for dv in divs:
            for w in r1[dv]:
                out["round2"][rows, w] = True
        step()
        r2 = {dv: play(r1[dv][0], r1[dv][1], True) for dv in divs}
        for w in r2.values():
            out["conf_final"][rows, w] = True
        step()
        champ = play(r2[divs[0]], r2[divs[1]], False)
        out["final"][rows, champ] = True
        champs.append(champ)
    if len(champs) == 2:
        step()
        cup = play(champs[0], champs[1], False)
        out["cup"][rows, cup] = True
    res.update(out)
    return res