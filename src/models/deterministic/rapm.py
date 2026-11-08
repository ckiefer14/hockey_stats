"""
rapm.py
=======
v2.2 reference: ridge RAPM (regularized adjusted plus-minus) on 5v5 stints, xG version.

Home: src/models/deterministic/rapm.py

Each 5v5 stint (both goalies in net) gives two rows, one per team on offense:

    xGF/60 of the offense = intercept + sum over the offense's 5 skaters of O_p
                                      + sum over the defense's 5 skaters of D_q
                                      + home (offense is the home team)
                                      + score state of the offense (-2 or less, -1, +1, +2 or more; tied = 0)
                                      + zone start of the offense (O, D, N faceoff; on-the-fly = 0)

weighted by stint length (minutes). O_p > 0: the team creates more xG with p on the ice.
D_q > 0: the opponent creates MORE xG with q on the ice, so reported defense = -D_q
(positive = good). Net = O - D_q: xG/60 differential impact. Player terms get the ridge
penalty lambda * sum(beta^2); intercept and covariates are not penalized. Players with less
than min_toi minutes in the window share one "other" offense and one "other" defense column.

Solved from the normal equations (X'WX + lambda P) beta = X'Wy with a sparse design, which is
exactly what scikit-learn's Ridge does with sample weights, but lets cross-validation reuse the
per-fold X'WX instead of refitting from scratch for every lambda. Lambda is chosen by K-fold
cross-validation grouped by game (no game in both training and test).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

SCORE_BUCKETS = {-2: "trail2+", -1: "trail1", 1: "lead1", 2: "lead2+"}
ZONES = {"O": "ozone", "D": "dzone", "N": "nzone"}
COVARIATES = ["home"] + [f"score_{v}" for v in SCORE_BUCKETS.values()] + [f"zone_{v}" for v in ZONES.values()]


def load_stints(conn, start, end) -> pd.DataFrame:
    """5v5 stints with both goalies in net, games from start (inclusive) to end (exclusive)."""
    st = pd.read_sql("""
        SELECT s.game_id, sc.game_date, sc.season, sc.home_team_abbrev AS home_team, sc.away_team_abbrev AS away_team,
               s.duration, s.home_skaters, s.away_skaters, s.score_diff,
               s.zone_start, s.home_xg, s.away_xg, s.home_goals, s.away_goals
        FROM   stints s JOIN schedule sc ON sc.game_id = s.game_id
        WHERE  s.n_home = 5 AND s.n_away = 5 AND s.home_goalie IS NOT NULL AND s.away_goalie IS NOT NULL
          AND  s.duration > 0 AND sc.game_date >= ? AND sc.game_date < ?
    """, conn, params=(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())), parse_dates=["game_date"])
    return st


def _rows(st: pd.DataFrame) -> pd.DataFrame:
    """Two offense rows per stint."""
    flip = {"O": "D", "D": "O"}
    h = pd.DataFrame({"game_id": st["game_id"], "minutes": st["duration"] / 60, "off": st["home_skaters"],
                      "def": st["away_skaters"], "xg": st["home_xg"], "home": 1, "score": st["score_diff"],
                      "zone": st["zone_start"]})
    a = pd.DataFrame({"game_id": st["game_id"], "minutes": st["duration"] / 60, "off": st["away_skaters"],
                      "def": st["home_skaters"], "xg": st["away_xg"], "home": 0, "score": -st["score_diff"],
                      "zone": st["zone_start"].map(lambda z: flip.get(z, z))})
    return pd.concat([h, a], ignore_index=True)


def design(st: pd.DataFrame, min_toi: float = 100.0):
    """Sparse design for the stints. Returns (X, y, w, groups, info) where info has the player
    list, column names, penalty mask and each player's 5v5 minutes in the window."""
    r = _rows(st)
    offs = r["off"].str.split(",")
    defs = r["def"].str.split(",")
    # minutes per player (on ice as offense; every player appears once per stint as offense)
    lens = offs.str.len().to_numpy()
    pl = np.concatenate(offs.to_numpy()).astype(np.int64)
    mins = pd.Series(np.repeat(r["minutes"].to_numpy(), lens)).groupby(pl).sum()
    players = np.sort(mins[mins >= min_toi].index.to_numpy())
    pidx = pd.Series(np.arange(len(players)), index=players)
    P = len(players)
    n = len(r)
    # columns: [O_players | O_other | D_players | D_other | intercept | covariates]
    c_ooth, c_dbase = P, P + 1
    c_doth = 2 * P + 1
    c_int = 2 * P + 2
    c_cov = {name: c_int + 1 + i for i, name in enumerate(COVARIATES)}
    K = c_int + 1 + len(COVARIATES)

    rows, cols = [], []
    for series, base, other in ((offs, 0, c_ooth), (defs, c_dbase, c_doth)):
        ln = series.str.len().to_numpy()
        ids = np.concatenate(series.to_numpy()).astype(np.int64)
        ri = np.repeat(np.arange(n), ln)
        k = pidx.reindex(ids).to_numpy()
        ci = np.where(np.isnan(k), other, base + np.nan_to_num(k).astype(int))
        rows.append(ri); cols.append(ci.astype(int))
    rows.append(np.arange(n)); cols.append(np.full(n, c_int))
    hm = np.nonzero(r["home"].to_numpy() == 1)[0]
    rows.append(hm); cols.append(np.full(len(hm), c_cov["home"]))
    sc = r["score"].clip(-2, 2).to_numpy()
    for v, name in SCORE_BUCKETS.items():
        m = np.nonzero(sc == v)[0]; rows.append(m); cols.append(np.full(len(m), c_cov[f"score_{name}"]))
    z = r["zone"].to_numpy()
    for v, name in ZONES.items():
        m = np.nonzero(z == v)[0]; rows.append(m); cols.append(np.full(len(m), c_cov[f"zone_{name}"]))
    ri, ci = np.concatenate(rows), np.concatenate(cols)
    X = sp.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(n, K))
    w = r["minutes"].to_numpy()
    y = r["xg"].to_numpy() / w * 60.0                      # xG per 60
    pen = np.zeros(K, bool)
    pen[:c_int] = True                                      # players and "other" columns
    names = ([f"O_{p}" for p in players] + ["O_other"] + [f"D_{p}" for p in players] + ["D_other", "intercept"]
             + COVARIATES)
    info = {"players": players, "names": names, "penalty": pen, "P": P, "minutes": mins.reindex(players).to_numpy(),
            "n_rows": n, "c_int": c_int}
    return X, y, w, r["game_id"].to_numpy(), info


def gram(X, y, w, idx=None):
    """X'WX (dense), X'Wy, y'Wy, sum w for the rows idx (all rows if None)."""
    if idx is not None:
        X, y, w = X[idx], y[idx], w[idx]
    Xw = X.multiply(w[:, None]).tocsr()
    return (X.T @ Xw).toarray(), np.asarray(X.T @ (w * y)).ravel(), float(np.sum(w * y * y)), float(w.sum())


def solve(A, b, lam: float, pen: np.ndarray, ridge_floor: float = 1e-8) -> np.ndarray:
    M = A.copy()
    M[np.diag_indices_from(M)] += np.where(pen, lam, ridge_floor)
    return cho_solve(cho_factor(M, lower=True), b)


def weighted_sse(beta, A, b, yy):
    """sum w (y - X beta)^2 from the Gram pieces of the evaluation rows."""
    return float(yy - 2 * beta @ b + beta @ A @ beta)


def cross_validate(X, y, w, groups, pen, lams, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Held-out weighted MSE (per minute of play) for each lambda, folds grouped by game.
    Also the covariates-only model (players removed) as the baseline."""
    rng = np.random.default_rng(seed)
    games = np.unique(groups)
    fold_of_game = pd.Series(rng.integers(0, folds, len(games)), index=games)
    f = fold_of_game.reindex(groups).to_numpy()
    parts = [gram(X, y, w, np.nonzero(f == k)[0]) for k in range(folds)]
    A_all = sum(p[0] for p in parts); b_all = sum(p[1] for p in parts)
    keep = ~pen                                            # baseline: intercept + covariates only
    out = []
    for lam in list(lams) + ["baseline"]:
        sse = tot_w = 0.0
        for k in range(folds):
            A_te, b_te, yy_te, w_te = parts[k]
            A_tr, b_tr = A_all - A_te, b_all - b_te
            if lam == "baseline":
                beta = np.zeros(len(b_tr))
                beta[keep] = solve(A_tr[np.ix_(keep, keep)], b_tr[keep], 0.0, np.zeros(keep.sum(), bool))
            else:
                beta = solve(A_tr, b_tr, float(lam), pen)
            sse += weighted_sse(beta, A_te, b_te, yy_te)
            tot_w += w_te
        out.append({"lambda": lam, "heldout_mse": sse / tot_w})
    return pd.DataFrame(out)


def fit(X, y, w, pen, lam):
    A, b, _, _ = gram(X, y, w)
    return solve(A, b, lam, pen)


def player_table(beta, info) -> pd.DataFrame:
    P = info["P"]
    o = beta[:P]
    d = beta[P + 1:2 * P + 1]
    return pd.DataFrame({"player_id": info["players"], "toi_5v5": info["minutes"], "off": o, "def": -d,
                         "net": o - d})


# ---------------------------------------------------------------------------
# Synthetic stints (recovery test)
# ---------------------------------------------------------------------------

def simulate_stints(n_teams=32, games_per_team=164, seed=0, sd_off=0.25, sd_def=0.20, base=2.4,
                    sd_d_off=None, sd_d_def=None, team_sd_off=0.0, team_sd_def=0.0):
    """Fake 5v5 stints with known player effects (xG/60), realistic deployment: each team has
    4 forward lines of 3 and 3 defence pairs, lines mix 10% of the time, 2-6 stints per line
    change pattern. Returns (stints DataFrame shaped like load_stints, truth DataFrame)."""
    rng = np.random.default_rng(seed)
    roster = {}
    pid = 1000
    for t in range(n_teams):
        fl = [list(range(pid + 3 * i, pid + 3 * i + 3)) for i in range(4)]; pid += 12
        dp = [list(range(pid + 2 * i, pid + 2 * i + 2)) for i in range(3)]; pid += 6
        roster[t] = (fl, dp)
    all_p = np.arange(1000, pid)
    is_d = np.zeros(len(all_p), bool)
    team_of = np.zeros(len(all_p), int)
    for t, (fl, dp) in roster.items():
        for line in fl:
            team_of[np.array(line) - 1000] = t
        for pair in dp:
            is_d[np.array(pair) - 1000] = True
            team_of[np.array(pair) - 1000] = t
    so = np.where(is_d, sd_d_off if sd_d_off is not None else sd_off, sd_off)
    sdf = np.where(is_d, sd_d_def if sd_d_def is not None else sd_def, sd_def)
    truth = pd.DataFrame({"player_id": all_p, "off": rng.normal(0, 1, len(all_p)) * so,
                          "def": rng.normal(0, 1, len(all_p)) * sdf, "position": np.where(is_d, "D", "C"),
                          "team": [f"T{t}" for t in team_of]})
    t_off, t_def = rng.normal(0, 1, n_teams) * team_sd_off, rng.normal(0, 1, n_teams) * team_sd_def
    o_eff = dict(zip(truth["player_id"], truth["off"]))
    d_eff = dict(zip(truth["player_id"], truth["def"]))
    shots_per_60 = 40.0                                     # unblocked 5v5 shots per team per 60
    rows = []
    gid = 0
    line_w, pair_w = np.array([.35, .3, .22, .13]), np.array([.4, .35, .25])
    for g in range(n_teams * games_per_team // 2):
        h, a = rng.choice(n_teams, 2, replace=False)
        gid += 1
        for k in range(rng.integers(40, 60)):
            def lineup(t):
                fl, dp = roster[t]
                f = list(fl[rng.choice(4, p=line_w)]); d = list(dp[rng.choice(3, p=pair_w)])
                if rng.random() < 0.1:                       # mixed lines now and then
                    f[rng.integers(3)] = fl[rng.integers(4)][rng.integers(3)]
                return sorted(set(f + d))
            hs, as_ = lineup(h), lineup(a)
            if len(hs) != 5 or len(as_) != 5:
                continue
            dur = float(rng.uniform(10, 90))
            rh = max(base + 0.15 + t_off[h] - t_def[a] + sum(o_eff[p] for p in hs) - sum(d_eff[p] for p in as_), 0.2)
            ra = max(base + t_off[a] - t_def[h] + sum(o_eff[p] for p in as_) - sum(d_eff[p] for p in hs), 0.2)

            def xg_of(rate):                                  # shots ~ Poisson, each shot's xG ~ Gamma
                n = rng.poisson(shots_per_60 * dur / 3600)
                return float(rng.gamma(0.7, rate / shots_per_60 / 0.7, n).sum()) if n else 0.0
            rows.append({"game_id": gid, "game_date": pd.Timestamp("2025-10-01") + pd.Timedelta(days=g // 8),
                         "season": "20252026", "home_team": f"T{h}", "away_team": f"T{a}",
                         "duration": dur, "home_skaters": ",".join(map(str, hs)), "away_skaters": ",".join(map(str, as_)),
                         "score_diff": 0, "zone_start": None, "home_xg": xg_of(rh), "away_xg": xg_of(ra),
                         "home_goals": 0, "away_goals": 0})
    truth.attrs["team_off"] = dict(zip([f"T{t}" for t in range(n_teams)], t_off))
    truth.attrs["team_def"] = dict(zip([f"T{t}" for t in range(n_teams)], t_def))
    return pd.DataFrame(rows), truth