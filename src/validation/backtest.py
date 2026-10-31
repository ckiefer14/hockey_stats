"""
backtest.py
===========
Walk-forward backtest of the Track B game model (v1 goals-only, or v1.1 xG + goals):
every game is predicted using only data from before it.

Home: src/validation/backtest.py

For a test season:
  1. Pre-season: fit Track B on the `window_days` before opening night; measure the
     late-game (OT, shootout) and score-state parameters on those games only.
  2. Every `refit_days`: refit Track B on the `window_days` before the refit date,
     then predict the games until the next refit.
  3. Each prediction uses the actual starting goalies with their Phase 3 ratings as of
     the game date (known by game time, as they are for closing lines), P(win) from the
     exact model, and score outputs from the score-state simulator.

Leakage: goalie ratings are as-of; xG for each season comes from a model that did not
train on that season (score_xg.py); everything else is fit on games before the refit date.

Results are appended to a CSV after every refit block, so a stopped run resumes.
"""

import os

import numpy as np
import pandas as pd

from src.processing.game_table import build_game_table, select_window
from src.processing.late_game import late_game_params
from src.processing.score_state import score_state_params
from src.models.stochastic.hierarchical import build_model, sample
from src.models.stochastic.hierarchical_xg import build_model_xg
from src.models.game_model import outcome_probs
from src.models.stochastic.montecarlo import simulate_games_score_state, calibrate_scale

N_DRAWS_PRED = 1000           # posterior draws used per prediction (of 4,000)
TOTAL_MAX = 14                # totals distribution stored for 0..14, 15+ lumped into 14
LEAGUE_TERMS = ["alpha0", "alpha_home", "rho"]


def draws_from_idata(idata) -> dict:
    """Pooled posterior draws from a fit held in memory (same layout as load_track_b)."""
    post = idata.posterior
    out = {}
    for name in LEAGUE_TERMS + ["mu_off", "mu_def"]:
        x = post[name].values
        out[name] = x.reshape(-1, *x.shape[2:])
    return out


def with_new_teams(draws: dict, teams: list[str], needed) -> tuple[dict, list[str]]:
    """Add league-average (zero) ratings for teams missing from the training window."""
    missing = sorted(set(needed) - set(teams))
    if not missing:
        return draws, teams
    d = dict(draws)
    n = d["mu_off"].shape[0]
    d["mu_off"] = np.hstack([d["mu_off"], np.zeros((n, len(missing)))])
    d["mu_def"] = np.hstack([d["mu_def"], np.zeros((n, len(missing)))])
    return d, teams + missing


def fit_window(gt_all: pd.DataFrame, end, window_days: int, seed: int,
               model: str = "goals", half_life: float | None = None):
    """Fit Track B on the window_days before `end`. Returns (idata, window_gt, teams).

    model: "goals" (v1, goals only) or "xg" (v1.1, ratings from xG + goals).
    half_life: v1.1 only; game weight 0.5 ** (age_days / half_life), age measured to `end`.
    """
    end = pd.Timestamp(end)
    w, teams = select_window(gt_all, end - pd.Timedelta(days=window_days), end)
    if model == "xg":
        m = build_model_xg(w, teams, half_life_days=half_life, ref_date=end)
    else:
        m = build_model(w, teams)
    idata = sample(m, seed=seed, progressbar=False)
    return idata, w, teams


def model_label(model: str, window_days: int, half_life: float | None) -> str:
    """Name used in result files and the notebook, e.g. trackb_w730, trackbxg_w365_hl120."""
    name = "trackbxg" if model == "xg" else "trackb"
    return f"{name}_w{window_days}" + (f"_hl{int(half_life)}" if half_life else "")


def predict_block(games: pd.DataFrame, draws: dict, teams: list[str], late: dict, ss: dict,
                  rng, score_sims: int) -> pd.DataFrame:
    """Predictions for a block of games from one fit (exact P(win) + score-state score outputs)."""
    draws, teams = with_new_teams(draws, teams, pd.concat([games["home"], games["away"]]))
    idx = {t: i for i, t in enumerate(teams)}
    hi = games["home"].map(idx).to_numpy()
    ai = games["away"].map(idx).to_numpy()
    gh, ga = games["gamma_home"].to_numpy(), games["gamma_away"].to_numpy()

    # 1. Exact outcome probabilities, one per draw (subset of draws for speed)
    sub = rng.choice(draws["alpha0"].size, min(N_DRAWS_PRED, draws["alpha0"].size), replace=False)
    a0, ah, rho = draws["alpha0"][sub, None], draws["alpha_home"][sub, None], draws["rho"][sub, None]
    off, dfn = draws["mu_off"][sub], draws["mu_def"][sub]
    lam_h = np.exp(a0 + ah + off[:, hi] - dfn[:, ai] - ga)
    lam_a = np.exp(a0 + off[:, ai] - dfn[:, hi] - gh)
    shape = lam_h.shape
    pr = outcome_probs(lam_h.ravel(), lam_a.ravel(), np.broadcast_to(rho, shape).ravel(), late)
    hw = pr["home_win"].reshape(shape)

    out = pd.DataFrame({
        "game_id": games["game_id"].to_numpy(),
        "p_home": hw.mean(axis=0),
        "p_home_lo": np.quantile(hw, 0.05, axis=0),
        "p_home_hi": np.quantile(hw, 0.95, axis=0),
        "p_tie_reg": pr["tie_reg"].reshape(shape).mean(axis=0),
        "p_reg_home": pr["reg_home"].reshape(shape).mean(axis=0),
        "p_reg_away": pr["reg_away"].reshape(shape).mean(axis=0),
        "lam_home": lam_h.mean(axis=0),
        "lam_away": lam_a.mean(axis=0),
    })

    # 2. Score outputs from the score-state simulator
    n = len(games)
    rep_h, rep_a = np.repeat(hi, score_sims), np.repeat(ai, score_sims)
    sims = simulate_games_score_state(draws, late, ss, rep_h, rep_a,
                                      np.repeat(gh, score_sims), np.repeat(ga, score_sims), rng=rng)
    fh = sims["final_home"].to_numpy().reshape(n, score_sims)
    fa = sims["final_away"].to_numpy().reshape(n, score_sims)
    total = np.minimum(fh + fa, TOTAL_MAX)
    margin = fh - fa
    out["sim_p_home"] = sims["home_win"].to_numpy().reshape(n, score_sims).mean(axis=1)
    out["p_over_5_5"] = (fh + fa > 5.5).mean(axis=1)
    out["p_over_6_5"] = (fh + fa > 6.5).mean(axis=1)
    out["p_home_m1_5"] = (margin >= 2).mean(axis=1)
    out["p_away_m1_5"] = (margin <= -2).mean(axis=1)
    out["exp_total"] = (fh + fa).mean(axis=1)
    for k in range(TOTAL_MAX + 1):
        out[f"p_total_{k}"] = (total == k).mean(axis=1)
    return out


def walk_forward(conn, season: str, window_days: int = 730, refit_days: int = 7,
                 score_sims: int = 5000, seed: int = 42, out_path=None, all_seasons=None,
                 verbose: bool = True, model: str = "goals", half_life: float | None = None) -> pd.DataFrame:
    """Walk-forward predictions for every regular-season game of `season`.

    out_path: CSV to append to after each refit block; games already in it are skipped.
    all_seasons: seasons to build the game table from (default: every season up to `season`).
    """
    # 0. Full game table once (goalie offsets are as-of each game date already)
    if all_seasons is None:
        seasons = pd.read_sql("SELECT DISTINCT season FROM schedule", conn)["season"].astype(str)
        all_seasons = sorted(s for s in seasons if s <= str(season))
    gt_all, _ = build_game_table(conn, all_seasons)
    test = gt_all[gt_all["season"].astype(str) == str(season)].sort_values(["game_date", "game_id"])
    if test.empty:
        raise ValueError(f"No games for season {season}.")
    opening = test["game_date"].min()

    done = set()
    if out_path is not None and os.path.exists(out_path):
        done = set(pd.read_csv(out_path, usecols=["game_id"])["game_id"])
        if verbose:
            print(f"Resuming: {len(done):,} games already in {out_path}")

    # 1. Pre-season fit: late-game and score-state parameters from prior games only
    if verbose:
        print(f"Pre-season fit: {window_days} days before {opening.date()}")
    idata0, w0, _ = fit_window(gt_all, opening, window_days, seed, model, half_life)
    late = late_game_params(conn, w0, idata0.posterior)
    ss = score_state_params(conn, w0, idata0.posterior)
    ss["scale"] = calibrate_scale(draws_from_idata(idata0), None, ss, w0)
    if verbose:
        print(f"  {len(w0):,} games; OT factor {late['ot']['ot_factor']:.2f}, "
              f"'tied late' {ss['state_factor']['tied late']:.2f}, scale {ss['scale']:.3f}")

    # 2. Weekly refits
    rng = np.random.default_rng(seed)
    refit = opening.normalize()
    last = test["game_date"].max()
    block_no = 0
    results = []
    while refit <= last:
        nxt = refit + pd.Timedelta(days=refit_days)
        block = test[(test["game_date"] >= refit) & (test["game_date"] < nxt)]
        block = block[~block["game_id"].isin(done)]
        if len(block):
            block_no += 1
            idata, w, teams = (idata0, w0, sorted(set(w0["home"]) | set(w0["away"]))) if refit == opening.normalize() \
                else fit_window(gt_all, refit, window_days, seed + block_no, model, half_life)
            pred = predict_block(block, draws_from_idata(idata), teams, late, ss, rng, score_sims)
            pred = block[["game_id", "game_date", "season", "home", "away", "home_goalie", "away_goalie",
                          "gamma_home", "gamma_away", "home_goals_reg", "away_goals_reg"]].merge(pred, on="game_id")
            pred.insert(0, "model", model_label(model, window_days, half_life))
            pred["refit_date"] = refit.date()
            pred["train_games"] = len(w)
            divs = int(idata.sample_stats["diverging"].values.sum())
            pred["divergences"] = divs
            results.append(pred)
            if out_path is not None:
                pred.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))
            if verbose:
                print(f"  {refit.date()}: {len(block):3d} games, trained on {len(w):,}"
                      + (f", {divs} divergences" if divs else ""))
        refit = nxt
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()