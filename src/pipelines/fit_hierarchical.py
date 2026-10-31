"""
fit_hierarchical.py
===================
Fit Track B on a training window, save the posterior and make it the active fit.

Home: src/pipelines/fit_hierarchical.py

Usage (from the project root):
    py -m src.pipelines.fit_hierarchical                      # v1.1, last 365 days of data (Phase 7 choice)
    py -m src.pipelines.fit_hierarchical --end 2026-12-01     # window ends the day before this date
    py -m src.pipelines.fit_hierarchical --model goals --window 730   # Track B v1, for comparison
    py -m src.pipelines.fit_hierarchical --overwrite

Model and window match the Phase 7 backtest: the fit uses games with
end - window <= game_date < end. Default end: the day after the last game in the database.

Writes to output/posteriors/ (label as in the backtest, e.g. trackbxg_w365; date = end):
    <label>_<YYYYMMDD>.nc            posterior + observed data (ArviZ NetCDF)
    <label>_<YYYYMMDD>_meta.json     settings, window, diagnostics, league terms
    <label>_<YYYYMMDD>_ratings.csv   one row per team: offense, defense, net (goal scale, mean, 90%)
    <label>_<YYYYMMDD>_trace.png     trace plot of the league terms (chain 1)
    active_fit.json                  points late_game, score_state and predict_game at this fit

Stops if any loaded game in the window (or after it, with the default end) has no xG
yet: run py -m src.pipelines.score_xg --season <season> first.

Then rebuild, in order:
    py -m src.processing.late_game
    py -m src.processing.score_state
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm

from src.data.db import get_connection
from src.processing.game_table import build_game_table, select_window
from src.models.stochastic import hierarchical as h1
from src.models.stochastic import hierarchical_xg as hx
from src.models.stochastic.hierarchical import (
    GOAL_COLS, DRAWS, TUNE, CHAINS, TARGET_ACCEPT, build_model, sample, team_ratings,
)
from src.models.stochastic.hierarchical_xg import build_model_xg, XG_COLS
from src.models.stochastic.am_sampler import plot_traces
from src.models.active_fit import (
    POSTERIOR_DIR, DEFAULT_MODEL, DEFAULT_WINDOW_DAYS, DEFAULT_HALF_LIFE, write_active,
)
from src.validation.backtest import model_label

RANDOM_SEED = 42

# Terms checked and summarized, per model (all exist in the posterior)
LEAGUE_TERMS_BY_MODEL = {
    "goals": ["alpha0", "alpha_home", "sigma_off", "sigma_def", "rho"],
    "xg":    ["alpha0", "alpha_home", "sigma_off", "sigma_def", "sigma_phi_off", "sigma_phi_def",
              "kappa", "rho"],
}
LEAGUE_TERMS = LEAGUE_TERMS_BY_MODEL["goals"]      # default for notebooks that import diagnostics()
TEAM_TERMS = ["mu_off", "mu_def"]

# Pass thresholds for the printed checks
MAX_RHAT = 1.01
MIN_ESS = 400
MIN_BFMI = 0.3


def rhat_ess(post, name: str) -> tuple[float, float]:
    """Worst R-hat and smallest ESS for one variable (scalar or one per team)."""
    x = post[name].values                         # (chain, draw) or (chain, draw, team)
    if x.ndim == 2:
        x = x[:, :, None]
    rhats = [float(az.rhat(x[:, :, i])) for i in range(x.shape[2])]
    esses = [float(az.ess(x[:, :, i])) for i in range(x.shape[2])]
    return max(rhats), min(esses)


def bfmi(idata) -> list[float]:
    """Bayesian fraction of missing information per chain, from the energy trace.

    Low values (< 0.3) mean NUTS explores the energy levels poorly; usually a sign
    the model needs reparameterizing.
    """
    energy = idata.sample_stats["energy"].values  # (chain, draw)
    return [float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2)) for e in energy]


def diagnostics(idata, league_terms=None, team_terms=None) -> dict:
    """R-hat, ESS, divergences, BFMI and league-term summaries; prints PASS / CHECK."""
    league_terms = league_terms or LEAGUE_TERMS
    team_terms = team_terms or TEAM_TERMS
    post = idata.posterior
    out = {"variables": {}, "league_terms": {}}

    for name in league_terms + team_terms:
        r, e = rhat_ess(post, name)
        out["variables"][name] = {"max_rhat": r, "min_ess": e}

    for name in league_terms:
        x = post[name].values.ravel()
        lo, hi = np.quantile(x, [0.05, 0.95])
        out["league_terms"][name] = {"mean": float(x.mean()), "sd": float(x.std()),
                                     "q05": float(lo), "q95": float(hi)}

    out["divergences"] = int(idata.sample_stats["diverging"].values.sum())
    out["bfmi"] = bfmi(idata)

    def check(label, ok, note=""):
        print(f"{'PASS ' if ok else 'CHECK'}  {label}" + (f"   ({note})" if note else ""))

    v = out["variables"]
    worst_r = max(x["max_rhat"] for x in v.values())
    worst_e = min(x["min_ess"] for x in v.values())
    check(f"R-hat < {MAX_RHAT} (all parameters)", worst_r < MAX_RHAT, f"worst {worst_r:.4f}")
    check(f"ESS > {MIN_ESS} (all parameters)", worst_e > MIN_ESS, f"smallest {worst_e:,.0f}")
    check("no divergences", out["divergences"] == 0, f"{out['divergences']}")
    check(f"BFMI > {MIN_BFMI} (every chain)", min(out["bfmi"]) > MIN_BFMI,
          ", ".join(f"{b:.2f}" for b in out["bfmi"]))
    return out


def save_netcdf(idata, nc_path) -> bool:
    """Write to a temporary file, then swap it in. Returns True if the final file is in place.

    On Windows a file that another program has open (a notebook kernel that
    loaded it, a VS Code tab, a file viewer) is locked and can't be overwritten.
    Writing to a new file first means a finished fit is never lost: if the swap
    fails, the new posterior stays in the .tmp file and the message says what to do.
    """
    tmp = nc_path.with_name(nc_path.name + ".tmp")
    idata.to_netcdf(tmp)
    try:
        os.replace(tmp, nc_path)
        return True
    except PermissionError:
        print(f"\nCHECK  {nc_path.name} is locked by another program (usually a notebook kernel "
              f"that loaded it, or a VS Code tab showing it).")
        print(f"       The new fit is saved as {tmp.name}. Close or restart whatever has the file open, then run:")
        print(f'       Move-Item -Force "{tmp}" "{nc_path}"')
        return False


def all_seasons(conn) -> list[str]:
    s = pd.read_sql("SELECT DISTINCT season FROM schedule", conn)["season"].astype(str)
    return sorted(s)


def unscored_games(conn, start, end=None) -> pd.DataFrame:
    """Loaded regular-season games (in `games`) from `start` on (before `end`, if given)
    with no rows in shot_xg. Such games silently drop out of the game table, so the fit
    would use stale data and wrong goalie ratings."""
    df = pd.read_sql("""
        SELECT  g.id AS game_id, s.game_date, s.season
        FROM    games g
        JOIN    schedule s ON s.game_id = g.id
        WHERE   g.game_type = 2
        AND     NOT EXISTS (SELECT 1 FROM shot_xg x WHERE x.game_id = g.id)
    """, conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    keep = df["game_date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= df["game_date"] < pd.Timestamp(end)
    return df[keep]


def priors(model: str) -> dict:
    if model == "xg":
        return {
            "beta0 (log xG per team-game)": {"mean": float(hx.BETA0_PRIOR[0]), "sd": hx.BETA0_PRIOR[1]},
            "beta_home_sd": hx.HOME_PRIOR_SD,
            "delta0, delta_home sd (goals per xG, log)": hx.DELTA_PRIOR_SD,
            "sigma_off, sigma_def halfnormal sd": hx.SIGMA_PRIOR_SD,
            "sigma_phi halfnormal sd (finishing)": hx.SIGMA_PHI_PRIOR_SD,
            "log_kappa": {"mean": float(hx.LOG_KAPPA_PRIOR[0]), "sd": hx.LOG_KAPPA_PRIOR[1]},
            "tau_sd (rho = exp(tau) - 1)": hx.TIE_PRIOR_SD,
        }
    return {
        "alpha0": {"mean": float(h1.ALPHA0_PRIOR[0]), "sd": h1.ALPHA0_PRIOR[1]},
        "alpha_home_sd": h1.HOME_PRIOR_SD,
        "sigma_halfnormal_sd": h1.SIGMA_PRIOR_SD,
        "tau_sd (rho = exp(tau) - 1)": h1.TIE_PRIOR_SD,
    }


def main():
    # Save plots without opening windows. Set here, not at import, so notebooks
    # that import diagnostics() from this file keep their inline plots.
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Fit Track B on a training window and make it the active fit.")
    parser.add_argument("--model", choices=["goals", "xg"], default=DEFAULT_MODEL,
                        help="xg = Track B v1.1 (default, Phase 7 choice); goals = Track B v1")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"training window in days (default {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE,
                        help="v1.1 only: recency half-life in days (default none)")
    parser.add_argument("--end", default=None,
                        help="window end, YYYY-MM-DD, not included (default: day after the last game in the database)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output")
    parser.add_argument("--skip-xg-check", action="store_true",
                        help="fit even if some loaded games have no xG yet (not recommended)")
    args = parser.parse_args()
    if args.half_life and args.model != "xg":
        raise SystemExit("--half-life is only supported with --model xg")

    # 1. Data and window
    conn = get_connection()
    gt_all, _ = build_game_table(conn, all_seasons(conn))
    end = pd.Timestamp(args.end) if args.end else gt_all["game_date"].max().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=args.window)

    # 1b. Every loaded game in the window (and, by default, after it) must have xG.
    #     Games without shot_xg rows are missing from the game table, so without this
    #     check the fit would quietly use old data.
    missing = unscored_games(conn, start, pd.Timestamp(args.end) if args.end else None)
    conn.close()
    if len(missing) and not args.skip_xg_check:
        by_season = missing.groupby("season").size()
        print(f"STOP  {len(missing):,} loaded games have no xG yet "
              f"({missing['game_date'].min().date()} to {missing['game_date'].max().date()}):")
        for season, n in by_season.items():
            print(f"        {season}: {n:,} games  ->  py -m src.pipelines.score_xg --season {season}")
        raise SystemExit("Score them, then rerun this fit. (Override with --skip-xg-check.)")
    print(f"xG check: every loaded game from {start.date()} on has xG." if not len(missing)
          else f"WARNING  {len(missing):,} games without xG skipped (--skip-xg-check).")
    gt, teams = select_window(gt_all, start, end)
    if gt.empty:
        raise ValueError(f"No games from {start.date()} to {end.date()}.")
    seasons = sorted(gt["season"].astype(str).unique())

    # 2. Paths and overwrite protection
    label = model_label(args.model, args.window, args.half_life)
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    stem = POSTERIOR_DIR / f"{label}_{end:%Y%m%d}"
    nc_path = stem.with_suffix(".nc")
    if nc_path.exists() and not args.overwrite:
        print(f"{nc_path} already exists. Use --overwrite to replace it.")
        return

    no_history = float(((gt["gamma_home"] == 0) | (gt["gamma_away"] == 0)).mean())
    hg, ag = GOAL_COLS
    print(f"{label}: {len(gt):,} games, {len(teams)} teams, "
          f"{gt['game_date'].min().date()} to {gt['game_date'].max().date()} (seasons {', '.join(seasons)})")
    print(f"Regulation goals/game: home {gt[hg].mean():.3f}, away {gt[ag].mean():.3f}; "
          f"tied after regulation: {(gt[hg] == gt[ag]).mean():.1%}; "
          f"games with a goalie at gamma = 0: {no_history:.1%}")
    if args.model == "xg":
        hx_, ax_ = XG_COLS
        print(f"Regulation xG/game: home {gt[hx_].mean():.3f}, away {gt[ax_].mean():.3f}")

    # 3. Fit
    if args.model == "xg":
        model = build_model_xg(gt, teams, half_life_days=args.half_life, ref_date=end)
    else:
        model = build_model(gt, teams)
    print(f"Sampling {CHAINS} chains ({TUNE:,} tune + {DRAWS:,} draws each)...")
    idata = sample(model, seed=RANDOM_SEED)

    # 4. Diagnose
    league_terms = LEAGUE_TERMS_BY_MODEL[args.model]
    diag = diagnostics(idata, league_terms)

    # 5. Save posterior (settings travel with the samples)
    model_name = "track_b_v1.1_xg" if args.model == "xg" else "track_b_v1_goals"
    idata.posterior.attrs.update({
        "model": model_name,
        "label": label,
        "seasons": ",".join(seasons),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "teams": ",".join(teams),
    })
    saved = save_netcdf(idata, nc_path)

    # 6. Ratings table (goal scale: what the game model uses)
    ratings = team_ratings(idata, teams)
    ratings.to_csv(f"{stem}_ratings.csv", index=False)

    # 7. Trace plot of the league terms (chain 1)
    league_chain1 = np.column_stack([idata.posterior[n].values[0] for n in league_terms])
    plot_traces(league_chain1, league_terms, savepath=f"{stem}_trace.png")
    plt.close("all")

    # 8. Metadata
    meta = {
        "model": model_name,
        "label": label,
        "window_days": args.window,
        "half_life_days": args.half_life,
        "window_start": str(start.date()),
        "window_end_excluded": str(end.date()),
        "seasons": seasons,
        "n_games": int(len(gt)),
        "n_teams": len(teams),
        "teams": teams,
        "first_game": str(gt["game_date"].min().date()),
        "last_game": str(gt["game_date"].max().date()),
        "likelihood": ("Gamma on regulation xG (mean nu, shape kappa) + tie-inflated Poisson on regulation goals "
                       "with finishing effects phi" if args.model == "xg"
                       else "tie-inflated Poisson (rho multiplies tied regulation scores by 1 + rho)"),
        "goals_target": "regulation goals (periods 1-3, empty-net goals excluded): " + ", ".join(GOAL_COLS),
        "goalie_input": "fixed gamma = -log(theta_mean), Phase 3 rating of the starter as of game date (goals only)",
        "share_games_gamma_zero": no_history,
        "priors": priors(args.model),
        "sampler": {"name": "NUTS", "chains": CHAINS, "tune": TUNE, "draws": DRAWS,
                    "target_accept": TARGET_ACCEPT, "seed": RANDOM_SEED},
        "diagnostics": diag,
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {"numpy": np.__version__, "pymc": pm.__version__, "arviz": az.__version__},
    }
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 9. Make it the active fit (only if the .nc is really in place)
    if saved:
        write_active({
            "fit_file": nc_path.name,
            "model": args.model,
            "label": label,
            "window_days": args.window,
            "half_life_days": args.half_life,
            "start": str(start.date()),
            "end": str(end.date()),
            "seasons": seasons,
            "teams": teams,
            "n_games": int(len(gt)),
            "fitted_at": meta["fitted_at"],
        })

    # 10. Summary
    print()
    for name, t in diag["league_terms"].items():
        print(f"{name:13s} mean {t['mean']:+.4f}   90% [{t['q05']:+.4f}, {t['q95']:+.4f}]")
    print("\nTop 5 / bottom 5 by net rating (goal scale, log; +0.10 is about 10% better):")
    show = ratings[["team", "off_mean", "def_mean", "net_mean", "net_lo", "net_hi"]].round(3)
    print(show.head(5).to_string(index=False))
    print("...")
    print(show.tail(5).to_string(index=False, header=False))
    print(f"\nSaved {nc_path.name}, _meta.json, _ratings.csv, _trace.png to {POSTERIOR_DIR}")
    if saved:
        print(f"Active fit is now {nc_path.name}. Next: py -m src.processing.late_game, "
              f"then py -m src.processing.score_state")
    else:
        print("Active fit NOT changed (the .nc file could not be put in place; see CHECK above).")


if __name__ == "__main__":
    main()