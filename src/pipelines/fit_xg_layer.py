"""
fit_xg_layer.py
===============
v2.1: fit the hierarchical xG layer (shooter finishing, goalie skill, rink recording effect,
season x shot-type conversion) on top of the LightGBM offsets in shot_xg, and test whether it
predicts held-out shots better than LightGBM alone.

Home: src/pipelines/fit_xg_layer.py

Usage (from the project root, after score_xg):
    py -m src.pipelines.fit_xg_layer --holdout            # walk-forward test: 2023-24, 2024-25, 2025-26
    py -m src.pipelines.fit_xg_layer --holdout --test-seasons 20252026
    py -m src.pipelines.fit_xg_layer --final              # fit every loaded season, save the effects
    py -m src.pipelines.fit_xg_layer --holdout --terms beta gamma   # only shooter + goalie effects
      (terms: beta = shooter finishing, gamma = goalie skill, delta = rink, eps = season x shot type;
       default all four; a subset is refit without the others, and saved with a suffix, e.g. _bg)

--holdout: for each test season, fit on all earlier seasons only, then score that season's
shots with (a) LightGBM alone and (b) LightGBM + the layer's posterior-mean effects (unseen
shooters / goalies / rinks get 0; a new season's shot-type effect is carried over from the
latest training season). Reports shot log loss for both with the paired difference and a 90%
range (bootstrap over games), the same with each term left out (which terms earn their place),
and goals vs expected by shot type and by rink in the test season.

--final: fit on every loaded season, print the spreads, the top / bottom shooters and goalies,
rinks and the season x shot-type table, posterior predictive checks (how often a shooter's,
goalie's and rink's goals fall inside its 90% predictive interval), and save:
    output/posteriors/xg_layer_<last season>.json      spreads, settings, fit info
    output/posteriors/xg_layer_<last season>_<term>.csv  effects (mean, sd, shots, goals, xG)
    output/posteriors/xg_layer_<last season>_gamma_cov.npy  goalie posterior covariance (for draws)

Note on the offsets: shot_xg scores each past season with a LightGBM model trained on the
other seasons (leave-one-season-out), so training shots' offsets came from models that also saw
later seasons. That affects only the shot-geometry curve, and it is the same for both methods
compared, so the held-out comparison stays fair.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR, SEASONS
from src.data.db import get_connection
from src.models.stochastic import xg_layer as xl

POSTERIOR_DIR = OUTPUT_DIR / "posteriors"
N_BOOT = 1000


def loaded_seasons(conn) -> list[str]:
    have = {str(s) for (s,) in conn.execute("SELECT DISTINCT season FROM shot_xg")}
    return [s for s in SEASONS if s in have]


def paired_game_bootstrap(loss_a, loss_b, games, n_boot=N_BOOT, seed=0):
    """Mean per-shot log loss difference a - b with a 90% range, resampling whole games."""
    d = pd.DataFrame({"g": games, "d": loss_a - loss_b})
    per = d.groupby("g")["d"].agg(["sum", "size"])
    s, c = per["sum"].to_numpy(), per["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), (n_boot, len(per)))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


def fmt(diff):
    d, lo, hi = diff
    verdict = "clearly better" if hi < 0 else "clearly worse" if lo > 0 else "within luck"
    return f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]  {verdict}"


def holdout(conn, test_seasons: list[str], terms: list[str] = xl.TERMS):
    seasons = loaded_seasons(conn)
    rows = []
    for test in test_seasons:
        train = [s for s in seasons if s < test]
        if not train:
            print(f"{test}: no earlier season to train on, skipped")
            continue
        print(f"\n=== Test season {test} (trained on {', '.join(train)}) ===")
        tr = xl.load_shots(conn, train)
        te = xl.load_shots(conn, [test])
        t0 = time.time()
        fit = xl.fit_laplace(tr, terms=terms)
        print(f"  fit {time.time() - t0:.0f}s")
        y = te["goal"].to_numpy()
        base = xl.shot_log_loss(te["logit_xg"].to_numpy(), y)
        full = xl.shot_log_loss(xl.adjusted_logit(te, fit["effects"], fit["levels"]), y)
        res = {"test": test, "shots": len(te), "lgbm": base.mean(), "layer": full.mean()}
        diff = paired_game_bootstrap(full, base, te["game_id"].to_numpy())
        res.update({"diff": diff[0], "lo": diff[1], "hi": diff[2]})
        print(f"  shot log loss: LightGBM {base.mean():.5f}, + layer {full.mean():.5f}; layer minus LightGBM {fmt(diff)}")
        # which terms earn their place: drop each term from the prediction
        for t in fit["terms"]:
            keep = [k for k in fit["terms"] if k != t]
            part = xl.shot_log_loss(xl.adjusted_logit(te, fit["effects"], fit["levels"], terms=keep), y)
            d = paired_game_bootstrap(full, part, te["game_id"].to_numpy())
            res[f"gain_{t}"] = -d[0]
            print(f"    {xl.TERM_NAMES[t]:20s} full minus without it {fmt(d)}")
        no_carry = xl.shot_log_loss(xl.adjusted_logit(te, fit["effects"], fit["levels"], carry_eps=False), y)
        if "eps" in terms:
            print(f"    shot-type effect carried from {train[-1]} vs set to 0: "
                  f"{fmt(paired_game_bootstrap(full, no_carry, te['game_id'].to_numpy()))}")
        # coverage of the test season by training levels
        codes, _ = xl.index_groups(te, fit["levels"])
        print("  test shots with a known " + ", ".join(
            f"{xl.TERM_NAMES[t]} {np.mean(codes[t] >= 0):.0%}" for t in ("beta", "gamma", "delta") if t in terms))
        # goals vs expected by shot type and rink, test season
        eta = xl.adjusted_logit(te, fit["effects"], fit["levels"])
        te = te.assign(p_layer=1 / (1 + np.exp(-eta)))
        g = te.groupby("shot_group").agg(shots=("goal", "size"), goals=("goal", "sum"),
                                         lgbm=("xg", "sum"), layer=("p_layer", "sum"))
        g["goals / LightGBM"] = g["goals"] / g["lgbm"]
        g["goals / layer"] = g["goals"] / g["layer"]
        print("  goals vs expected by shot type (test season):")
        print(g.round(3).to_string())
        r = te.groupby("rink").agg(goals=("goal", "sum"), lgbm=("xg", "sum"), layer=("p_layer", "sum"))
        spread = lambda col: float(np.std(r["goals"] / r[col]))
        print(f"  rinks: spread of goals / expected across rinks: LightGBM {spread('lgbm'):.3f}, layer {spread('layer'):.3f} "
              f"(pure chance alone gives about {float(np.mean(1 / np.sqrt(r['lgbm']))):.3f})")
        rows.append(res)
    if rows:
        t = pd.DataFrame(rows)
        print("\n=== Summary (negative = layer better) ===")
        print(t.round(5).to_string(index=False))
        lo_all = (t["hi"] < 0).all()
        print("Verdict: " + ("the layer beats LightGBM alone in every test season" if lo_all else
                             "the layer does not clearly beat LightGBM in every test season: see the rows above"))
    return rows


def ppc(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """Share of shooters / goalies / rinks whose goals fall inside their 90% predictive interval
    (normal approximation: binomial noise of the shots plus the effect's own posterior sd)."""
    eta = xl.adjusted_logit(df, fit["effects"], fit["levels"])
    p = 1 / (1 + np.exp(-eta))
    out = []
    for t in fit["terms"]:
        col = xl.TERM_COLS[t]
        g = pd.DataFrame({"k": df[col], "y": df["goal"], "p": p, "v": p * (1 - p)}).groupby("k").sum()
        sd_eff = fit["effects"][t].set_index("level")["sd"].reindex(g.index).fillna(0).to_numpy()
        var = g["v"] + (g["v"] * sd_eff) ** 2           # d(sum p)/d(effect) = sum p(1-p)
        z = (g["y"] - g["p"]) / np.sqrt(var)
        big = g["p"] >= 5                                # groups with at least 5 expected goals
        out.append({"term": xl.TERM_NAMES[t], "groups": int(big.sum()),
                    "inside 90%": float((np.abs(z[big]) < 1.645).mean())})
    return pd.DataFrame(out)


def suffix(terms: list[str]) -> str:
    return "" if list(terms) == xl.TERMS else "_" + "".join({"beta": "b", "gamma": "g", "delta": "d", "eps": "e"}[t] for t in terms)


def final(conn, terms: list[str] = xl.TERMS):
    seasons = loaded_seasons(conn)
    df = xl.load_shots(conn, seasons)
    print(f"Final fit on {', '.join(seasons)}: {len(df):,} shots, {df.goal.sum():,} goals, LightGBM xG {df.xg.sum():,.0f}")
    t0 = time.time()
    fit = xl.fit_laplace(df, terms=terms)
    print(f"  fit {time.time() - t0:.0f}s")
    names = pd.read_sql("SELECT player_id AS level, MAX(name) AS name FROM goalie_game_stats GROUP BY player_id", conn)
    try:
        skaters = pd.read_sql("SELECT id AS level, first_name || ' ' || last_name AS name FROM players", conn)
    except Exception:
        skaters = pd.DataFrame(columns=["level", "name"])

    print("\nSpreads (logit scale; 0.1 = about 10% more or fewer goals from the same shots):")
    for t in fit["terms"]:
        lo, hi = fit["sigma_ci"][t]
        print(f"  {xl.TERM_NAMES[t]:20s} sigma {fit['sigma'][t]:.3f}  90% [{lo:.3f}, {hi:.3f}]")

    def show(t, label, names_df, min_shots, n=10):
        e = fit["effects"][t].merge(names_df, on="level", how="left") if len(names_df) else fit["effects"][t]
        e = e[e["shots"] >= min_shots].copy()
        e["goals - xG"] = e["goals"] - e["xg"]
        sign = -1 if t == "gamma" else 1                 # gamma: positive = better goalie
        e = e.sort_values("mean", ascending=False)
        cols = [c for c in ["level", "name", "shots", "goals", "xg", "goals - xG", "mean", "sd"] if c in e]
        print(f"\nTop {n} {label} (effect, logit scale):")
        print(e.head(n)[cols].round(3).to_string(index=False))
        print(f"Bottom {n} {label}:")
        print(e.tail(n)[cols].round(3).to_string(index=False))

    if "beta" in terms:
        show("beta", "finishers (shooters, 300+ shots)", skaters, 300)
    if "gamma" in terms:
        show("gamma", "goalies (2,000+ shots faced; positive = better)", names, 2000)
    if "delta" in terms:
        r = fit["effects"]["delta"].sort_values("mean")
        print("\nRinks (positive = more goals than LightGBM expects from the recorded shots there):")
        print(r[["level", "shots", "goals", "xg", "mean", "sd"]].round(3).to_string(index=False))
    if "eps" in terms:
        e = fit["effects"]["eps"].copy()
        e[["season", "group"]] = e["level"].str.split(":", expand=True)
        print("\nSeason x shot type (logit effect):")
        print(e.pivot(index="group", columns="season", values="mean").round(3).to_string())
    print("\nPosterior predictive check (groups with 5+ expected goals; about 90% expected):")
    print(ppc(df, fit).round(3).to_string(index=False))

    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    stem = POSTERIOR_DIR / f"xg_layer_{seasons[-1]}{suffix(terms)}"
    for t, eff in fit["effects"].items():
        eff.to_csv(f"{stem}_{t}.csv", index=False)
    if fit["cov_gamma"] is not None:
        np.save(f"{stem}_gamma_cov.npy", fit["cov_gamma"])
    meta = {"seasons": seasons, "n_shots": fit["n_shots"], "terms": fit["terms"], "sigma": fit["sigma"],
            "sigma_ci": fit["sigma_ci"], "log_marginal": fit["log_marginal"], "converged": fit["converged"],
            "sigma_priors": xl.SIGMA_PRIORS, "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    with open(f"{stem}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {stem.name}.json, _<term>.csv, _gamma_cov.npy to {POSTERIOR_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Fit / test the v2.1 hierarchical xG layer.")
    parser.add_argument("--holdout", action="store_true", help="walk-forward held-out test")
    parser.add_argument("--test-seasons", nargs="+", default=None,
                        help="test seasons for --holdout (default: the last three loaded, excluding the current one)")
    parser.add_argument("--final", action="store_true", help="fit every loaded season and save the effects")
    parser.add_argument("--terms", nargs="+", choices=xl.TERMS, default=xl.TERMS,
                        help="effects to fit: beta (shooter), gamma (goalie), delta (rink), eps (season x type)")
    args = parser.parse_args()
    if not (args.holdout or args.final):
        parser.error("choose --holdout and/or --final")
    conn = get_connection()
    if args.holdout:
        seasons = loaded_seasons(conn)
        tests = args.test_seasons or [s for s in seasons[1:] if s != SEASONS[-1]][-3:]
        holdout(conn, tests, args.terms)
    if args.final:
        final(conn, args.terms)
    conn.close()


if __name__ == "__main__":
    main()