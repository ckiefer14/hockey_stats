"""
check_calibration.py
====================
Is the live model's backtest calibrated, and would a recalibration help on games it has not
seen?

Home: scripts/check_calibration.py

Usage (from the project root):
    py .\\scripts\\check_calibration.py                       # the live model's backtest label
    py .\\scripts\\check_calibration.py --label trackbtv_w730_r

1. Calibration slope and intercept (logit(P actual) = a + b * logit(p)), pooled and per
   season, with 90% bootstrap ranges. b = 1: calibrated; b > 1: predictions too cautious
   (should be further from 50%); b < 1: too confident. a: overall home bias.
2. Walk-forward recalibration: for each season after the first, fit a and b on the earlier
   seasons only, apply p' = sigmoid(a + b * logit(p)) to that season, and compare log loss
   with the raw predictions (paired, 90% range). Adopt only if the pooled range is entirely
   below 0, the usual rule.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.validation.scoring import calibration_slope, log_loss, paired_difference

BT = OUTPUT_DIR / "backtests"
EPS = 1e-6


def load(label):
    parts = []
    for f in sorted(BT.glob(f"{label}_*_scored.csv")):
        season = f.name[len(label) + 1:].split("_")[0]
        if season.isdigit():
            d = pd.read_csv(f, usecols=["game_id", "p_home", "home_won"]).drop_duplicates("game_id", keep="last")
            parts.append(d.assign(season=season))
    if not parts:
        raise SystemExit(f"No backtest files for {label} in {BT}")
    return pd.concat(parts, ignore_index=True)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def boot_slope(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        x, y = calibration_slope(s["p_home"], s["home_won"])
        a.append(x); b.append(y)
    return np.quantile(a, [0.05, 0.95]), np.quantile(b, [0.05, 0.95])


def main():
    parser = argparse.ArgumentParser(description="Calibration of the live model's backtest.")
    parser.add_argument("--label", default="trackbtv_w730_r_lu", help="backtest label (default: the live model)")
    args = parser.parse_args()
    d = load(args.label)
    print(f"{args.label}: seasons {', '.join(sorted(d['season'].unique()))}, {len(d):,} games\n")

    print("1. Calibration (b = 1 calibrated; b > 1 too cautious; a = home bias on the logit scale)")
    rows = []
    for name, g in [("pooled", d)] + [(s, g) for s, g in d.groupby("season")]:
        a, b = calibration_slope(g["p_home"], g["home_won"])
        (alo, ahi), (blo, bhi) = boot_slope(g)
        rows.append({"set": name, "games": len(g), "slope b": b, "b 90%": f"[{blo:.2f}, {bhi:.2f}]",
                     "intercept a": a, "a 90%": f"[{alo:+.3f}, {ahi:+.3f}]",
                     "mean predicted": g["p_home"].mean(), "home won": g["home_won"].mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n2. Walk-forward recalibration (fit on earlier seasons, apply to the next)")
    seasons = sorted(d["season"].unique())
    out = []
    for i, s in enumerate(seasons[1:], start=1):
        train = d[d["season"].isin(seasons[:i])]
        test = d[d["season"] == s].copy()
        a, b = calibration_slope(train["p_home"], train["home_won"])
        test["p_cal"] = 1 / (1 + np.exp(-(a + b * logit(test["p_home"]))))
        r = paired_difference(test, "p_cal", "p_home")
        print(f"  {s}: a {a:+.3f}, b {b:.3f} from {', '.join(seasons[:i])}; recalibrated minus raw "
              f"{r['difference']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        out.append(test)
    if out:
        t = pd.concat(out)
        r = paired_difference(t, "p_cal", "p_home")
        verdict = "clearly better: adopt" if r["hi"] < 0 else "clearly worse" if r["lo"] > 0 else "within luck: keep raw"
        print(f"  pooled ({len(t):,} games): raw {log_loss(t['p_home'], t['home_won']):.4f}, recalibrated "
              f"{log_loss(t['p_cal'], t['home_won']):.4f}; difference {r['difference']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
