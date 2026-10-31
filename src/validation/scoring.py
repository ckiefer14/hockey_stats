"""
scoring.py
==========
Scoring rules and calibration for the Phase 7 backtest.

Home: src/validation/scoring.py

Win probability:  log loss, Brier score, accuracy, calibration table, calibration slope,
                  paired model-vs-model differences with bootstrap intervals.
Score outputs:    ranked probability score for total goals; Brier for over/unders and puck lines.

All "90%" ranges are bootstrap intervals over games (resampling whole games keeps the
comparison paired: both models are scored on the same resampled games).
"""

import numpy as np
import pandas as pd
from scipy import optimize

EPS = 1e-6


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def accuracy(p, y) -> float:
    return float(np.mean((np.asarray(p) > 0.5) == np.asarray(y)))


def calibration_slope(p, y) -> tuple[float, float]:
    """Fit y ~ Bernoulli(sigmoid(a + b * logit(p))). Returns (a, b).

    b = 1: well calibrated. b > 1: predictions too cautious (should be more extreme).
    b < 1: predictions too confident. a shifts all predictions (e.g. home ice off).
    """
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    x = np.log(p / (1 - p))
    y = np.asarray(y, float)
    if np.std(x) < 1e-9:                     # same prediction for every game: slope undefined
        return float(np.log(y.mean() / (1 - y.mean())) - x.mean()), float("nan")

    def nll(c):
        z = c[0] + c[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    a, b = optimize.minimize(nll, x0=[0.0, 1.0], method="BFGS").x
    return float(a), float(b)


def _bootstrap_idx(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Games in equal-size groups by predicted probability: predicted vs actual, with a 90% band."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), games=("y", "size"))
    t["band_90"] = 1.645 * np.sqrt(t["predicted"] * (1 - t["predicted"]) / t["games"])
    t["inside"] = (t["actual"] - t["predicted"]).abs() <= t["band_90"]
    return t


def score_models(df: pd.DataFrame, cols: dict, y_col: str = "home_won", reference: str | None = None,
                 n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per model: log loss, Brier, accuracy, calibration slope (with 90% ranges).

    cols: {display name: probability column}. reference: display name to compare every
    model against (paired log-loss difference; negative = better than the reference).
    """
    y = df[y_col].to_numpy(float)
    idx = _bootstrap_idx(len(df), n_boot, seed)
    rows = []
    ref_p = df[cols[reference]].to_numpy(float) if reference else None
    for name, col in cols.items():
        p = df[col].to_numpy(float)
        a, b = calibration_slope(p, y)
        boot_b = np.array([calibration_slope(p[i], y[i])[1] for i in idx[:200]])   # slope is slower; 200 is plenty
        slope_range = "n/a" if np.isnan(b) else f"[{np.nanquantile(boot_b, 0.05):.2f}, {np.nanquantile(boot_b, 0.95):.2f}]"
        row = {"model": name, "games": len(df), "log loss": log_loss(p, y), "Brier": brier(p, y),
               "accuracy": accuracy(p, y), "calibration slope": b,
               "slope 90%": slope_range,
               "intercept": a}
        if reference:
            pc = np.clip(p, EPS, 1 - EPS); rc = np.clip(ref_p, EPS, 1 - EPS)
            per_game = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)) + (y * np.log(rc) + (1 - y) * np.log(1 - rc))
            boots = per_game[idx].mean(axis=1)
            row[f"vs {reference}"] = per_game.mean()
            row[f"vs {reference} 90%"] = f"[{np.quantile(boots, 0.05):+.4f}, {np.quantile(boots, 0.95):+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(df: pd.DataFrame, col_a: str, col_b: str, y_col: str = "home_won",
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Log loss of A minus log loss of B, with a 90% bootstrap interval (negative = A better)."""
    y = df[y_col].to_numpy(float)
    pa = np.clip(df[col_a].to_numpy(float), EPS, 1 - EPS)
    pb = np.clip(df[col_b].to_numpy(float), EPS, 1 - EPS)
    per_game = -(y * np.log(pa) + (1 - y) * np.log(1 - pa)) + (y * np.log(pb) + (1 - y) * np.log(1 - pb))
    boots = per_game[_bootstrap_idx(len(df), n_boot, seed)].mean(axis=1)
    return {"difference": float(per_game.mean()),
            "lo": float(np.quantile(boots, 0.05)), "hi": float(np.quantile(boots, 0.95)),
            "share_of_boots_A_better": float((boots < 0).mean())}


def by_group(df: pd.DataFrame, col: str, group: str, y_col: str = "home_won") -> pd.DataFrame:
    """Log loss and calibration slope per group (e.g. season, month)."""
    rows = []
    for g, sub in df.groupby(group):
        a, b = calibration_slope(sub[col], sub[y_col])
        rows.append({group: g, "games": len(sub), "log loss": log_loss(sub[col], sub[y_col]),
                     "calibration slope": b, "mean predicted": sub[col].mean(), "actual": sub[y_col].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Score outputs
# ---------------------------------------------------------------------------

def total_goals_rps(prob_cols: pd.DataFrame, actual_total) -> np.ndarray:
    """Ranked probability score per game for the total-goals distribution (lower is better).

    prob_cols: columns P(total = 0), P(total = 1), ..., last column = that many or more.
    """
    P = prob_cols.to_numpy(float)
    k = np.arange(P.shape[1])
    actual = np.minimum(np.asarray(actual_total), k[-1])
    cdf_pred = np.cumsum(P, axis=1)
    cdf_obs = (k[None, :] >= actual[:, None]).astype(float)
    return np.sum((cdf_pred - cdf_obs) ** 2, axis=1) / (P.shape[1] - 1)


def score_outputs(df: pd.DataFrame, climatology: dict | None = None) -> pd.DataFrame:
    """Brier scores for over/unders and puck lines, and RPS for totals.

    climatology: optional {market: probability} from earlier seasons, scored as a baseline
    (predicting the same historical frequency for every game).
    """
    total = df["final_home"] + df["final_away"]
    margin = df["final_home"] - df["final_away"]
    markets = {
        "over 5.5": ("p_over_5_5", total > 5.5),
        "over 6.5": ("p_over_6_5", total > 6.5),
        "home -1.5": ("p_home_m1_5", margin >= 2),
        "away -1.5": ("p_away_m1_5", margin <= -2),
    }
    rows = []
    for name, (col, outcome) in markets.items():
        y = outcome.to_numpy(float)
        row = {"market": name, "predicted": df[col].mean(), "actual": y.mean(), "Brier": brier(df[col], y)}
        if climatology and name in climatology:
            row["Brier, historical rate"] = brier(np.full(len(df), climatology[name]), y)
        rows.append(row)
    prob_cols = df[[c for c in df.columns if c.startswith("p_total_")]]
    rps = total_goals_rps(prob_cols, total)
    rows.append({"market": "total goals (RPS)", "predicted": df["exp_total"].mean(),
                 "actual": total.mean(), "Brier": float(rps.mean())})
    return pd.DataFrame(rows)