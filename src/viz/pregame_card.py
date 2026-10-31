"""Pregame matchup card (PNG).

One card per game, built from what ``predict_game`` / ``predict_day`` already
compute:

- ``p_home_draws``: P(home win) for each posterior draw from the exact game
  model (``game_model.outcome_probs`` per draw). Mean = the prediction, the
  5th-95th percentiles = the 90% interval.
- ``outcome``: mean probabilities of the six ways a game can end.
- ``final_scores``: P(final home goals = i, final away goals = j) from the
  score-state simulator (official final scores, OT/SO winner +1, empty-net
  goals included). Build it from simulated scores with
  ``final_score_matrix(home_goals, away_goals)``.

Usage::

    from src.viz.pregame_card import PregameData, plot_pregame_card
    fig = plot_pregame_card(data)
    fig.savefig("output/game_summaries/2026-04-10_DAL_at_COL_pregame.png", dpi=200)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from matplotlib.patches import Rectangle

from . import card_style as cs

OUTCOME_KEYS = ("home_reg", "home_ot", "home_so", "away_so", "away_ot", "away_reg")


@dataclass
class PregameData:
    home: str
    away: str
    date: str                               # "2026-04-10"
    p_home_draws: np.ndarray                # one P(home win) per posterior draw
    outcome: dict                           # keys OUTCOME_KEYS, mean probabilities
    p_tied_reg: float                       # P(tied after regulation)
    final_scores: np.ndarray                # [home goals, away goals] probability matrix
    home_goalie: str = ""
    away_goalie: str = ""
    home_goalie_confirmed: bool = False
    away_goalie_confirmed: bool = False
    model_label: str = ""                   # e.g. "Track B v1.1 · trackbxg_w365_20260417"
    note: str = ""                          # free text for the footer
    colors: tuple | None = None             # (home, away); None = team colours


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def final_score_matrix(home_goals, away_goals, max_goals: int = 10) -> np.ndarray:
    """Probability matrix from simulated final scores (arrays of ints).

    Scores above ``max_goals`` are folded into the last row / column.
    """
    h = np.minimum(np.asarray(home_goals, int), max_goals)
    a = np.minimum(np.asarray(away_goals, int), max_goals)
    m = np.zeros((max_goals + 1, max_goals + 1))
    np.add.at(m, (h, a), 1)
    return m / m.sum()


def score_summaries(final_scores: np.ndarray) -> dict:
    """Totals, puck lines and margin shares from the final-score matrix."""
    n_h, n_a = final_scores.shape
    hh, aa = np.meshgrid(np.arange(n_h), np.arange(n_a), indexing="ij")
    total = hh + aa
    margin = hh - aa
    totals = np.bincount(total.ravel(), weights=final_scores.ravel())
    return {
        "totals": totals,                                   # P(total = k)
        "over_5_5": final_scores[total >= 6].sum(),
        "over_6_5": final_scores[total >= 7].sum(),
        "home_minus_1_5": final_scores[margin >= 2].sum(),
        "away_minus_1_5": final_scores[margin <= -2].sum(),
        "mean_total": (total * final_scores).sum(),
    }


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def _header(fig, d: PregameData, top: float):
    hc, ac = cs.ink_for(d.colors[0]), cs.ink_for(d.colors[1])
    fig.text(0.04, top, f"{d.away}", fontsize=26, fontweight="bold", color=ac,
             va="top")
    w_away = 0.016 * max(len(d.away), 3) + 0.035
    fig.text(0.04 + w_away, top - 0.004, "at", fontsize=16, color=cs.MUTED, va="top")
    fig.text(0.04 + w_away + 0.032, top, f"{d.home}", fontsize=26,
             fontweight="bold", color=hc, va="top")
    fig.text(0.96, top - 0.002, "PREGAME", fontsize=10, color=cs.MUTED,
             fontweight="bold", ha="right", va="top")
    fig.text(0.96, top - 0.027, d.date, fontsize=13, color=cs.INK, ha="right",
             va="top")

    def g(name, ok):
        if not name:
            return "starter unknown"
        return f"{name} ({'confirmed' if ok else 'projected'})"

    fig.text(0.04, top - 0.052,
             f"Goalies:  {d.away} {g(d.away_goalie, d.away_goalie_confirmed)}"
             f"   ·   {d.home} {g(d.home_goalie, d.home_goalie_confirmed)}",
             fontsize=9.5, color=cs.INK_2, va="top")


def _hero(fig, rect, d: PregameData):
    """Big P(win) numbers + the posterior strip of per-draw P(home win)."""
    hc, ac = d.colors
    p = np.asarray(d.p_home_draws, float)
    mean = p.mean()
    lo, hi = np.percentile(p, [5, 95])
    x0, y0, w, h = rect

    fig.text(x0, y0 + h, "Win probability", fontsize=11, fontweight="bold",
             va="top")
    fig.text(x0, y0 + h - 0.026,
             "Each posterior draw of the team ratings gives one P(win); "
             "the spread is our uncertainty about team strength.",
             fontsize=8.5, color=cs.MUTED, va="top")

    # numbers
    fig.text(x0, y0 + h - 0.045, cs.pct(mean), fontsize=34, fontweight="bold",
             color=cs.ink_for(hc), va="top")
    fig.text(x0, y0 + h - 0.093, f"{d.home} wins", fontsize=10, color=cs.INK_2,
             va="top")
    fig.text(x0 + w, y0 + h - 0.045, cs.pct(1 - mean), fontsize=34,
             fontweight="bold", color=cs.ink_for(ac), va="top", ha="right")
    fig.text(x0 + w, y0 + h - 0.093, f"{d.away} wins", fontsize=10,
             color=cs.INK_2, va="top", ha="right")
    fig.text(x0 + w / 2, y0 + h - 0.052,
             f"90% interval\n{cs.pct(lo)} – {cs.pct(hi)}", fontsize=9.5,
             color=cs.INK_2, ha="center", va="top", linespacing=1.4)

    # strip: histogram of per-draw P(home win)
    ax = fig.add_axes([x0, y0, w, h - 0.135])
    ax.set_facecolor(cs.SURFACE)
    span = max(0.12, (hi - lo) * 2.2)
    left = min(0.5 - 0.02, mean - span / 2)
    right = max(0.5 + 0.02, mean + span / 2)
    bins = np.linspace(left, right, 45)
    counts, edges = np.histogram(p, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    inside = (centers >= lo) & (centers <= hi)
    width = (edges[1] - edges[0]) * 0.82
    ax.bar(centers[inside], counts[inside], width=width, color=hc, linewidth=0)
    ax.bar(centers[~inside], counts[~inside], width=width,
           color=cs.tint(hc, 0.65), linewidth=0)
    ax.axvline(0.5, color=cs.MUTED, lw=1)
    ax.text(0.5 - 0.003, counts.max() * 1.25, "50%", color=cs.MUTED,
            fontsize=8.5, ha="right", va="top")
    ax.axvline(mean, color=cs.INK, lw=1.4)
    ax.set_xlim(left, right)
    ax.set_ylim(0, counts.max() * 1.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ticks = np.arange(np.ceil(left * 20) / 20, right + 1e-9, 0.05)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel(f"P({d.home} win) per posterior draw   ·   "
                  f"dark bars = middle 90%", fontsize=8.5, color=cs.MUTED)


def _odds_tiles(fig, rect, d: PregameData, s: dict):
    """Fair odds and market-style numbers as stat tiles."""
    hc, ac = d.colors
    mean = float(np.mean(d.p_home_draws))
    x0, y0, w, h = rect
    fig.text(x0, y0 + h, "Fair odds (no margin)", fontsize=11,
             fontweight="bold", va="top")
    tiles = [
        (f"{d.home} moneyline", cs.fair_american_odds(mean), hc),
        (f"{d.away} moneyline", cs.fair_american_odds(1 - mean), ac),
        ("Tied after 60 min", cs.pct(d.p_tied_reg), cs.INK),
        (f"{d.home} −1.5", f"{cs.pct(s['home_minus_1_5'])}", hc),
        (f"{d.away} −1.5", f"{cs.pct(s['away_minus_1_5'])}", ac),
        ("Over 5.5 / 6.5", f"{cs.pct(s['over_5_5'], 0)} / {cs.pct(s['over_6_5'], 0)}",
         cs.INK),
    ]
    cols, rows = 2, 3
    gap = 0.012
    tw = (w - gap * (cols - 1)) / cols
    th = (h - 0.045 - gap * (rows - 1)) / rows
    for i, (label, value, color) in enumerate(tiles):
        r, c = divmod(i, cols)
        tx = x0 + c * (tw + gap)
        ty = y0 + h - 0.045 - (r + 1) * th - r * gap
        fig.patches.append(Rectangle((tx, ty), tw, th, transform=fig.transFigure,
                                     facecolor=cs.PAGE, edgecolor="none",
                                     zorder=-5))
        fig.text(tx + 0.012, ty + th - 0.008, label, fontsize=8.5,
                 color=cs.INK_2, va="top")
        fig.text(tx + 0.012, ty + 0.007, value, fontsize=15, fontweight="bold",
                 color=cs.ink_for(color), va="bottom")


def _win_paths(fig, rect, d: PregameData):
    """One stacked bar: how each team wins (regulation / OT / shootout)."""
    hc, ac = d.colors
    ax = fig.add_axes(rect)
    ax.set_facecolor(cs.SURFACE)
    o = d.outcome
    segs = [
        ("Regulation", o["home_reg"], hc),
        ("OT", o["home_ot"], cs.tint(hc, 0.35)),
        ("SO", o["home_so"], cs.tint(hc, 0.62)),
        ("SO", o["away_so"], cs.tint(ac, 0.62)),
        ("OT", o["away_ot"], cs.tint(ac, 0.35)),
        ("Regulation", o["away_reg"], ac),
    ]
    total = sum(v for _, v, _ in segs)
    left = 0.0
    gap = 0.003
    for label, v, color in segs:
        v = v / total
        ax.barh(0, max(v - gap, 0), left=left + gap / 2, height=0.8,
                color=color, linewidth=0)
        mid = left + v / 2
        if v >= 0.05:
            ax.text(mid, 0.17, label, ha="center", va="center", fontsize=8.5,
                    color=cs.text_on(color))
            ax.text(mid, -0.15, cs.pct(v), ha="center", va="center", fontsize=10,
                    fontweight="bold", color=cs.text_on(color))
        else:
            ax.text(mid, -0.48, f"{label}\n{cs.pct(v)}", ha="center", va="top",
                    fontsize=8, color=cs.INK_2, linespacing=1.2)
        left += v
    ax.axvline(float(np.mean(d.p_home_draws)), ymin=0.18, ymax=0.82,
               color=cs.INK, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 0.55)
    ax.axis("off")
    ax.text(0, 0.66, "How the game ends", fontsize=11, fontweight="bold",
            va="bottom", transform=ax.transData)
    ax.text(1, 0.66, f"← {d.home} wins   ·   {d.away} wins →", fontsize=9,
            color=cs.MUTED, ha="right", va="bottom")


def _score_grid(fig, rect, d: PregameData, max_show: int = 7):
    """Heat map of final scores; the five most likely are labelled."""
    ax = fig.add_axes(rect)
    m = d.final_scores[: max_show + 1, : max_show + 1].copy()
    disp = m.T                                        # rows = away goals
    vmax = disp.max()
    ramps = (cs.team_ramp(d.colors[0], "home"), cs.team_ramp(d.colors[1], "away"))
    n = max_show + 1
    for a in range(n):
        for h in range(n):
            if h == a:          # final scores can't be tied
                color = cs.PAGE
            else:
                ramp = ramps[0] if h > a else ramps[1]
                color = ramp(disp[a, h] / vmax)
            ax.add_patch(Rectangle((h - 0.5, a - 0.5), 1, 1, facecolor=color,
                                   edgecolor=cs.SURFACE, linewidth=2))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    top = np.argsort(disp.ravel())[::-1][:5]
    for rank, idx in enumerate(top):
        a, h = divmod(idx, disp.shape[1])
        v = disp[a, h]
        ramp = ramps[0] if h > a else ramps[1]
        ax.text(h, a, f"{v:.1%}", ha="center", va="center", fontsize=8.5,
                fontweight="bold" if rank == 0 else "normal",
                color=cs.text_on(ramp(v / vmax)))
    ax.set_xticks(range(max_show + 1))
    ax.set_yticks(range(max_show + 1))
    ax.set_xticklabels([str(i) if i < max_show else f"{i}+"
                        for i in range(max_show + 1)])
    ax.set_yticklabels([str(i) if i < max_show else f"{i}+"
                        for i in range(max_show + 1)])
    ax.set_xlabel(f"{d.home} goals", color=cs.ink_for(d.colors[0]),
                  fontweight="bold")
    ax.set_ylabel(f"{d.away} goals", color=cs.ink_for(d.colors[1]),
                  fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Likely final scores", pad=22)
    ax.text(0, 1.02, f"{d.home} wins below the diagonal, {d.away} above · OT/SO +1, "
                     "empty-netters included",
            transform=ax.transAxes, fontsize=8.5, color=cs.MUTED, va="bottom")


def _totals(fig, rect, d: PregameData, s: dict):
    ax = fig.add_axes(rect)
    totals = s["totals"]
    k = np.arange(len(totals))
    keep = k <= 12
    tk, tv = k[keep], totals[keep].copy()
    tv[-1] += totals[~keep].sum()
    ax.bar(tk, tv, width=0.78, color=cs.tint(cs.INK_2, 0.45), linewidth=0)
    ymax = tv.max() * 1.55
    for line, key in ((5.5, "over_5_5"), (6.5, "over_6_5")):
        ax.plot([line, line], [0, ymax * 0.80], color=cs.INK, lw=1)
        left_side = line < 6
        ax.text(line + (-0.12 if left_side else 0.12), ymax * 0.97,
                f"Line {line}\nOver {cs.pct(s[key], 0)}\nUnder {cs.pct(1 - s[key], 0)}",
                ha="right" if left_side else "left", va="top", fontsize=8.5,
                color=cs.INK, linespacing=1.2)
    ax.set_ylim(0, ymax * 1.06)
    ax.set_xlim(-0.6, tk.max() + 0.6)
    ax.set_xticks(tk)
    ax.set_xticklabels([str(i) if i < 12 else "12+" for i in tk])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_yticks(np.arange(0, tv.max() * 1.05, 0.05))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Total goals (final score)")
    ax.set_title("Total goals", pad=22)
    ax.text(0, 1.02, f"Mean {s['mean_total']:.2f}", transform=ax.transAxes,
            fontsize=8.5, color=cs.MUTED, va="bottom")


# --------------------------------------------------------------------------
# card
# --------------------------------------------------------------------------

def plot_pregame_card(d: PregameData):
    """Draw the pregame card and return the figure."""
    if d.colors is None:
        d = replace(d, colors=cs.matchup_colors(d.home, d.away))
    s = score_summaries(d.final_scores)
    fig = cs.new_figure(12, 11)

    cs.card_background(fig, [
        (0.025, 0.635, 0.975, 0.895),   # hero + tiles
        (0.025, 0.515, 0.975, 0.62),    # win paths
        (0.025, 0.055, 0.975, 0.50),    # score grid + totals
    ])

    _header(fig, d, top=0.965)
    _hero(fig, (0.05, 0.678, 0.50, 0.202), d)
    _odds_tiles(fig, (0.60, 0.655, 0.355, 0.225), d, s)
    _win_paths(fig, (0.05, 0.527, 0.90, 0.07), d)
    _score_grid(fig, (0.04, 0.09, 0.44, 0.35), d)
    _totals(fig, (0.56, 0.105, 0.39, 0.335), d, s)

    foot = d.model_label or "Track B v1.1"
    if d.note:
        foot += f"   ·   {d.note}"
    fig.text(0.04, 0.025, foot, fontsize=8, color=cs.MUTED, va="bottom")
    fig.text(0.96, 0.025, "P(win): exact game model per draw · scores: "
                          "score-state simulator", fontsize=8, color=cs.MUTED,
             ha="right", va="bottom")
    return fig