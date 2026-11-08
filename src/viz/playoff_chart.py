"""Playoff odds charts (PNG), in the card style.

Home: src/viz/playoff_chart.py

From the files playoff_odds.py writes (output/playoffs/):

- ``plot_playoff_board(tab, ...)``: one run (playoff_odds_<date>.csv). Four division panels;
  per team one bar = P(playoffs), split into how it gets there: wins the division (team
  colour), 2nd/3rd in the division (lighter), wild card (lightest). Points now and projected
  points (90% range) beside it.
- ``plot_playoff_trend(hist, ...)``: every run of the season (playoff_odds_history_<season>.csv).
  Four division panels, one line per team of P(playoffs) by run date, labelled at the end.

Team colours: each team's own colour; when two teams in a division are too close (the
card_style distance rule), the later one switches to its next team colour, and if none works
its line is dashed. Labels carry the team name, so colour is never the only key.

Usage::

    from src.viz.playoff_chart import plot_playoff_board, plot_playoff_trend
    plot_playoff_board(pd.read_csv("output/playoffs/playoff_odds_2026-11-08.csv")).savefig(..., dpi=200)
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from . import card_style as cs

DIVISION_ORDER = ["Atlantic", "Metropolitan", "Central", "Pacific"]   # East first, then West


def _pct(p: float) -> str:
    if p > 0.999:
        return ">99.9%"
    if p < 0.001:
        return "<0.1%"
    return f"{100 * p:.1f}%" if p < 0.1 or p > 0.9 else f"{100 * p:.0f}%"


def division_colors(teams: list[str]) -> dict[str, tuple[str, bool]]:
    """team -> (colour, dashed). Teams earlier in the list keep their first colour."""
    out, used = {}, []
    for t in teams:
        opts = cs._team_colors(t) or [cs.ROAD_NEUTRAL]
        # team order, but pale colours (L > 0.84, their tints wash out) after near-black
        opts = sorted(opts, key=lambda c: 1 if cs.oklab(c)[0] > 0.84 else 0)
        pick = next((c for c in opts if all(cs.color_distance(c, u)["normal"] >= cs.MIN_NORMAL for u in used)), None)
        out[t] = (pick or opts[0], pick is None)
        used.append(pick or opts[0])
    return out


def _divisions(tab: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = tab[["conference", "division"]].drop_duplicates()
    order = {d: i for i, d in enumerate(DIVISION_ORDER)}
    pairs = pairs.assign(o=pairs["division"].map(order).fillna(99)).sort_values(["o", "conference", "division"])
    return list(pairs[["conference", "division"]].itertuples(index=False, name=None))


def _panels(fig, n: int, top: float, bottom: float):
    """Rects (x0, y0, w, h) for n panels in a 2-column grid, plus their background."""
    rows = int(np.ceil(n / 2))
    gap = 0.018
    h = (top - bottom - gap * (rows - 1)) / rows
    rects = []
    for i in range(n):
        r, c = divmod(i, 2)
        x0 = 0.025 + c * (0.475 + gap)
        y0 = top - (r + 1) * h - r * gap
        rects.append((x0, y0, 0.475 - gap / 2, h))
    cs.card_background(fig, [(x, y, x + w, y + h) for x, y, w, h in rects])
    return rects


def _title(fig, title: str, subtitle: str):
    fig.text(0.04, 0.975, title, fontsize=22, fontweight="bold", va="top")
    fig.text(0.04, 0.938, subtitle, fontsize=9.5, color=cs.INK_2, va="top")


# --------------------------------------------------------------------------
# board: one run
# --------------------------------------------------------------------------

def plot_playoff_board(tab: pd.DataFrame, date: str | None = None, note: str = ""):
    """Bars of P(playoffs) per team, split by route, in four division panels."""
    date = date or str(tab["date"].iloc[0])
    divs = _divisions(tab)
    fig = cs.new_figure(12, 13.5)
    sims = int(tab["sims"].iloc[0]) if "sims" in tab else None
    _title(fig, f"Playoff odds · {date}",
           f"{sims:,} simulated rest-of-seasons from the active fit" if sims else "Simulated rest of season")

    # legend for the bar shades (gray, the team colour carries identity)
    lx, ly = 0.04, 0.905
    for label, amt in (("Wins division", 0.0), ("2nd / 3rd in division", 0.45), ("Wild card", 0.72)):
        fig.patches.append(FancyBboxPatch((lx, ly - 0.006), 0.022, 0.012, boxstyle="round,pad=0,rounding_size=0.002",
                                          transform=fig.transFigure, facecolor=cs.tint(cs.INK_2, amt), edgecolor="none"))
        t = fig.text(lx + 0.028, ly, label, fontsize=9, color=cs.INK_2, va="center")
        lx += 0.028 + t.get_window_extent(renderer=fig.canvas.get_renderer()).width / (fig.dpi * 12) + 0.025
    fig.text(0.96, ly, "bar length = P(playoffs), shades = route in, in each team's colour",
             fontsize=9, color=cs.MUTED, va="center", ha="right")

    rects = _panels(fig, len(divs), top=0.88, bottom=0.05)
    for (conf, div), (x0, y0, w, h) in zip(divs, rects):
        g = tab[(tab["division"] == div) & (tab["conference"] == conf)].copy()
        g = g.sort_values(["p_playoffs", "points_mean"], ascending=False).reset_index(drop=True)
        cols = division_colors(list(g["team"]))
        fig.text(x0 + 0.018, y0 + h - 0.014, div, fontsize=12.5, fontweight="bold", va="top")
        fig.text(x0 + w - 0.018, y0 + h - 0.016, f"{conf} Conference".replace("Conference Conference", "Conference"),
                 fontsize=9, color=cs.MUTED, va="top", ha="right")

        ax = fig.add_axes([x0 + 0.075, y0 + 0.035, w - 0.205, h - 0.095])
        ax.set_facecolor("none")
        n = len(g)
        ax.set_xlim(0, 1.2)
        ax.set_ylim(-0.6, n - 0.4)
        for i, r in g.iterrows():
            y = n - 1 - i
            c, _ = cols[r["team"]]
            top3_only = max(r["p_div_top3"] - r["p_div_win"], 0.0)
            f = 1 - min(max(cs.oklab(c)[0] - 0.7, 0) * 3, 0.6)      # pale team colours: lighter tints
            segs = [(r["p_div_win"], c), (top3_only, cs.tint(c, 0.45 * f)), (r["p_wildcard"], cs.tint(c, 0.72 * f))]
            left = 0.0
            ax.barh(y, 1.0, height=0.62, color=cs.PAGE, linewidth=0, zorder=0)
            for v, col in segs:
                if v > 0.0005:
                    ax.barh(y, v, left=left, height=0.62, color=col, linewidth=0, edgecolor=cs.SURFACE, zorder=1)
                    if left > 0:
                        ax.plot([left, left], [y - 0.31, y + 0.31], color=cs.SURFACE, lw=1.5, zorder=2)
                left += v
            p = r["p_playoffs"]
            ax.text(max(p, 0.0) + 0.015, y, _pct(p), va="center", ha="left", fontsize=10,
                    fontweight="bold", color=cs.INK, zorder=3)
            fig_y = ax.transData.transform((0, y))
            yy = fig.transFigure.inverted().transform(fig_y)[1]
            fig.text(x0 + 0.018, yy, r["team"], fontsize=11, fontweight="bold", color=cs.ink_for(c), va="center")
            fig.text(x0 + w - 0.018, yy + 0.006, f"{int(r['points_now'])} pts · {int(r['gp_now'])} GP",
                     fontsize=8.5, color=cs.INK_2, va="center", ha="right")
            fig.text(x0 + w - 0.018, yy - 0.0075,
                     f"proj. {r['points_mean']:.0f} ({r['points_lo']:.0f}–{r['points_hi']:.0f})",
                     fontsize=8, color=cs.MUTED, va="center", ha="right")
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.axvline(0.5, color=cs.GRID, lw=1, zorder=-1)
        for sp in ax.spines.values():
            sp.set_visible(False)

    foot = "Top 3 in each division + 2 wild cards per conference. Ties: points, regulation wins, ROW, wins."
    if note:
        foot += f"   ·   {note}"
    fig.text(0.04, 0.02, foot, fontsize=8, color=cs.MUTED, va="bottom")
    return fig


# --------------------------------------------------------------------------
# trend: the season so far
# --------------------------------------------------------------------------

def _spread_labels(ys: list[float], min_gap: float, lo: float = 0.0, hi: float = 1.0) -> list[float]:
    """Nudge label positions apart (keeps order); ys in data units."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    over = pos[-1] - hi if len(pos) else 0
    if over > 0:
        pos -= over
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - min_gap)
    pos = np.maximum(pos, lo - 0.02)
    out = np.empty_like(pos)
    out[order] = pos
    return list(out)


def plot_playoff_trend(hist: pd.DataFrame, season: str = "", note: str = ""):
    """P(playoffs) by run date, one line per team, four division panels."""
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").drop_duplicates(["date", "team"], keep="last")
    dates = sorted(hist["date"].unique())
    last = hist[hist["date"] == dates[-1]]
    divs = _divisions(last)
    fig = cs.new_figure(12, 11.5)
    s = f"{season[:4]}-{season[6:]}" if len(season) == 8 else season
    _title(fig, f"Playoff odds through the season{' · ' + s if s else ''}",
           f"{len(dates)} run{'s' if len(dates) != 1 else ''}, latest {pd.Timestamp(dates[-1]).date()}"
           + ("   ·   one point so far: the lines grow with each weekly run" if len(dates) == 1 else ""))
    rects = _panels(fig, len(divs), top=0.905, bottom=0.05)
    span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    for (conf, div), (x0, y0, w, h) in zip(divs, rects):
        g_last = last[(last["division"] == div) & (last["conference"] == conf)].sort_values(
            ["p_playoffs", "points_mean"], ascending=False)
        teams = list(g_last["team"])
        cols = division_colors(teams)
        fig.text(x0 + 0.018, y0 + h - 0.014, div, fontsize=12.5, fontweight="bold", va="top")
        ax = fig.add_axes([x0 + 0.065, y0 + 0.045, w - 0.155, h - 0.095])
        for t in teams[::-1]:                                   # best team drawn on top
            d = hist[hist["team"] == t]
            c, dashed = cols[t]
            ax.plot(d["date"], d["p_playoffs"], color=c, lw=2, ls=(0, (4, 2)) if dashed else "-",
                    marker="o" if len(d) <= 12 else None, ms=4.5, mec=cs.SURFACE, mew=1.2, zorder=3)
        ys = _spread_labels(list(g_last["p_playoffs"]), min_gap=0.075)
        x_end = pd.Timestamp(dates[-1])
        pad = pd.Timedelta(days=max(1, span) * 0.03 + 0.4)
        for t, p, yl in zip(teams, g_last["p_playoffs"], ys):
            c, _ = cols[t]
            ax.text(x_end + pad, yl, f"{t} {_pct(p)}", va="center", fontsize=9, fontweight="bold",
                    color=cs.ink_for(c), clip_on=False)
        ax.axhline(0.5, color=cs.AXIS, lw=0.8, zorder=1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        if span == 0:
            ax.set_xlim(x_end - pd.Timedelta(days=7), x_end + pd.Timedelta(days=1))
            ax.set_xticks([x_end])
        else:
            ax.set_xlim(pd.Timestamp(dates[0]) - pd.Timedelta(days=1), x_end + pd.Timedelta(days=1))
        if span:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    foot = "P(playoffs) from each playoff_odds run (10,000 simulated seasons each). Dashed line: team colour too close to a division rival's."
    if note:
        foot += f"   ·   {note}"
    fig.text(0.04, 0.02, foot, fontsize=8, color=cs.MUTED, va="bottom")
    return fig
