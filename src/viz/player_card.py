"""Player card (PNG): percentile bars with uncertainty, in the card style.

Home: src/viz/player_card.py

Built by src/pipelines/make_player_card.py. Each bar is the player's percentile among
regulars at his position (forwards or defensemen): the dot is where his rating's mean falls,
the whisker where the ends of its 90% range fall. A wide whisker = the model is not sure yet.

    Offense / Defense / Net   v2.2 5v5 player ratings (player_rating_asof, Bayesian RAPM on
                              xG per 60, controlling for teammates, opponents, score, zone)
    Finishing                 v2.1 shooter effect (xg_layer beta: goals vs LightGBM xG on the
                              same shots, with goalie / rink / season terms)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import card_style as cs
from .team_card import TEAM_NAMES, _ordinal, _team_color


@dataclass
class PlayerCardData:
    name: str
    player_id: int
    team: str
    position: str                         # C / L / R / D
    group_label: str                      # "forwards" / "defensemen"
    as_of: str                            # rating fit date
    bars: list = field(default_factory=list)       # dicts label, value_text, pct, pct_lo, pct_hi, note
    tiles: list = field(default_factory=list)      # (label, value, sub)
    path: pd.DataFrame | None = None      # fit_date, net, net_lo, net_hi, grp_lo, grp_mid, grp_hi (xG/60)
    notes: list = field(default_factory=list)      # footnotes (staleness, small samples)
    model_label: str = ""
    color: str | None = None
    # trend panel / header text (goalie cards change these)
    trend_title: str = "Net impact over time"
    trend_subtitle: str | None = None
    trend_pct: bool = False               # path values are fractions: show as %
    kind: str = "PLAYER CARD"


def _bar_row(fig, x0, x1, y, h, c, b):
    """One percentile bar: label + value at left, track 0-100 with fill to the percentile and a whisker."""
    fig.text(x0, y + h * 0.18, b["label"], fontsize=11, fontweight="bold", va="bottom")
    fig.text(x0, y - h * 0.10, b["value_text"], fontsize=8.5, color=cs.INK_2, va="top")
    tx0, tx1 = x0 + 0.20, x1 - 0.085
    W = tx1 - tx0
    th = h * 0.42
    fig.patches.append(FancyBboxPatch((tx0, y - th / 2), W, th, boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, facecolor=cs.PAGE, edgecolor="none", zorder=1))
    if b.get("pct") is None:
        fig.text(tx0 + W / 2, y, b.get("note") or "no rating", fontsize=9, color=cs.MUTED, ha="center", va="center")
        return
    p, lo, hi = b["pct"], b["pct_lo"], b["pct_hi"]
    fig.patches.append(FancyBboxPatch((tx0, y - th / 2), max(W * p / 100, 0.002), th,
                                      boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                      facecolor=cs.tint(c, 0.55), edgecolor="none", zorder=2))
    # 90% range whisker and the dot
    fig.add_artist(_line([tx0 + W * lo / 100, tx0 + W * hi / 100], [y, y], c, 2.2, 3))
    for e in (lo, hi):
        fig.add_artist(_line([tx0 + W * e / 100] * 2, [y - th * 0.32, y + th * 0.32], c, 2.2, 3))
    fig.add_artist(_dot(tx0 + W * p / 100, y, c))
    for q in (25, 50, 75):
        fig.add_artist(_line([tx0 + W * q / 100] * 2, [y - th / 2, y + th / 2], cs.SURFACE, 1.0, 2))
    fig.text(x1, y + h * 0.05, _ordinal(round(p)), fontsize=17, fontweight="bold", color=cs.INK, ha="right",
             va="center")
    fig.text(x1, y - h * 0.32, f"{_ordinal(round(lo))}–{_ordinal(round(hi))}", fontsize=8.5, color=cs.MUTED,
             ha="right", va="center")
    if b.get("note"):
        fig.text(tx0, y - th / 2 - 0.006, b["note"], fontsize=8, color=cs.MUTED, va="top")


def _line(xs, ys, color, lw, z):
    from matplotlib.lines import Line2D
    return Line2D(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round")


def _dot(x, y, color):
    from matplotlib.lines import Line2D
    return Line2D([x], [y], marker="o", ms=10, color=color, mec=cs.SURFACE, mew=1.8, zorder=4)


def _trend(fig, rect, d: PlayerCardData, c: str):
    p = d.path.sort_values("fit_date")
    ax = fig.add_axes(rect)
    ax.set_title(d.trend_title, pad=24)
    sub = d.trend_subtitle or ("5v5 xG per 60 for minus against vs an average player, each rating fit "
                               "(2-season window)")
    ax.text(0, 1.03, f"{sub} · band = 90% range · gray = middle 80% of {d.group_label}",
            transform=ax.transAxes, fontsize=8.5, color=cs.MUTED, va="bottom")
    x = pd.to_datetime(p["fit_date"])
    ax.fill_between(x, p["grp_lo"], p["grp_hi"], color=cs.GRID, alpha=0.6, linewidth=0, zorder=1)
    ax.fill_between(x, p["net_lo"], p["net_hi"], color=cs.tint(c, 0.62), linewidth=0, zorder=2)
    ax.plot(x, p["net"], color=c, lw=2, zorder=3)
    ax.axhline(0, color=cs.AXIS, lw=0.9, zorder=2)
    last = p.iloc[-1]
    ax.plot([x.iloc[-1]], [last["net"]], "o", color=c, ms=6, mec=cs.SURFACE, mew=1.5, zorder=4)
    lab = f"{100 * last['net']:+.1f}%" if d.trend_pct else f"{last['net']:+.2f}"
    ax.annotate(lab.replace("-", "−"), (x.iloc[-1], last["net"]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9.5, fontweight="bold", annotation_clip=False)
    ax.yaxis.set_major_formatter((lambda v, _: f"{100 * v:+.0f}%".replace("-", "−")) if d.trend_pct
                                 else (lambda v, _: f"{v:+.1f}".replace("-", "−")))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def plot_player_card(d: PlayerCardData):
    c = d.color or _team_color(d.team)
    has_path = d.path is not None and len(d.path) >= 2
    top = 1.45
    bot = top + max(0.95 + 0.82 * len(d.bars), 0.75 + 0.72 * len(d.tiles))
    H = bot + (3.3 if has_path else 0.55)
    fig = cs.new_figure(12, H)
    Y = lambda inches: 1 - inches / H

    fig.text(0.04, Y(0.35), d.name, fontsize=28, fontweight="bold", color=cs.ink_for(c), va="top")
    fig.text(0.04, Y(0.95), " · ".join(x for x in (TEAM_NAMES.get(d.team, d.team), {"C": "Center", "L": "Left wing",
             "R": "Right wing", "D": "Defenseman", "G": "Goaltender"}.get(d.position, d.position)) if x), fontsize=11, color=cs.INK_2,
             va="top")
    fig.text(0.96, Y(0.36), d.kind, fontsize=10, color=cs.MUTED, fontweight="bold", ha="right", va="top")
    fig.text(0.96, Y(0.62), f"ratings as of {d.as_of}", fontsize=11, color=cs.INK, ha="right", va="top")

    # percentile bars
    cs.card_background(fig, [(0.025, Y(bot), 0.64, Y(top))])
    fig.text(0.045, Y(top + 0.25), f"Percentile among {d.group_label}", fontsize=11, fontweight="bold", va="top")
    fig.text(0.045, Y(top + 0.50), "dot = best estimate · whisker = 90% range · 50th = typical regular",
             fontsize=8.5, color=cs.MUTED, va="top")
    for i, b in enumerate(d.bars):
        _bar_row(fig, 0.045, 0.62, Y(top + 1.25 + 0.82 * i), 0.82 / H, c, b)

    # tiles
    cs.card_background(fig, [(0.655, Y(bot), 0.975, Y(top))])
    fig.text(0.675, Y(top + 0.25), "Usage", fontsize=11, fontweight="bold", va="top")
    th = 0.62
    for j, (label, value, sub) in enumerate(d.tiles):
        ty = top + 0.62 + j * (th + 0.1)
        fig.patches.append(Rectangle((0.675, Y(ty + th)), 0.28, th / H, transform=fig.transFigure,
                                     facecolor=cs.PAGE, edgecolor="none", zorder=-5))
        fig.text(0.688, Y(ty + 0.1), label, fontsize=8.5, color=cs.INK_2, va="top")
        fig.text(0.688, Y(ty + th - 0.1), value, fontsize=15, fontweight="bold", va="bottom")
        if sub:
            fig.text(0.945, Y(ty + th - 0.1), sub, fontsize=8, color=cs.MUTED, va="bottom", ha="right")

    if has_path:
        cs.card_background(fig, [(0.025, Y(H - 0.55), 0.975, Y(bot + 0.18))])
        _trend(fig, (0.075, Y(H - 0.95), 0.80, (H - 0.95 - bot - 0.95) / H), d, c)

    foot = d.model_label
    if d.notes:
        foot = "   ·   ".join([foot] + d.notes) if foot else "   ·   ".join(d.notes)
    fig.text(0.04, Y(H - 0.2), foot, fontsize=8, color=cs.MUTED, va="bottom")
    return fig