"""League rankings card (PNG): every team on one card, in the model's order.

Home: src/viz/rankings_card.py

Built by src/pipelines/make_team_card.py (``--rankings``, also with ``--all``) from
league_table(): position 1..N by the posterior mean net rating (offense + defense, starting
goalie excluded), the 90% range of the rank across posterior draws, offense / defense
positions, and the latest playoff odds. Same numbers as the rank tiles on the team cards.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.patches import Rectangle

from . import card_style as cs
from .team_card import _ordinal, _pp, _team_color


def plot_rankings_card(table: pd.DataFrame, date: str, records: dict | None = None,
                       playoffs: pd.DataFrame | None = None, odds_date: str | None = None, model_label: str = ""):
    """table: league_table(fit) sorted by net_rank; records: team -> dict(w, l, otl, pts, gp);
    playoffs: playoff_odds_<date>.csv indexed by team."""
    n = len(table)
    row_in = 0.36
    H = 2.55 + n * row_in
    fig = cs.new_figure(12, H)
    Y = lambda inches: 1 - inches / H                     # inches from the top -> figure fraction

    fig.text(0.04, Y(0.35), "Power rankings", fontsize=26, fontweight="bold", va="top")
    fig.text(0.96, Y(0.38), date, fontsize=13, ha="right", va="top")
    fig.text(0.04, Y(0.95), "Ordered by the model's net rating (offense + defense, starting goalie excluded). "
             "Ranges are 90%: where the team could really rank given what the model is still unsure about.",
             fontsize=9, color=cs.INK_2, va="top")

    top_in, bot_in = 1.35, H - 0.45
    cs.card_background(fig, [(0.025, Y(bot_in), 0.975, Y(top_in))])
    hdr = Y(top_in + 0.22)
    ax_x0, ax_x1 = 0.30, 0.60
    cols = {"range": 0.665, "off": 0.74, "def": 0.81, "po": 0.88, "pts": 0.94}
    fig.text(0.045, hdr, "#", fontsize=9, color=cs.MUTED, fontweight="bold", va="center")
    fig.text(0.09, hdr, "Team", fontsize=9, color=cs.MUTED, fontweight="bold", va="center")
    fig.text((ax_x0 + ax_x1) / 2, hdr, "Net rating vs average team (90% range)", fontsize=9, color=cs.MUTED,
             fontweight="bold", va="center", ha="center")
    for k, lab in (("range", "Rank range"), ("off", "Offense"), ("def", "Defense"),
                   ("po", "Playoffs"), ("pts", "Proj. pts")):
        fig.text(cols[k], hdr, lab, fontsize=9, color=cs.MUTED, fontweight="bold", va="center", ha="center")

    rows_top = top_in + 0.62
    y_of = lambda i: Y(rows_top + (i + 0.5) * row_in)
    ax = fig.add_axes([ax_x0, Y(rows_top + n * row_in), ax_x1 - ax_x0, n * row_in / H])
    ax.set_facecolor("none")
    lo = min(table["net_pct_lo"].min(), -0.05)
    hi = max(table["net_pct_hi"].max(), 0.05)
    pad = (hi - lo) * 0.04
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(n, 0)
    ax.axvline(0, color=cs.AXIS, lw=0.9, zorder=1)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda v, _: _pp(v, 0))
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)

    for i, r in table.reset_index(drop=True).iterrows():
        t = r["team"]
        c = _team_color(t)
        yy = y_of(i)
        if i % 2 == 0:
            fig.patches.append(Rectangle((0.035, yy - row_in / H / 2), 0.93, row_in / H, transform=fig.transFigure,
                                         facecolor=cs.PAGE, edgecolor="none", zorder=-6))
        fig.text(0.045, yy, str(int(r["net_rank"])), fontsize=12, fontweight="bold", va="center")
        fig.text(0.09, yy, t, fontsize=12, fontweight="bold", color=cs.ink_for(c), va="center")
        rec = (records or {}).get(t)
        if rec and rec.get("gp"):
            fig.text(0.145, yy, f"{int(rec['w'])}-{int(rec['l'])}-{int(rec['otl'])} · {int(rec['pts'])} pts",
                     fontsize=8.5, color=cs.INK_2, va="center")
        ax.plot([r["net_pct_lo"], r["net_pct_hi"]], [i + 0.5, i + 0.5], color=cs.tint(c, 0.5), lw=4,
                solid_capstyle="round", zorder=2)
        ax.plot([r["net_pct"]], [i + 0.5], "o", color=c, ms=7.5, mec=cs.SURFACE, mew=1.3, zorder=3)
        fig.text(cols["range"], yy, f"{int(r['net_rank_lo'])}–{int(r['net_rank_hi'])}", fontsize=9.5,
                 color=cs.INK_2, va="center", ha="center")
        fig.text(cols["off"], yy, _ordinal(r["off_rank"]), fontsize=9.5, color=cs.INK, va="center", ha="center")
        fig.text(cols["def"], yy, _ordinal(r["def_rank"]), fontsize=9.5, color=cs.INK, va="center", ha="center")
        if playoffs is not None and t in playoffs.index:
            p = float(playoffs.loc[t, "p_playoffs"])
            ptxt = ">99.9%" if p > 0.999 else "<0.1%" if p < 0.001 else f"{100 * p:.0f}%"
            fig.text(cols["po"], yy, ptxt, fontsize=9.5, fontweight="bold", color=cs.INK, va="center", ha="center")
            fig.text(cols["pts"], yy, f"{playoffs.loc[t, 'points_mean']:.0f}", fontsize=9.5, color=cs.INK_2,
                     va="center", ha="center")

    foot = model_label or "Track B"
    if odds_date:
        foot += f"   ·   playoff odds of {odds_date}"
    fig.text(0.04, Y(H - 0.2), foot, fontsize=8, color=cs.MUTED, va="bottom")
    fig.text(0.96, Y(H - 0.2), "Offense / Defense: position in the league by that rating alone",
             fontsize=8, color=cs.MUTED, ha="right", va="bottom")
    return fig