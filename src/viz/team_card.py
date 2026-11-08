"""Team card (PNG): where a team stands in the model, how it got there, and its playoff odds.

Home: src/viz/team_card.py

Built by src/pipelines/make_team_card.py from the active fit, the database and the playoff
odds history. Every rating is shown as "% better than an average team":
    offense  exp(mu_off) - 1   goals scored vs average (positive = scores more)
    defense  exp(mu_def) - 1   goals kept out vs average (positive = allows fewer)
Bands are 90% posterior ranges. Weekly paths come from the fit's random walk
(xg_mu_*_path + phi_*); the last point is the rating as of the fit date.

Usage::

    from src.viz.team_card import TeamCardData, plot_team_card
    plot_team_card(data).savefig("output/teams/BOS_2026-11-08.png", dpi=200)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import card_style as cs

TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montréal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders", "NYR": "New York Rangers",
    "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "STL": "St. Louis Blues",
    "SJS": "San Jose Sharks", "SEA": "Seattle Kraken", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}


@dataclass
class TeamCardData:
    team: str
    date: str                                   # card date (fit's ratings_as_of or today)
    division: str = ""
    conference: str = ""
    record: dict | None = None                  # gp, w, l, otl, pts
    # weekly paths, "% better than average": DataFrame date, off_mid, off_lo, off_hi, def_mid, def_lo, def_hi,
    # league_off_lo/hi, league_def_lo/hi (the spread of all teams' means), season
    path: pd.DataFrame | None = None
    ranks: dict = field(default_factory=dict)   # "net"/"off"/"def" -> (position, lo, hi), 1 = best
    n_teams: int = 32
    components: list = field(default_factory=list)   # (label, mid, lo, hi) in "% better", positive = good
    goalies: list = field(default_factory=list)      # dicts name, share, starts, effect, lo, hi
    playoff_hist: pd.DataFrame | None = None    # date, p_playoffs, points_mean, points_lo, points_hi
    model_label: str = ""
    color: str | None = None


def _ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _pp(x: float, digits: int = 1) -> str:
    return f"{100 * x:+.{digits}f}%".replace("-", "−")


def _team_color(team: str) -> str:
    opts = cs._team_colors(team) or [cs.HOME]
    opts = sorted(opts, key=lambda c: 1 if cs.oklab(c)[0] > 0.84 else 0)
    return opts[0]


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def _header(fig, d: TeamCardData, c: str):
    fig.text(0.04, 0.972, d.team, fontsize=30, fontweight="bold", color=cs.ink_for(c), va="top")
    name = TEAM_NAMES.get(d.team, "")
    sub = " · ".join(x for x in (name, f"{d.division} Division" if d.division else "",
                                 f"{d.conference} Conference" if d.conference else "") if x)
    fig.text(0.04, 0.925, sub.replace("Conference Conference", "Conference"), fontsize=10.5,
             color=cs.INK_2, va="top")
    fig.text(0.96, 0.970, "TEAM CARD", fontsize=10, color=cs.MUTED, fontweight="bold", ha="right", va="top")
    fig.text(0.96, 0.950, d.date, fontsize=13, color=cs.INK, ha="right", va="top")
    if d.record:
        r = d.record
        fig.text(0.96, 0.925, f"{int(r['w'])}-{int(r['l'])}-{int(r['otl'])} · {int(r['pts'])} pts in {int(r['gp'])} GP",
                 fontsize=10.5, color=cs.INK_2, ha="right", va="top")


def _tiles(fig, d: TeamCardData, c: str, rect):
    x0, y0, w, h = rect
    tiles = []
    for key, label in (("net", "Overall"), ("off", "Offense"), ("def", "Defense")):
        if key in d.ranks:
            m, lo, hi = d.ranks[key]
            tiles.append((f"{label} rank", _ordinal(m), f"likely {_ordinal(lo)}–{_ordinal(hi)} of {d.n_teams}"))
    ph = d.playoff_hist
    if ph is not None and len(ph):
        last = ph.sort_values("date").iloc[-1]
        p = float(last["p_playoffs"])
        ptxt = ">99.9%" if p > 0.999 else "<0.1%" if p < 0.001 else cs.pct(p)
        tiles.append(("Make playoffs", ptxt, f"odds of {pd.Timestamp(last['date']).date()}"))
        tiles.append(("Projected points", f"{last['points_mean']:.0f}",
                      f"90% range {last['points_lo']:.0f}–{last['points_hi']:.0f}"))
    if not tiles:
        return
    gap = 0.012
    tw = (w - gap * (len(tiles) - 1)) / len(tiles)
    for i, (label, value, sub) in enumerate(tiles):
        tx = x0 + i * (tw + gap)
        fig.patches.append(Rectangle((tx, y0), tw, h, transform=fig.transFigure, facecolor=cs.PAGE,
                                     edgecolor="none", zorder=-5))
        fig.text(tx + 0.012, y0 + h - 0.010, label, fontsize=9, color=cs.INK_2, va="top")
        fig.text(tx + 0.012, y0 + h * 0.40, value, fontsize=21, fontweight="bold",
                 color=cs.ink_for(c) if i == 0 else cs.INK, va="center")
        fig.text(tx + 0.012, y0 + 0.010, sub, fontsize=8.5, color=cs.MUTED, va="bottom")


SUMMER_SHOWN_DAYS = 21                      # an off-season is drawn as a 3-week gap


def _with_x(p: pd.DataFrame) -> pd.DataFrame:
    """x = days since the first step, with each off-season squeezed to SUMMER_SHOWN_DAYS."""
    p = p.sort_values("date").reset_index(drop=True)
    gaps = p["date"].diff().dt.days.fillna(0).to_numpy()
    new_season = np.array(p["season"] != p["season"].shift())
    new_season[0] = False
    gaps = np.where(new_season, SUMMER_SHOWN_DAYS, gaps)
    return p.assign(x=np.cumsum(gaps))


def _segments(p: pd.DataFrame):
    """Rows split at off-season gaps (one segment per season)."""
    for _, g in p.groupby("season", sort=False):
        yield g


def _month_ticks(p: pd.DataFrame):
    """(x, label) at the 1st of Oct / Dec / Feb / Apr inside each season's segment."""
    ticks = []
    for g in _segments(p):
        for m in pd.date_range(g["date"].min().to_period("M").to_timestamp(), g["date"].max(), freq="MS"):
            if m < g["date"].min() or m.month not in (10, 12, 2, 4):
                continue
            x = np.interp(m.value, g["date"].astype("datetime64[ns]").astype("int64"), g["x"])
            ticks.append((x, m.strftime("%b %Y") if m.month in (10, 2) else m.strftime("%b")))
    return ticks


def _path_panel(fig, rect, d: TeamCardData, c: str, key: str, title: str, subtitle: str):
    p = _with_x(d.path)
    ax = fig.add_axes(rect)
    ax.set_title(title, pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8.5, color=cs.MUTED, va="bottom")
    for g in _segments(p):
        ax.fill_between(g["x"], g[f"league_{key}_lo"], g[f"league_{key}_hi"], color=cs.GRID, alpha=0.55,
                        linewidth=0, zorder=1)
        ax.fill_between(g["x"], g[f"{key}_lo"], g[f"{key}_hi"], color=cs.tint(c, 0.62), linewidth=0, zorder=2)
        ax.plot(g["x"], g[f"{key}_mid"], color=c, lw=2, zorder=3)
    ax.axhline(0, color=cs.AXIS, lw=0.9, zorder=2)
    last = p.iloc[-1]
    ax.plot([last["x"]], [last[f"{key}_mid"]], "o", color=c, ms=6, mec=cs.SURFACE, mew=1.5, zorder=4)
    ax.annotate(f"{_pp(last[f'{key}_mid'])}\n({_pp(last[f'{key}_lo'], 0)} to {_pp(last[f'{key}_hi'], 0)})",
                (last["x"], last[f"{key}_mid"]), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=9, fontweight="bold", color=cs.INK, linespacing=1.3, annotation_clip=False)
    # season labels at the top of each segment
    for g in _segments(p):
        ax.text(g["x"].iloc[0], 1.0, f"  {g['season'].iloc[0]}", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=cs.MUTED, va="top")
    lo = min(p[f"league_{key}_lo"].min(), p[f"{key}_lo"].min())
    hi = max(p[f"league_{key}_hi"].max(), p[f"{key}_hi"].max())
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.yaxis.set_major_formatter(lambda v, _: _pp(v, 0))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ticks = _month_ticks(p)
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([lab for _, lab in ticks])
    ax.set_xlim(p["x"].min() - 4, p["x"].max() + 4)


def _components(fig, rect, d: TeamCardData, c: str):
    """Dot + 90% whisker per component, positive = good for the team."""
    x0, y0, w, h = rect
    fig.text(x0, y0 + h, "Inside the ratings", fontsize=11, fontweight="bold", va="top")
    fig.text(x0, y0 + h - 0.020, "% better than an average team now · 90% range · right = good",
             fontsize=8.5, color=cs.MUTED, va="top")
    comps = d.components
    n_rows = len(comps)
    ax = fig.add_axes([x0 + 0.20, y0 + h - 0.045 - 0.029 * n_rows, w - 0.215, 0.029 * n_rows])
    ax.set_facecolor("none")
    lim = max([abs(v) for _, m, lo, hi in comps for v in (lo, hi)] + [0.05]) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, n_rows - 0.4)
    for i, (label, m, lo, hi) in enumerate(comps):
        y = n_rows - 1 - i
        ax.plot([lo, hi], [y, y], color=cs.tint(c, 0.45), lw=4, solid_capstyle="round", zorder=2)
        ax.plot([m], [y], "o", color=c, ms=8, mec=cs.SURFACE, mew=1.5, zorder=3)
        ax.text(lim * 1.02, y, _pp(m), va="center", ha="left", fontsize=9, fontweight="bold", color=cs.INK)
        yy = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
        fig.text(x0, yy, label, fontsize=9, color=cs.INK_2, va="center")
    ax.axvline(0, color=cs.AXIS, lw=0.9, zorder=1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda v, _: _pp(v, 0))
    ax.tick_params(axis="x", labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # goalies
    gy = y0 + h - 0.045 - 0.029 * n_rows - 0.038
    fig.text(x0, gy, "Goalies (share of the last 20 starts)", fontsize=10, fontweight="bold", va="top")
    fig.text(x0, gy - 0.018, "goals kept out vs expected from the shots faced (Phase 3 rating), 90% range",
             fontsize=8.5, color=cs.MUTED, va="top")
    if not d.goalies:
        fig.text(x0, gy - 0.042, "no starts found", fontsize=9, color=cs.MUTED, va="top")
    for j, g in enumerate(d.goalies[:3]):
        ty = gy - 0.043 - j * 0.022
        fig.text(x0, ty, f"{g['name']}", fontsize=9.5, fontweight="bold", color=cs.INK, va="top")
        fig.text(x0 + 0.155, ty, f"{g['share']:.0%} ({int(g['starts'])} GS)", fontsize=9, color=cs.INK_2, va="top")
        if g.get("effect") is not None:
            fig.text(x0 + w, ty, f"{_pp(g['effect'])}  ({_pp(g['lo'], 0)} to {_pp(g['hi'], 0)})",
                     fontsize=9, color=cs.INK, ha="right", va="top")
        else:
            fig.text(x0 + w, ty, "no rating yet", fontsize=9, color=cs.MUTED, ha="right", va="top")


def _playoffs(fig, rect, d: TeamCardData, c: str):
    x0, y0, w, h = rect
    ph = d.playoff_hist
    fig.text(x0, y0 + h, "Playoff odds this season", fontsize=11, fontweight="bold", va="top")
    if ph is None or not len(ph):
        fig.text(x0, y0 + h - 0.022, "no playoff_odds runs yet", fontsize=8.5, color=cs.MUTED, va="top")
        return
    ph = ph.sort_values("date")
    fig.text(x0, y0 + h - 0.020, f"{len(ph)} weekly run{'s' if len(ph) != 1 else ''} of playoff_odds",
             fontsize=8.5, color=cs.MUTED, va="top")
    ax = fig.add_axes([x0 + 0.045, y0 + 0.03, w - 0.09, h - 0.085])
    dates = pd.to_datetime(ph["date"])
    ax.plot(dates, ph["p_playoffs"], color=c, lw=2, marker="o" if len(ph) <= 15 else None, ms=5,
            mec=cs.SURFACE, mew=1.2, zorder=3)
    ax.axhline(0.5, color=cs.AXIS, lw=0.8)
    last = ph.iloc[-1]
    ax.annotate(cs.pct(float(last["p_playoffs"]), 0) if 0.001 < last["p_playoffs"] < 0.999 else
                (">99.9%" if last["p_playoffs"] >= 0.999 else "<0.1%"),
                (dates.iloc[-1], last["p_playoffs"]), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=9.5, fontweight="bold", color=cs.INK, annotation_clip=False)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    span = (dates.max() - dates.min()).days
    if span == 0:
        ax.set_xlim(dates.iloc[0] - pd.Timedelta(days=7), dates.iloc[0] + pd.Timedelta(days=7))
        ax.set_xticks([dates.iloc[0]])
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


# --------------------------------------------------------------------------
# card
# --------------------------------------------------------------------------

def plot_team_card(d: TeamCardData):
    c = d.color or _team_color(d.team)
    fig = cs.new_figure(12, 14)
    cs.card_background(fig, [
        (0.025, 0.775, 0.975, 0.895),      # tiles
        (0.025, 0.375, 0.975, 0.76),       # rating paths
        (0.025, 0.045, 0.555, 0.36),       # inside the ratings
        (0.57, 0.045, 0.975, 0.36),        # playoff odds
    ])
    _header(fig, d, c)
    _tiles(fig, d, c, (0.045, 0.787, 0.91, 0.094))
    if d.path is not None and len(d.path):
        _path_panel(fig, (0.075, 0.592, 0.80, 0.122), d, c, "off", "Offense",
                    "goals scored vs an average team, weekly · band = 90% range · gray = spread of all teams")
        _path_panel(fig, (0.075, 0.405, 0.80, 0.122), d, c, "def", "Defense",
                    "goals kept out vs an average team, starting goalie excluded (rated separately below) · positive = allows fewer")
    _components(fig, (0.045, 0.06, 0.49, 0.285), d, c)
    _playoffs(fig, (0.59, 0.06, 0.37, 0.285), d, c)
    fig.text(0.04, 0.02, d.model_label or "Track B", fontsize=8, color=cs.MUTED, va="bottom")
    fig.text(0.96, 0.02, "ratings: weekly random walk of the active fit · goalies: Phase 3 goalie model",
             fontsize=8, color=cs.MUTED, ha="right", va="bottom")
    return fig