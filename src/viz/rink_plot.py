"""
rink_plot.py
============
Pretty NHL rink rendering + game event plotting.

Intended home in the project: src/viz/rink_plot.py

Two public functions:

  draw_rink(ax=None, **style)        -> ax with a regulation NHL rink
  plot_game_events(events, ...)      -> rink + events from a PBP DataFrame

`plot_game_events` expects the schema produced by nhl_pbp_loader.py
(columns: event_type, team_id, x, y, period, time_in_period, is_rush,
is_rebound, shooter_id ...) but only event_type / team_id / x / y are
strictly required.

Coordinate system: NHL API convention. x in [-100, 100] (ft, center ice
origin), y in [-42.5, 42.5]. Rink is 200 x 85 ft, corner radius 28 ft.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------

ICE = "#fbfdfe"
BOARDS = "#34495e"
RED = "#d96b6b"          # softened NHL red for lines
BLUE = "#6b8fd9"         # softened NHL blue
CREASE = "#cfe3f5"
FAINT = 0.9

GOAL_COLOR = "#f0b429"   # accent for goal markers
RUSH_RING = "#2ec4b6"
REBOUND_RING = "#9b5de5"

EVENT_STYLE = {
    # event_type: (marker, base size, alpha, zorder)
    "goal":         ("*", 480, 1.00, 7),
    "shot-on-goal": ("o", 130, 0.85, 6),
    "missed-shot":  ("o", 110, 0.45, 5),
    "blocked-shot": ("s",  75, 0.40, 4),
    "hit":          ("^",  70, 0.30, 3),
    "takeaway":     ("D",  55, 0.30, 3),
    "giveaway":     ("d",  55, 0.30, 3),
}

EVENT_LABEL = {
    "goal": "Goal", "shot-on-goal": "Shot on goal", "missed-shot": "Missed shot",
    "blocked-shot": "Blocked shot", "hit": "Hit", "takeaway": "Takeaway",
    "giveaway": "Giveaway",
}


# ----------------------------------------------------------------------------
# Rink geometry
# ----------------------------------------------------------------------------

def _faceoff_circle(ax, x, y, with_hashes=True):
    ax.add_patch(Circle((x, y), 15, fill=False, ec=RED, lw=1.4, alpha=FAINT, zorder=1))
    ax.add_patch(Circle((x, y), 0.7, fc=RED, ec="none", alpha=FAINT, zorder=1))
    if with_hashes:
        for sx in (-1, 1):
            for sy in (-1, 1):
                ax.plot([x + sx * 2.9, x + sx * 2.9],
                        [y + sy * 15, y + sy * 17],
                        color=RED, lw=1.4, alpha=FAINT * 0.8, zorder=1,
                        solid_capstyle="butt")


def draw_rink(ax=None, figsize=(14, 6.4)):
    """Draw a regulation NHL rink. Returns the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # ice surface + boards (rounded rectangle, 28 ft corner radius)
    rink = FancyBboxPatch(
        (-100, -42.5), 200, 85,
        boxstyle="round,pad=0,rounding_size=28",
        fc=ICE, ec=BOARDS, lw=3, zorder=0, mutation_aspect=1,
    )
    ax.add_patch(rink)

    # clip everything ice-bound to the boards
    clip = rink

    def add(patch):
        patch.set_clip_path(clip)
        ax.add_patch(patch)
        return patch

    # center red line + blue lines + goal lines
    add(Rectangle((-0.5, -42.5), 1.0, 85, fc=RED, ec="none", alpha=FAINT, zorder=1))
    for x in (-25.5, 24.5):
        add(Rectangle((x, -42.5), 1.0, 85, fc=BLUE, ec="none", alpha=FAINT, zorder=1))
    for x in (-89, 89):
        ln, = ax.plot([x, x], [-42.5, 42.5], color=RED, lw=1.6, alpha=FAINT, zorder=1)
        ln.set_clip_path(clip)

    # center circle + dot
    add(Circle((0, 0), 15, fill=False, ec=BLUE, lw=1.4, alpha=FAINT, zorder=1))
    add(Circle((0, 0), 0.7, fc=BLUE, ec="none", alpha=FAINT, zorder=1))

    # end-zone faceoff circles, neutral-zone dots
    for x in (-69, 69):
        for y in (-22, 22):
            _faceoff_circle(ax, x, y)
    for x in (-20, 20):
        for y in (-22, 22):
            ax.add_patch(Circle((x, y), 0.7, fc=RED, ec="none", alpha=FAINT, zorder=1))

    for s in (-1, 1):
        gx = 89 * s
        # crease: rectangle + arc, light blue fill
        ax.add_patch(Polygon(
            [[gx, -4], [gx - 4.5 * s, -4], [gx - 4.5 * s, 4], [gx, 4]],
            fc=CREASE, ec=RED, lw=1.2, alpha=0.95, zorder=1))
        ax.add_patch(Arc((gx - 4.5 * s, 0), 3.2, 8, theta1=90 if s > 0 else 270,
                         theta2=270 if s > 0 else 90, ec=RED, lw=1.2, zorder=1))
        # goal frame (6 ft wide, ~3.3 ft deep) behind the goal line
        ax.add_patch(Rectangle((gx if s > 0 else gx - 3.3 * 0, -3),
                               3.3 * s, 6, fc="none", ec=BOARDS, lw=1.6, zorder=2))
        # trapezoid
        for sy in (-1, 1):
            tz, = ax.plot([gx, 100 * s], [11 * sy, 14.5 * sy],
                          color=RED, lw=1.2, alpha=FAINT * 0.8, zorder=1)
            tz.set_clip_path(clip)

    ax.set_xlim(-104, 104)
    ax.set_ylim(-46.5, 46.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ----------------------------------------------------------------------------
# Event plotting
# ----------------------------------------------------------------------------

def _normalize_attack_sides(ev: pd.DataFrame, home_id, right_team_id) -> pd.DataFrame:
    """
    Flip coordinates per (team, period) so `right_team_id` always attacks the
    right net and the other team the left net. Heuristic: a team's unblocked
    shot attempts cluster in its offensive zone, so the sign of their mean
    shot x reveals which way they were attacking that period.
    """
    ev = ev.copy()
    shot_mask = ev["event_type"].isin(["goal", "shot-on-goal", "missed-shot"])
    for (team, period), grp in ev[shot_mask].groupby(["team_id", "period"]):
        if grp["x"].notna().sum() == 0:
            continue
        attacking_right = grp["x"].mean() > 0
        want_right = team == right_team_id
        if attacking_right != want_right:
            idx = (ev["team_id"] == team) & (ev["period"] == period)
            # flip ALL of that team's events that period (x and y, a 180° turn)
            ev.loc[idx, ["x", "y"]] *= -1
    return ev


def plot_game_events(
    events: pd.DataFrame,
    home_team: str = "Home",
    away_team: str = "Away",
    home_color: str = "#6f263d",
    away_color: str = "#236192",
    home_team_id=None,
    title: str | None = None,
    subtitle: str | None = None,
    event_types: tuple = ("goal", "shot-on-goal", "missed-shot", "blocked-shot"),
    highlight_flags: bool = True,
    normalize_sides: bool = True,
    annotate_goals: bool = True,
    ax=None,
):
    """
    Plot game events on a rink.

    events           PBP DataFrame (nhl_pbp_loader schema). Needs event_type,
                     team_id, x, y; uses period/time/is_rush/is_rebound if present.
    home_team_id     team_id treated as home; inferred from `home_team_id`
                     column if absent.
    event_types      which events to draw, in draw order.
    highlight_flags  ring rush chances (teal) and rebounds (violet).
    normalize_sides  flip per-period coordinates so home always attacks right.
    """
    ev = events.dropna(subset=["x", "y"]).copy()
    ev = ev[ev["event_type"].isin(event_types)]
    if home_team_id is None and "home_team_id" in ev.columns and len(ev):
        home_team_id = ev["home_team_id"].iloc[0]

    if normalize_sides and {"period", "team_id"} <= set(ev.columns):
        ev = _normalize_attack_sides(ev, home_team_id, right_team_id=home_team_id)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7.6))
        fig.patch.set_facecolor("white")
    draw_rink(ax)

    team_color = lambda tid: home_color if tid == home_team_id else away_color

    for etype in event_types:
        sub = ev[ev["event_type"] == etype]
        if sub.empty:
            continue
        marker, size, alpha, z = EVENT_STYLE.get(etype, ("o", 80, 0.5, 3))
        colors = [team_color(t) for t in sub["team_id"]]
        ax.scatter(sub["x"], sub["y"], s=size, marker=marker, c=colors,
                   alpha=alpha, zorder=z, linewidths=1.2,
                   edgecolors=GOAL_COLOR if etype == "goal" else "white")

        # flag rings for unblocked shots
        if highlight_flags and etype in ("goal", "shot-on-goal", "missed-shot"):
            for ring_col, ring_color in (("is_rush", RUSH_RING),
                                         ("is_rebound", REBOUND_RING)):
                if ring_col in sub.columns:
                    ringed = sub[sub[ring_col].fillna(False).astype(bool)]
                    if len(ringed):
                        ax.scatter(ringed["x"], ringed["y"], s=size * 3.1,
                                   facecolors="none", edgecolors=ring_color,
                                   linewidths=1.8, zorder=z + 0.5, alpha=0.95)

    # annotate goals with period + clock
    if annotate_goals and {"period", "time_in_period"} <= set(ev.columns):
        # cycle offsets so labels on clustered goals don't collide
        offsets = [(0, 16), (0, -26), (42, 6), (-42, 6), (42, -18), (-42, -18)]
        goals_sorted = ev[ev["event_type"] == "goal"].sort_values(["x", "y"])
        for k, (_, g) in enumerate(goals_sorted.iterrows()):
            t = g["time_in_period"]
            clock = f"P{int(g['period'])} {int(t // 60):02d}:{int(t % 60):02d}"
            ax.annotate(clock, (g["x"], g["y"]), xytext=offsets[k % len(offsets)],
                        textcoords="offset points", ha="center",
                        fontsize=8.5, fontweight="bold", color="#33414e",
                        zorder=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#d7dee5", lw=0.8, alpha=0.9))

    # titles + attack-direction hints
    if title:
        ax.text(0, 50.5, title, ha="center", fontsize=17, fontweight="bold",
                color="#22303c")
    if subtitle:
        ax.text(0, 46.3, subtitle, ha="center", fontsize=10.5, color="#7a8794")
    if normalize_sides:
        ax.text(60, -45.6, f"{home_team} attacking \u2192", ha="center",
                fontsize=9.5, color=home_color, fontweight="bold")
        ax.text(-60, -45.6, f"\u2190 {away_team} attacking", ha="center",
                fontsize=9.5, color=away_color, fontweight="bold")
    ax.set_ylim(-49, 54 if title else 47)

    # legend: event glyphs (neutral gray) + team swatches + flag rings
    handles = [
        Line2D([], [], marker=EVENT_STYLE[e][0], color="none",
               markerfacecolor="#8a97a3", markeredgecolor="white",
               markersize=np.sqrt(EVENT_STYLE[e][1]) * 0.85,
               label=EVENT_LABEL.get(e, e))
        for e in event_types if e in EVENT_STYLE
    ]
    handles += [
        Line2D([], [], marker="s", color="none", markerfacecolor=home_color,
               markersize=10, label=home_team),
        Line2D([], [], marker="s", color="none", markerfacecolor=away_color,
               markersize=10, label=away_team),
    ]
    if highlight_flags:
        handles += [
            Line2D([], [], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=RUSH_RING, markeredgewidth=1.8,
                   markersize=12, label="Rush chance"),
            Line2D([], [], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=REBOUND_RING, markeredgewidth=1.8,
                   markersize=12, label="Rebound"),
        ]
    ax.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
              bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9.5,
              handletextpad=0.4, columnspacing=1.3)

    return ax


# ----------------------------------------------------------------------------
# Database convenience wrapper (hockey-stats schema)
# ----------------------------------------------------------------------------

def _hex_dist(a: str, b: str) -> float:
    """Euclidean RGB distance between two hex colors."""
    a, b = a.lstrip("#"), b.lstrip("#")
    ra, ga, ba = (int(a[i:i + 2], 16) for i in (0, 2, 4))
    rb, gb, bb = (int(b[i:i + 2], 16) for i in (0, 2, 4))
    return ((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5


def plot_game(conn, game_id: int, ax=None, **kwargs):
    """
    One-call game plot straight from the hockey_stats.db schema.

    Reads shot_events LEFT JOIN shot_context (rebound/rush rings appear
    automatically once backfill_game_events.py has run), pulls team names,
    colors, and the final score from the teams / schedule / games tables.

    Extra kwargs pass through to plot_game_events (event_types, etc.).
    """
    meta = conn.execute("""
        SELECT s.home_team_abbrev, s.away_team_abbrev,
               s.home_team_id, s.away_team_id, s.game_date,
               th.common_name, ta.common_name,
               th.primary_color, ta.primary_color, ta.secondary_color,
               g.home_score, g.away_score
        FROM schedule s
        JOIN teams th ON th.abbreviation = s.home_team_abbrev
        JOIN teams ta ON ta.abbreviation = s.away_team_abbrev
        LEFT JOIN games g ON g.id = s.game_id
        WHERE s.game_id = ?
    """, (game_id,)).fetchone()
    if meta is None:
        raise ValueError(f"game_id {game_id} not found in schedule")
    (home_ab, away_ab, home_id, away_id, game_date, home_name, away_name,
     home_color, away_color, away_secondary, home_score, away_score) = meta

    # if the two primaries are too close to tell apart (e.g. TOR vs TBL navy),
    # fall back to the away team's secondary color
    if _hex_dist(home_color, away_color) < 60:
        away_color = away_secondary
        if _hex_dist(away_color, "#FFFFFF") < 60:   # near-white won't read on ice
            away_color = "#7f8a94"

    ev = pd.read_sql("""
        SELECT se.event_type, se.team_id, se.period, se.time_in_period,
               se.x_coord AS x, se.y_coord AS y,
               sc.is_rebound, sc.is_rush
        FROM shot_events se
        LEFT JOIN shot_context sc
               ON sc.game_id = se.game_id AND sc.event_id = se.event_id
        WHERE se.game_id = ? AND se.period <= 4
    """, conn, params=(game_id,))

    # "MM:SS" -> seconds for the goal annotations
    mins_secs = ev["time_in_period"].str.split(":", expand=True).astype(int)
    ev["time_in_period"] = mins_secs[0] * 60 + mins_secs[1]
    ev["home_team_id"] = home_id

    title = f"{home_name} {home_score} \u2014 {away_score} {away_name}" \
        if home_score is not None else f"{home_name} vs {away_name}"

    return plot_game_events(
        ev,
        home_team=home_name, away_team=away_name,
        home_color=home_color, away_color=away_color,
        home_team_id=home_id,
        title=title,
        subtitle=f"{away_ab} @ {home_ab} \u00b7 {game_date} \u00b7 all shot attempts",
        ax=ax,
        **kwargs,
    )


if __name__ == "__main__":
    # tiny smoke test: empty rink
    draw_rink()
    plt.savefig("rink_demo.png", dpi=160, bbox_inches="tight")
