"""Postgame card (PNG): shot map, deserve-to-win meter, xG worm, model vs
result, top shooters and goalies.

Input is one game's shots from ``shot_xg`` (unblocked, goalie in net,
periods 1-4) as a DataFrame with columns:

    team      team abbreviation of the shooter
    period    1-4
    seconds   game seconds elapsed (0-3600 regulation, 3600+ OT)
    x, y      raw feed coordinates (ends switch by period; normalized here)
    xg        shot xG
    is_goal   bool
    shooter   display name
    goalie    display name of the goalie in net

``load_game_shots(conn, game_id)`` builds it from the database;
``src/pipelines/make_cards.py`` does the whole card from a game id.

Usage::

    from src.viz.postgame_card import PostgameData, plot_postgame_card
    fig = plot_postgame_card(data)
    fig.savefig("output/game_summaries/2026-04-10_DAL_at_COL_postgame.png", dpi=200)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle, Wedge

from ..analysis.deserved import deserve_to_win
from . import card_style as cs


@dataclass
class PostgameData:
    home: str
    away: str
    date: str
    home_goals: int                     # official final score
    away_goals: int
    ended: str                          # "REG", "OT" or "SO"
    shots: pd.DataFrame
    goalie_toi: dict = field(default_factory=dict)   # {"name": "60:00"}
    empty_net_goals: list = field(default_factory=list)  # [(team, seconds)]
    pregame_p_home: float | None = None              # from the forward-test log
    pregame_interval: tuple | None = None            # (lo, hi)
    model_label: str = ""
    note: str = ""
    n_sims: int = 10_000
    heat_weight: str = "xg"             # "xg" (danger) or "count" (like NST)
    heat_bandwidth: float = 6.0         # kernel width in feet
    colors: tuple | None = None         # (home, away); None = team colours


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _clock_seconds(period, time_in_period) -> np.ndarray:
    """Game seconds elapsed from period + "MM:SS" elapsed in the period."""
    mmss = pd.Series(time_in_period, dtype="string").str.split(":", expand=True)
    sec = mmss[0].astype(float) * 60 + mmss[1].astype(float)
    return ((np.asarray(period, float) - 1) * 1200 + sec).to_numpy()


def player_names(conn, game_id: int) -> dict:
    """player_id -> display name ("N. MacKinnon") for everyone in one game.

    Boxscore names come first (they cover traded and retired players); the
    ``players`` table fills any gaps.
    """
    names: dict[int, str] = {}
    pl = pd.read_sql_query("SELECT id, first_name, last_name FROM players", conn)
    for r in pl.itertuples():
        if r.last_name:
            first = f"{r.first_name[0]}. " if r.first_name else ""
            names[int(r.id)] = f"{first}{r.last_name}"
    for table in ("skater_game_stats", "goalie_game_stats"):
        cols = set(pd.read_sql_query(f"PRAGMA table_info({table})", conn)["name"])
        if {"game_id", "player_id", "name"} <= cols:
            box = pd.read_sql_query(
                f"SELECT player_id, name FROM {table} WHERE game_id = ?",
                conn, params=(game_id,))
            names.update({int(i): n for i, n in zip(box.player_id, box.name) if n})
    return names


def load_game_shots(conn, game_id: int) -> pd.DataFrame:
    """One game's shots for the card (rows of ``shot_xg``: unblocked,
    goalie in net, periods 1-4), in play order."""
    sql = """
        SELECT se.event_id, se.period, se.time_in_period,
               se.x_coord AS x, se.y_coord AS y,
               sx.xg,
               se.event_type,
               se.team_id,
               se.shooting_player_id, se.goalie_id,
               ge.sort_order,
               CASE se.team_id WHEN sc.home_team_id THEN sc.home_team_abbrev
                               WHEN sc.away_team_id THEN sc.away_team_abbrev
               END AS team
        FROM shot_xg sx
        JOIN shot_events se ON se.game_id = sx.game_id AND se.event_id = sx.event_id
        JOIN schedule sc ON sc.game_id = sx.game_id
        LEFT JOIN game_events ge ON ge.game_id = se.game_id
                                AND ge.event_id = se.event_id
        WHERE sx.game_id = ?
    """
    df = pd.read_sql_query(sql, conn, params=(game_id,))
    if df.empty:
        return df
    df["seconds"] = _clock_seconds(df["period"], df["time_in_period"])
    df["is_goal"] = df["event_type"].eq("goal")
    names = player_names(conn, game_id)
    df["shooter"] = [names.get(int(i), f"#{int(i)}") if pd.notna(i) else "unknown"
                     for i in df["shooting_player_id"]]
    df["goalie"] = [names.get(int(i), f"#{int(i)}") if pd.notna(i) else "unknown"
                    for i in df["goalie_id"]]
    df = df.sort_values(["sort_order", "seconds"], na_position="last")
    return df[["team", "period", "seconds", "x", "y", "xg", "is_goal",
               "shooter", "goalie", "event_id"]].reset_index(drop=True)


def load_empty_net_goals(conn, game_id: int) -> list[tuple[str, float]]:
    """(team, game seconds) of empty-net goals in periods 1-4.

    Empty net = the opposing goalie digit in ``situation_code`` is 0
    (away goalie is the 1st digit, home goalie the 4th).
    """
    sql = """
        SELECT se.period, se.time_in_period, se.team_id, ge.situation_code,
               sc.home_team_id, sc.home_team_abbrev, sc.away_team_abbrev
        FROM shot_events se
        JOIN schedule sc ON sc.game_id = se.game_id
        LEFT JOIN game_events ge ON ge.game_id = se.game_id
                                AND ge.event_id = se.event_id
        WHERE se.game_id = ? AND se.event_type = 'goal' AND se.period <= 4
    """
    g = pd.read_sql_query(sql, conn, params=(game_id,))
    out = []
    for r in g.itertuples():
        code = str(r.situation_code or "")
        if len(code) != 4:
            continue
        home_shot = r.team_id == r.home_team_id
        opp_goalie = code[0] if home_shot else code[3]
        if opp_goalie == "0":
            team = r.home_team_abbrev if home_shot else r.away_team_abbrev
            sec = _clock_seconds([r.period], [r.time_in_period])[0]
            out.append((team, float(sec)))
    return out


def normalize_coords(shots: pd.DataFrame, home: str) -> pd.DataFrame:
    """Home shots attack the right net (x > 0), away shots the left.

    Teams switch ends each period, so the attacking direction is found per
    team and period from where most of its shots were taken.
    """
    df = shots.copy()
    flip = df.groupby(["team", "period"])["x"].transform("median") < 0
    df.loc[flip, ["x", "y"]] *= -1
    away = df["team"] != home
    df.loc[away, ["x", "y"]] *= -1
    return df


# --------------------------------------------------------------------------
# rink
# --------------------------------------------------------------------------

RED_LINE = "#e3a7a7"
BLUE_LINE = "#a9c6ec"
RINK_LINE = "#d9d7cf"


def draw_rink(ax):
    """Full NHL rink, center ice at (0, 0), x in [-100, 100] feet.

    Returns the boards patch, so a heat map can be clipped to the rink.
    """
    boards = FancyBboxPatch((-100, -42.5), 200, 85,
                            boxstyle="round,pad=0,rounding_size=28",
                            facecolor="#ffffff", edgecolor=cs.AXIS, lw=1.6,
                            zorder=0)
    ax.add_patch(boards)
    ax.plot([0, 0], [-42.5, 42.5], color=RED_LINE, lw=2.6, zorder=1)
    for x in (-25, 25):
        ax.plot([x, x], [-42.5, 42.5], color=BLUE_LINE, lw=2.6, zorder=1)
    for x in (-89, 89):
        # goal line stops at the curved boards
        ax.plot([x, x], [-37.0, 37.0], color=RED_LINE, lw=1.0, zorder=1)
        sign = np.sign(x)
        ax.add_patch(Arc((x, 0), 12, 12, theta1=90 if sign > 0 else -90,
                         theta2=270 if sign > 0 else 90, color=RED_LINE,
                         lw=1.0, zorder=1))
        ax.add_patch(Wedge((x, 0), 6, 90 if sign > 0 else -90,
                           270 if sign > 0 else 90, facecolor="#e8f1fc",
                           edgecolor="none", zorder=0.5))
        ax.add_patch(Rectangle((x, -3) if sign > 0 else (x - 3.33, -3),
                               3.33, 6, facecolor="none", edgecolor=cs.MUTED,
                               lw=1.0, zorder=1))
        for y in (-22, 22):
            ax.add_patch(Circle((sign * 69, y), 15, facecolor="none",
                                edgecolor=RED_LINE, lw=1.0, zorder=1))
            ax.add_patch(Circle((sign * 69, y), 1, color=RED_LINE, zorder=1))
            ax.add_patch(Circle((sign * 20, y), 1, color=RED_LINE, zorder=1))
    ax.add_patch(Circle((0, 0), 15, facecolor="none", edgecolor=BLUE_LINE,
                        lw=1.0, zorder=1))
    ax.add_patch(Circle((0, 0), 0.8, color=BLUE_LINE, zorder=1))
    ax.set_xlim(-101.5, 101.5)
    ax.set_ylim(-44, 44)
    ax.set_aspect("equal")
    ax.axis("off")
    return boards


# grid for the heat map (feet); 0.5 ft cells
_GX = np.linspace(-100, 100, 401)
_GY = np.linspace(-42.5, 42.5, 171)


def shot_density(x, y, w=None, bandwidth: float = 6.0) -> np.ndarray:
    """Gaussian kernel density of shot locations on the rink grid.

    ``w`` weights each shot (xG for a danger map, None for a count map).
    Returns a (len(_GY), len(_GX)) array in weight per square foot.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    if x.size == 0:
        return np.zeros((_GY.size, _GX.size))
    kx = np.exp(-0.5 * ((_GX[None, :] - x[:, None]) / bandwidth) ** 2)
    ky = np.exp(-0.5 * ((_GY[None, :] - y[:, None]) / bandwidth) ** 2)
    z = np.einsum("n,ny,nx->yx", w, ky, kx)
    return z / (2 * np.pi * bandwidth ** 2)


def _heat_rgba(z: np.ndarray, color: str, vmax: float) -> np.ndarray:
    """One-hue heat layer: light tint at low density, full colour at the peak,
    transparent where there is (almost) nothing."""
    from matplotlib.colors import to_rgb
    # square root so the low-density areas still show next to the slot
    t = np.sqrt(np.clip(z / vmax, 0, 1)) if vmax > 0 else np.zeros_like(z)
    lo = np.array(cs.tint(color, 0.7))
    hi = np.array(to_rgb(color)) * 0.78          # a darker step at the peak
    rgb = lo[None, None, :] + (hi - lo)[None, None, :] * t[..., None] ** 1.4
    alpha = np.clip((t - 0.10) / 0.35, 0, 1) * 0.92
    return np.dstack([rgb, alpha])


def _shot_map(fig, rect, d: PostgameData, shots: pd.DataFrame):
    """Heat map of each team's shots (xG-weighted by default), goals on top.

    Both teams share one colour scale, so equal shade = equal density.
    """
    hc, ac = d.colors
    ax = fig.add_axes(rect)
    boards = draw_rink(ax)

    layers = {}
    for team in (d.home, d.away):
        s = shots[shots["team"] == team]
        w = s["xg"] if d.heat_weight == "xg" else None
        layers[team] = shot_density(s["x"], s["y"], w, d.heat_bandwidth)
    vmax = max(z.max() for z in layers.values())

    extent = (_GX[0], _GX[-1], _GY[0], _GY[-1])
    for team, color in ((d.home, hc), (d.away, ac)):
        im = ax.imshow(_heat_rgba(layers[team], color, vmax), extent=extent,
                       origin="lower", interpolation="bilinear", zorder=0.8)
        im.set_clip_path(boards)

    for team, color in ((d.home, hc), (d.away, ac)):
        g = shots[(shots["team"] == team) & shots["is_goal"]]
        ax.scatter(g["x"], g["y"], s=95, facecolor=color, edgecolor="#ffffff",
                   linewidth=2.4, zorder=5)
        ax.scatter(g["x"], g["y"], s=95, facecolor="none", edgecolor=cs.INK,
                   linewidth=1.2, zorder=6)

    what = "xG-weighted" if d.heat_weight == "xg" else "shot count"
    ax.text(0, 1.02, "Shot heat map", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(1, 1.02, f"Unblocked shots with a goalie in net · {what} · "
                     "one shared scale",
            transform=ax.transAxes, fontsize=8.5, color=cs.MUTED, ha="right",
            va="bottom")

    # team labels under each end
    for team, color, x_ax, ha in ((d.away, ac, 0.0, "left"),
                                  (d.home, hc, 1.0, "right")):
        s = shots[shots["team"] == team]
        arrow = "← " if ha == "left" else ""
        tail = "" if ha == "left" else " →"
        ax.text(x_ax, -0.04, f"{arrow}{team} attacking{tail}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=cs.ink_for(color), ha=ha, va="top")
        ax.text(x_ax, -0.12,
                f"{len(s)} shots · {s['xg'].sum():.2f} xG · {int(s['is_goal'].sum())} G",
                transform=ax.transAxes, fontsize=9, color=cs.INK_2, ha=ha,
                va="top")

    # legend: one gradient strip per team + the goal marker
    ramp = np.linspace(0, 1, 120)[None, :] * vmax
    for i, (team, color) in enumerate(((d.away, ac), (d.home, hc))):
        bx = ax.inset_axes([0.33, -0.06 - i * 0.065, 0.22, 0.04])
        bx.imshow(_heat_rgba(ramp, color, vmax), aspect="auto",
                  extent=(0, 1, 0, 1))
        bx.set_xticks([])
        bx.set_yticks([])
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.text(-0.04, 0.5, team, transform=bx.transAxes, ha="right",
                va="center", fontsize=8.5, color=cs.ink_for(color),
                fontweight="bold")
    ax.text(0.33, -0.135, "less" if d.heat_weight == "xg" else "fewer", transform=ax.transAxes, fontsize=8,
            color=cs.MUTED, va="top")
    ax.text(0.55, -0.135, "more danger" if d.heat_weight == "xg" else "more shots",
            transform=ax.transAxes, fontsize=8, color=cs.MUTED, va="top",
            ha="right")
    ax.scatter([0.615], [-0.07], s=95, facecolor=cs.INK_2, edgecolor=cs.INK,
               linewidth=1.2, transform=ax.transAxes, clip_on=False)
    ax.text(0.633, -0.07, "goal", transform=ax.transAxes, fontsize=8.5,
            color=cs.INK_2, va="center")


def _meter(fig, rect, d: PostgameData, res: dict):
    """Semicircle gauge: home fills its share from the right, away from the
    left; the needle sits on the boundary. The 50% tick is at the top."""
    hc, ac = d.colors
    p = res["p_home"]
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.95, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    split = 180 * p                        # degrees from the +x axis
    gap = 0.8
    ax.add_patch(Wedge((0, 0), 1.0, 0, split - gap, width=0.28,
                       facecolor=hc, edgecolor="none"))
    ax.add_patch(Wedge((0, 0), 1.0, split + gap, 180, width=0.28,
                       facecolor=ac, edgecolor="none"))
    # 50% tick
    ax.plot([0, 0], [0.66, 1.06], color=cs.SURFACE, lw=2.2, zorder=3)
    ax.plot([0, 0], [1.03, 1.10], color=cs.MUTED, lw=1, zorder=3)
    # needle
    ang = np.radians(split)
    ax.plot([0, 0.80 * np.cos(ang)], [0, 0.80 * np.sin(ang)], color=cs.INK,
            lw=2.4, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((0, 0), 0.06, color=cs.INK, zorder=5))

    winner, wp, wc = (d.home, p, hc) if p >= 0.5 else (d.away, 1 - p, ac)
    ax.text(0, -0.22, cs.pct(wp), ha="center", va="top", fontsize=30,
            fontweight="bold", color=cs.ink_for(wc))
    ax.text(0, -0.56, f"{winner} deserved to win", ha="center", va="top",
            fontsize=10.5, color=cs.INK)
    ax.text(-1.0, -0.08, f"{d.away}\n{cs.pct(1 - p)}", ha="center", va="top",
            fontsize=9.5, color=cs.ink_for(ac), fontweight="bold",
            linespacing=1.2)
    ax.text(1.0, -0.08, f"{d.home}\n{cs.pct(p)}", ha="center", va="top",
            fontsize=9.5, color=cs.ink_for(hc), fontweight="bold",
            linespacing=1.2)
    ax.text(0, 1.2, "Deserve to win", ha="center", va="bottom", fontsize=11,
            fontweight="bold")
    ax.text(0, -0.76,
            f"Share of {res['n_sims']:,} replays of this game's shots won\n"
            f"(each shot scores with its xG; replays tied {cs.pct(res['p_tied'], 0)})",
            ha="center", va="top", fontsize=8.5, color=cs.MUTED,
            linespacing=1.35)


def _worm(fig, rect, d: PostgameData, shots: pd.DataFrame):
    hc, ac = d.colors
    ax = fig.add_axes(rect)
    end = max(3600.0, float(shots["seconds"].max()) + 30)
    if d.ended in ("OT", "SO"):
        end = max(end, 3900.0)
    top = 0.0
    for team, color in ((d.away, ac), (d.home, hc)):
        s = shots[shots["team"] == team].sort_values("seconds")
        t = np.concatenate([[0], s["seconds"].to_numpy(), [end]]) / 60
        c = np.concatenate([[0], s["xg"].cumsum().to_numpy(),
                            [s["xg"].sum()]])
        ax.step(t, c, where="post", color=color, lw=2)
        top = max(top, c[-1])
        g = s[s["is_goal"]]
        gc = s["xg"].cumsum()[s["is_goal"]]
        ax.scatter(g["seconds"] / 60, gc, s=70, facecolor=color,
                   edgecolor=cs.SURFACE, linewidth=2, zorder=4)
        for team_en, sec in d.empty_net_goals:
            if team_en == team:
                y = s.loc[s["seconds"] <= sec, "xg"].sum()
                ax.scatter([sec / 60], [y], s=70, marker="D", facecolor=color,
                           edgecolor=cs.SURFACE, linewidth=2, zorder=4)
                ax.text(sec / 60, y + 0.12, "EN", fontsize=7.5, ha="center",
                        color=cs.INK_2)
        ax.text(t[-1] + 0.6, c[-1], f"{team} {c[-1]:.2f}", color=cs.ink_for(color),
                fontsize=9.5, fontweight="bold", va="center")
    for m in (20, 40, 60):
        if m * 60 < end:
            ax.axvline(m, color=cs.GRID, lw=1, zorder=0)
    ax.set_xlim(0, end / 60)
    ax.set_ylim(0, top * 1.15 + 0.1)
    labels = [(10, "1st"), (30, "2nd"), (50, "3rd")]
    if end > 3600:
        labels.append(((60 + end / 60) / 2, "OT"))
    ax.set_xticks([x for x, _ in labels])
    ax.set_xticklabels([l for _, l in labels])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Cumulative xG")
    ax.set_title("Expected goals over the game", pad=22)
    ax.text(0, 1.02, "Dots = goals with a goalie in net", transform=ax.transAxes,
            fontsize=8.5, color=cs.MUTED, va="bottom")


def _model_vs_result(fig, rect, d: PostgameData, res: dict):
    """P(home win): before the game, from the shots, and what happened."""
    hc, _ = d.colors
    ax = fig.add_axes(rect)
    home_won = d.home_goals > d.away_goals
    rows = []
    if d.pregame_p_home is not None:
        rows.append(("Pregame model", d.pregame_p_home, d.pregame_interval))
    rows.append(("Deserved (shots)", res["p_home"], None))
    rows.append(("Result", 1.0 if home_won else 0.0, None))
    n = len(rows)
    for i, (label, v, ci) in enumerate(rows):
        y = n - 1 - i
        ax.plot([0, 1], [y, y], color=cs.GRID, lw=1, zorder=0)
        if ci is not None:
            ax.plot(ci, [y, y], color=cs.tint(hc, 0.55), lw=7,
                    solid_capstyle="round", zorder=2)
        marker = "D" if label == "Result" else "o"
        ax.scatter([v], [y], s=90, marker=marker, facecolor=hc,
                   edgecolor=cs.SURFACE, linewidth=2, zorder=3, clip_on=False)
        ax.text(-0.03, y, label, ha="right", va="center", fontsize=9.5,
                color=cs.INK)
        if label == "Result":
            txt = f"{d.home if home_won else d.away} won"
            if d.ended != "REG":
                txt += f" ({d.ended})"
        else:
            txt = cs.pct(v)
            if ci is not None:
                txt += f"  [{cs.pct(ci[0], 0)}–{cs.pct(ci[1], 0)}]"
        off = 0.03 if v < 0.75 else -0.03
        ax.text(v + off, y + 0.3, txt, ha="left" if off > 0 else "right",
                va="center", fontsize=8.5, color=cs.INK_2)
    ax.axvline(0.5, color=cs.MUTED, lw=1, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"P({d.home} win)")
    ax.set_title("Model, luck and result", pad=22, x=-0.42)
    ax.text(-0.42, 1.02 + 0.0, "Did the model miss, or did the bounces?",
            transform=ax.transAxes, fontsize=8.5, color=cs.MUTED, va="bottom")


def _top_shooters(fig, rect, d: PostgameData, shots: pd.DataFrame, k: int = 8):
    hc, ac = d.colors
    ax = fig.add_axes(rect)
    g = (shots.groupby(["shooter", "team"])
              .agg(xg=("xg", "sum"), goals=("is_goal", "sum"),
                   n=("xg", "size"))
              .reset_index().sort_values("xg", ascending=False).head(k)
              .iloc[::-1])
    y = np.arange(len(g))
    colors = [hc if t == d.home else ac for t in g["team"]]
    ax.barh(y, g["xg"], height=0.62, color=colors, linewidth=0)
    for yi, (_, r) in zip(y, g.iterrows()):
        extra = f"   {int(r.goals)} G" if r.goals else ""
        ax.text(r.xg + 0.02, yi, f"{r.xg:.2f}  ·  {int(r.n)} shots{extra}",
                va="center", fontsize=8.5, color=cs.INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}" for s in g["shooter"]])
    for lab, t in zip(ax.get_yticklabels(), g["team"]):
        lab.set_color(cs.INK)
    ax.set_xlim(0, g["xg"].max() * 1.75)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Top shooters by xG", pad=22)
    ax.text(0, 1.02, f"■ {d.home}", transform=ax.transAxes, fontsize=9,
            color=cs.ink_for(hc), fontweight="bold", va="bottom")
    ax.text(0.14, 1.02, f"■ {d.away}", transform=ax.transAxes, fontsize=9,
            color=cs.ink_for(ac), fontweight="bold", va="bottom")


def _goalies(fig, rect, d: PostgameData, shots: pd.DataFrame):
    hc, ac = d.colors
    ax = fig.add_axes(rect)
    ax.axis("off")
    g = (shots.groupby(["goalie", "team"])
              .agg(sa=("xg", "size"), xga=("xg", "sum"), ga=("is_goal", "sum"))
              .reset_index())
    # the goalie faced the *other* team's shots
    g["side"] = np.where(g["team"] == d.home, d.away, d.home)
    g = g.sort_values("side", key=lambda s: s != d.home)
    cols = [("Goalie", 0.0, "left"), ("TOI", 0.42, "right"),
            ("Shots", 0.56, "right"), ("xGA", 0.70, "right"),
            ("GA", 0.82, "right"), ("GSAx", 1.0, "right")]
    ax.text(0, 1.0, "Goalies", fontsize=11, fontweight="bold", va="bottom",
            transform=ax.transAxes)
    hy = 0.80
    for name, x, ha in cols:
        ax.text(x, hy, name, ha=ha, va="center", fontsize=8.5, color=cs.MUTED,
                transform=ax.transAxes)
    ax.plot([0, 1], [hy - 0.08, hy - 0.08], color=cs.GRID, lw=1,
            transform=ax.transAxes)
    for i, (_, r) in enumerate(g.iterrows()):
        y = hy - 0.24 - i * 0.22
        color = hc if r.side == d.home else ac
        gsax = r.xga - r.ga
        vals = [f"{r.goalie} ({r.side})", d.goalie_toi.get(r.goalie, "—"),
                f"{int(r.sa)}", f"{r.xga:.2f}", f"{int(r.ga)}", f"{gsax:+.2f}"]
        ax.add_patch(Circle((0.0, y), 0.0, transform=ax.transAxes))
        for (_, x, ha), v in zip(cols, vals):
            ax.text(x, y, v, ha=ha, va="center", fontsize=9.5,
                    color=cs.ink_for(color) if ha == "left" else cs.INK,
                    fontweight="bold" if (ha == "left" or v == vals[-1]) else "normal",
                    transform=ax.transAxes)
    ax.text(0, 0.0, "GSAx = xG against − goals against (empty-netters excluded)",
            fontsize=8, color=cs.MUTED, transform=ax.transAxes, va="bottom")


# --------------------------------------------------------------------------
# card
# --------------------------------------------------------------------------

def plot_postgame_card(d: PostgameData):
    """Draw the postgame card and return the figure."""
    if d.colors is None:
        d = replace(d, colors=cs.matchup_colors(d.home, d.away))
    hc, ac = d.colors
    shots = normalize_coords(d.shots, d.home)
    res = deserve_to_win(shots, d.home, d.away, n_sims=d.n_sims)

    fig = cs.new_figure(12, 13)
    cs.card_background(fig, [
        (0.025, 0.565, 0.975, 0.905),   # shot map + meter
        (0.025, 0.31, 0.975, 0.55),     # worm + model vs result
        (0.025, 0.055, 0.975, 0.295),   # shooters + goalies
    ])

    # header
    suffix = {"OT": " / OT", "SO": " / SO"}.get(d.ended, "")
    fig.text(0.04, 0.965, d.away, fontsize=24, fontweight="bold", color=cs.ink_for(ac),
             va="top")
    fig.text(0.115, 0.965, f"{d.away_goals}", fontsize=24, color=cs.INK,
             va="top")
    fig.text(0.15, 0.957, "at", fontsize=14, color=cs.MUTED, va="top")
    fig.text(0.18, 0.965, d.home, fontsize=24, fontweight="bold", color=cs.ink_for(hc),
             va="top")
    fig.text(0.255, 0.965, f"{d.home_goals}", fontsize=24, color=cs.INK,
             va="top")
    fig.text(0.96, 0.963, f"FINAL{suffix}", fontsize=10, color=cs.MUTED,
             fontweight="bold", ha="right", va="top")
    fig.text(0.96, 0.94, d.date, fontsize=13, color=cs.INK, ha="right",
             va="top")
    fig.text(0.04, 0.925,
             f"xG {d.away} {res['xg_away']:.2f} – {res['xg_home']:.2f} {d.home}",
             fontsize=10, color=cs.INK_2, va="top")

    _shot_map(fig, (0.04, 0.625, 0.62, 0.25), d, shots)
    _meter(fig, (0.69, 0.60, 0.27, 0.285), d, res)
    _worm(fig, (0.08, 0.35, 0.44, 0.15), d, shots)
    _model_vs_result(fig, (0.75, 0.355, 0.20, 0.145), d, res)
    _top_shooters(fig, (0.13, 0.08, 0.36, 0.17), d, shots)
    _goalies(fig, (0.56, 0.09, 0.39, 0.17), d, shots)

    foot = d.model_label or "Track B v1.1"
    if d.note:
        foot += f"   ·   {d.note}"
    fig.text(0.04, 0.025, foot, fontsize=8, color=cs.MUTED, va="bottom")
    fig.text(0.96, 0.025, "xG: leave-one-season-out LightGBM (shot_xg)",
             fontsize=8, color=cs.MUTED, ha="right", va="bottom")
    return fig