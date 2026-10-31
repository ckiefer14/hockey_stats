"""Shared look for the pregame and postgame cards.

Colors are the validated default chart palette (light mode). Home and away get
categorical slots 1 and 2 (blue / orange), which stay distinguishable for
colour-blind readers. Pass ``colors=(home_hex, away_hex)`` to a card to use team
colours instead (check that the pair is still distinguishable).

If ``theme.py`` already defines fonts or colours you prefer, import them here so
every card follows one theme.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# --- chrome ------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f4f3ef"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- series --------------------------------------------------------------------
HOME = "#2a78d6"   # slot 1, blue
AWAY = "#eb6834"   # slot 2, orange

# Sequential blue ramp (light -> dark) for the score heat map
BLUE_RAMP = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

FONT_FAMILY = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


def apply_style() -> None:
    """Set matplotlib defaults for the cards (call once before drawing)."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.facecolor": SURFACE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def tint(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour toward white. amount 0 = unchanged, 1 = white."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def text_on(hex_or_rgb) -> str:
    """White or black label text, whichever reads better on a fill."""
    r, g, b = to_rgb(hex_or_rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.6 else "#ffffff"


def panel(ax, title: str | None = None, subtitle: str | None = None) -> None:
    """White panel background and a left-aligned title (+ muted subtitle)."""
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc="left", pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
                color=MUTED, va="bottom", ha="left")


def card_background(fig, rects) -> None:
    """Draw white rounded panels behind groups of axes.

    ``rects`` are (x0, y0, x1, y1) in figure fraction.
    """
    from matplotlib.patches import FancyBboxPatch
    for x0, y0, x1, y1 in rects:
        fig.patches.append(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure, facecolor=SURFACE,
            edgecolor=(0.043, 0.043, 0.043, 0.10), linewidth=0.8, zorder=-10))


def fair_american_odds(p: float) -> str:
    """Odds with no bookmaker margin. p >= 0.5 -> negative, else positive."""
    if p <= 0 or p >= 1:
        return "n/a"
    if p >= 0.5:
        return f"{-100 * p / (1 - p):+.0f}"
    return f"+{100 * (1 - p) / p:.0f}"


def pct(p: float, digits: int = 1) -> str:
    return f"{100 * p:.{digits}f}%"


def new_figure(width: float, height: float):
    apply_style()
    return plt.figure(figsize=(width, height))


# --------------------------------------------------------------------------
# team colours
# --------------------------------------------------------------------------
# Team colours come from config/teams.py. Two teams' colours can clash (both
# navy, or red vs green for a colour-blind reader), so ``matchup_colors``
# tries primary/secondary combinations in order and keeps the first pair that
# is easy to tell apart. Distances are OKLab x 100, the same scale as the
# chart-palette rules: >= 15 for normal vision, >= 8 under red-green colour
# blindness (simulated with the Machado et al. 2009 matrices).

import numpy as _np

MIN_NORMAL = 15.0
MIN_CVD = 8.0

_MACHADO = {
    "protan": _np.array([[0.152286, 1.052583, -0.204868],
                         [0.114503, 0.786281, 0.099216],
                         [-0.003882, -0.048116, 1.051998]]),
    "deutan": _np.array([[0.367322, 0.860646, -0.227968],
                         [0.280085, 0.672501, 0.047413],
                         [-0.011820, 0.042940, 0.968881]]),
}


def _lin(c):
    c = _np.asarray(c, float)
    return _np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab_lin(rgb_lin):
    m1 = _np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = _np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = _np.cbrt(m1 @ rgb_lin)
    return m2 @ lms


def oklab(hex_color: str):
    return _oklab_lin(_lin(to_rgb(hex_color)))


def color_distance(a: str, b: str) -> dict:
    """OKLab distance x 100: normal vision and worst red-green simulation."""
    la, lb = _lin(to_rgb(a)), _lin(to_rgb(b))
    normal = 100 * _np.linalg.norm(_oklab_lin(la) - _oklab_lin(lb))
    cvd = min(100 * _np.linalg.norm(
        _oklab_lin(_np.clip(m @ la, 0, 1)) - _oklab_lin(_np.clip(m @ lb, 0, 1)))
        for m in _MACHADO.values())
    return {"normal": float(normal), "cvd": float(cvd)}


def _usable(hex_color: str) -> bool:
    """Not near-black (clashes with ink, needle, outlines) and not near-white
    (vanishes on the card)."""
    L = oklab(hex_color)[0]
    return 0.20 <= L <= 0.90


ROAD_NEUTRAL = "#6e7781"   # slate gray: the away team's "road whites" when no team colours work


def _team_colors(team: str) -> list[str]:
    """A team's own colours that can be drawn on the card, in preference order:
    primary, secondary, then any extra colours config/teams.py lists
    ("tertiary", "alt"). White and near-white are dropped (they vanish on the
    card); near-black is kept but tried last (it clashes with ink and outlines)."""
    try:
        from config.teams import get_team_colors
        tc = get_team_colors(team) or {}
    except ImportError:
        return []
    raw = [tc.get(k) for k in ("primary", "secondary", "tertiary", "alt")]
    raw = [c for c in raw if isinstance(c, str) and c.startswith("#")]
    good = [c for c in raw if _usable(c)]
    dark = [c for c in raw if not _usable(c) and oklab(c)[0] < 0.2]
    return good + dark


def _distinct(a: str, b: str, cvd_safe: bool) -> bool:
    d = color_distance(a, b)
    return d["normal"] >= MIN_NORMAL and (d["cvd"] >= MIN_CVD or not cvd_safe)


def matchup_colors(home: str, away: str, cvd_safe: bool = True) -> tuple[str, str]:
    """Fill colours for (home, away), like jerseys.

    1. Both teams' real colours: primaries first; when a change is needed the away
       team switches before the home team (the road team wears the alternate),
       then both switch.
    2. If no pair of real team colours is distinguishable (also for colour-blind
       readers), the away team goes to a neutral slate gray (ROAD_NEUTRAL) and the
       home team keeps its colour. A colour that belongs to neither team is never used.
    3. Teams missing from config/teams.py fall back to the default palette.
    """
    hc, ac = _team_colors(home), _team_colors(away)
    if not hc and not ac:
        return HOME, AWAY
    if not hc:
        hc = [HOME if _distinct(HOME, ac[0], cvd_safe) else AWAY]
    if not ac:
        ac = [AWAY if _distinct(hc[0], AWAY, cvd_safe) else HOME]

    # 1. real team colours: fewest changes first; away changes before home
    pairs = sorted(((i, j) for i in range(len(hc)) for j in range(len(ac))),
                   key=lambda ij: (ij[0] + ij[1], ij[0]))
    for i, j in pairs:
        if _distinct(hc[i], ac[j], cvd_safe):
            return hc[i], ac[j]

    # 2. road neutral for the away team (home keeps its first workable colour)
    for c in hc:
        if _distinct(c, ROAD_NEUTRAL, cvd_safe):
            return c, ROAD_NEUTRAL
    for c in ac:                                     # home colour is itself gray-ish
        if _distinct(ROAD_NEUTRAL, c, cvd_safe):
            return ROAD_NEUTRAL, c
    return HOME, AWAY


def _contrast(a, b) -> float:
    def lum(c):
        r, g, bb = _lin(to_rgb(c))
        return 0.2126 * r + 0.7152 * g + 0.0722 * bb
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def ink_for(hex_color: str, min_contrast: float = 3.0) -> str:
    """The team colour, darkened just enough to read as text on the card
    (gold or light blue would otherwise wash out)."""
    from matplotlib.colors import to_hex
    r, g, b = to_rgb(hex_color)
    for k in _np.linspace(1.0, 0.3, 36):
        c = (r * k, g * k, b * k)
        if _contrast(c, SURFACE) >= min_contrast:
            return to_hex(c)
    return INK


def team_ramp(hex_color: str, name: str = "team"):
    """Sequential ramp for one team: near-white tint -> the team colour."""
    return LinearSegmentedColormap.from_list(
        name, [tint(hex_color, 0.92), tint(hex_color, 0.55), hex_color])