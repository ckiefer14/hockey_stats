# src/viz/shot_heatmap.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import gaussian_kde

SHOT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "shot_heat",
    [
        (0.0, (1, 1, 1, 0)),
        (0.3, (1, 0.6, 0, 0.7)),
        (0.7, (1, 0.1, 0, 0.9)),
        (1.0, (0.5, 0, 0, 1)),
    ],
)


def draw_half_rink(ax):
    """Draw the offensive zone half of an NHL rink."""
    ax.set_xlim(23, 101)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.axis("off")

    lw = 1.5
    rink_color = "#e8e8e8"
    line_color = "#cccccc"
    blue_color = "#0056a3"
    red_color = "#c8102e"

    # Bottom board
    ax.plot([25, 72], [-42.5, -42.5], color=line_color, lw=lw)
    # Top board
    ax.plot([25, 72], [42.5, 42.5], color=line_color, lw=lw)

    # End boards with rounded corners
    corner_radius = 28
    corner_br = patches.Arc((72, -42.5 + corner_radius), corner_radius * 2, corner_radius * 2,
                            angle=0, theta1=270, theta2=360, color=line_color, lw=lw)
    ax.add_patch(corner_br)
    corner_tr = patches.Arc((72, 42.5 - corner_radius), corner_radius * 2, corner_radius * 2,
                            angle=0, theta1=0, theta2=90, color=line_color, lw=lw)
    ax.add_patch(corner_tr)
    # Right end board
    ax.plot([100, 100], [-42.5 + corner_radius, 42.5 - corner_radius], color=line_color, lw=lw)

    # Blue line
    ax.plot([25, 25], [-42.5, 42.5], color=blue_color, lw=2.5)

    # Goal line
    ax.plot([89, 89], [-37, 37], color=red_color, lw=1.5, alpha=0.7)

    # Goal crease
    crease = patches.Arc((89, 0), 12, 12, angle=0, theta1=90, theta2=270,
                          color=red_color, lw=1.5, zorder=2)
    ax.add_patch(crease)

    # Goal net
    net = patches.Rectangle((89, -3), 2.5, 6, linewidth=1.5,
                             edgecolor="#333333", facecolor="none", zorder=2)
    ax.add_patch(net)

    # Faceoff circles (offensive zone)
    for y_pos in [-22, 22]:
        circle = patches.Circle((69, y_pos), 15, fill=False,
                                 edgecolor=red_color, lw=1, alpha=0.6, zorder=2)
        ax.add_patch(circle)
        dot = patches.Circle((69, y_pos), 1, fill=True,
                               facecolor=red_color, edgecolor=red_color, zorder=2)
        ax.add_patch(dot)

    # Fill the rink background
    rink_bg = patches.FancyBboxPatch((23, -43), 78, 86, boxstyle="round,pad=2",
                                      facecolor=rink_color, edgecolor="none", zorder=-10)
    ax.add_patch(rink_bg)


def _draw_kde(ax, x_oz, y_oz, bw=4.0, threshold=0.08):
    """Fit and render an isotropic KDE heatmap onto ax."""
    kde = gaussian_kde(np.vstack([x_oz, y_oz]), bw_method=bw / x_oz.std())

    xi = np.linspace(25, 89, 300)
    yi = np.linspace(-42.5, 42.5, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    cutoff = Zi.max() * threshold
    levels = np.linspace(cutoff, Zi.max(), 25)
    ax.contourf(Xi, Yi, Zi, levels=levels, cmap=SHOT_CMAP,
                alpha=0.75, zorder=1, extend="neither")


def normalize_coords(shots):
    """Flip all shots so they point toward the net at x=89."""
    x_coords = []
    y_coords = []
    for shot in shots:
        x = shot.x_coord
        y = shot.y_coord
        if x < 0:
            x = -x
            y = -y
        x_coords.append(x)
        y_coords.append(y)
    return np.array(x_coords), np.array(y_coords)


def plot_game_shot_heatmap(home_shots, away_shots, home_abbrev, away_abbrev, game_date, save_path=None):
    """Plot a shot heat map for a single game showing both teams."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), facecolor="#f5f5f5")
    fig.suptitle(f"{away_abbrev}  @  {home_abbrev}  —  {game_date}",
                 fontsize=18, fontweight="bold", color="#222222", y=0.97)
    fig.text(0.5, 0.92, "Offensive Zone Shot Heat Map", ha="center",
             fontsize=12, color="#666666")

    panels = [
        (axs[0], away_shots, away_abbrev),
        (axs[1], home_shots, home_abbrev),
    ]

    for ax, shots, abbrev in panels:
        draw_half_rink(ax)

        if shots:
            on_net = [s for s in shots if s.event_type in ("shot-on-goal", "goal")]
            x, y = normalize_coords(on_net)

            ozone_mask = x > 25
            x_oz = x[ozone_mask]
            y_oz = y[ozone_mask]

            if len(x_oz) > 2:
                _draw_kde(ax, x_oz, y_oz)

            # Shot dots — drawn above heatmap (zorder=3+) so rink lines stay visible
            missed_list = [s for s in shots if s.event_type == "missed-shot"]
            on_goal_list = [s for s in shots if s.event_type == "shot-on-goal"]
            goals_list = [s for s in shots if s.event_type == "goal"]

            if missed_list:
                m_x, m_y = normalize_coords(missed_list)
                ax.scatter(m_x, m_y, color="#777777", marker="x",
                           s=30, zorder=4, alpha=0.7, linewidth=1.2)

            if on_goal_list:
                og_x, og_y = normalize_coords(on_goal_list)
                ax.scatter(og_x, og_y, color="#dddddd", edgecolor="#444444",
                           s=28, zorder=4, alpha=0.85, linewidth=0.8)

            if goals_list:
                g_x, g_y = normalize_coords(goals_list)
                ax.scatter(g_x, g_y, color="#ff2222", edgecolor="white",
                           s=70, zorder=5, alpha=0.95, linewidth=1.2, marker="*")

        if shots:
            goal_count = len([s for s in shots if s.event_type == "goal"])
            sog_count = len([s for s in shots if s.event_type == "shot-on-goal"])
            missed_count = len([s for s in shots if s.event_type == "missed-shot"])
        else:
            goal_count = sog_count = missed_count = 0

        ax.set_title(f"{abbrev}   {sog_count} SOG  •  {missed_count} missed  •  {goal_count} goals",
                     fontsize=14, fontweight="bold", color="#333333", pad=10)

        ax.set_xlim(23, 101)
        ax.set_ylim(-45, 45)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_season_shot_heatmap(shots, team_abbrev, season, save_path=None):
    """Plot an aggregate shot heat map for a full season."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="#f5f5f5")

    draw_half_rink(ax)

    if shots:
        on_net = [s for s in shots if s.event_type in ("shot-on-goal", "goal")]
        x, y = normalize_coords(on_net)

        ozone_mask = x > 25
        x_oz = x[ozone_mask]
        y_oz = y[ozone_mask]

        if len(x_oz) > 2:
            _draw_kde(ax, x_oz, y_oz)

    shot_count = len(shots)
    goal_count = len([s for s in shots if s.event_type == "goal"])
    fig.suptitle(f"{team_abbrev} Season Shot Heat Map — {season}",
                 fontsize=16, fontweight="bold", color="#222222")
    ax.set_title(f"{shot_count} shots  •  {goal_count} goals",
                 fontsize=12, color="#666666", pad=10)

    ax.set_xlim(23, 101)
    ax.set_ylim(-45, 45)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
