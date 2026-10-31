"""Render both cards on EXAMPLE data (no database needed).

    py -m scripts.demo_cards

Pregame numbers follow the README's COL vs DAL example (55.8%, 90% 50.0-61.6%,
tied after regulation 23.8%); the score table and every shot in the postgame
card are synthetic. Writes to output/game_summaries/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.deserved import deserve_to_win
from src.viz.postgame_card import PostgameData, plot_postgame_card
from src.viz.pregame_card import PregameData, final_score_matrix, plot_pregame_card

OUT = Path("output/game_summaries")


def example_pregame() -> PregameData:
    rng = np.random.default_rng(42)
    # per-draw P(home win): mean 0.558, 90% about 0.50-0.616
    p_draws = rng.beta(111.6, 88.4, size=1000)

    # synthetic final scores: Poisson regulation + OT/SO winner +1
    n = 100_000
    h = rng.poisson(2.95, n)
    a = rng.poisson(2.70, n)
    tie = h == a
    home_wins_tie = rng.random(n) < 0.52
    h = h + (tie & home_wins_tie)
    a = a + (tie & ~home_wins_tie)

    return PregameData(
        home="COL", away="DAL", date="2026-04-10",
        p_home_draws=p_draws,
        outcome={"home_reg": 0.4355, "home_ot": 0.083, "home_so": 0.0395,
                 "away_so": 0.0395, "away_ot": 0.076, "away_reg": 0.3265},
        p_tied_reg=0.238,
        final_scores=final_score_matrix(h, a),
        home_goalie="M. Blackwood", away_goalie="J. Oettinger",
        home_goalie_confirmed=True, away_goalie_confirmed=False,
        model_label="Track B v1.1 · trackbxg_w365_20260417",
        note="EXAMPLE DATA",
    )


def _team_shots(rng, team, goalie, n, players, attack_right=True, boost=1.0):
    """Synthetic unblocked shots with a distance-based xG."""
    rows = []
    for _ in range(n):
        dist = np.clip(rng.gamma(2.6, 13), 4, 70)
        angle = rng.normal(0, 32)
        gx = 89 - dist * np.cos(np.radians(angle))
        gy = dist * np.sin(np.radians(angle))
        gy = float(np.clip(gy, -40, 40))
        xg = 1 / (1 + np.exp(-(-0.9 - 0.075 * dist - 0.012 * abs(angle))))
        xg = float(np.clip(xg * boost * rng.lognormal(0, 0.35), 0.005, 0.6))
        period = int(rng.choice([1, 2, 3], p=[0.32, 0.34, 0.34]))
        sec = (period - 1) * 1200 + float(rng.uniform(0, 1200))
        # raw coordinates switch ends by period, as in the NHL feed
        flip = (period == 2) != (not attack_right)
        x, y = (-gx, -gy) if flip else (gx, gy)
        rows.append(dict(team=team, period=period, seconds=sec, x=x, y=y, xg=xg,
                         shooter=str(rng.choice(players, p=_weights(len(players)))),
                         goalie=goalie))
    return rows


def _weights(k):
    w = np.linspace(2.2, 0.6, k)
    return w / w.sum()


def example_postgame() -> PostgameData:
    rng = np.random.default_rng(7)
    col = ["MacKinnon", "Necas", "Nichushkin", "Makar", "Drury", "Kadri",
           "Lehkonen", "Toews", "Burns", "Colton"]
    dal = ["Robertson", "Johnston", "Rantanen", "Hintz", "Heiskanen",
           "Duchene", "Benn", "Harley", "Seguin", "Marchment"]
    shots = _team_shots(rng, "COL", "J. Oettinger", 48, col, True, 1.15)
    shots += _team_shots(rng, "DAL", "M. Blackwood", 35, dal, True, 0.9)
    df = pd.DataFrame(shots).sort_values("seconds").reset_index(drop=True)

    # choose goals so the example matches a 3-2 COL regulation win
    df["is_goal"] = False
    for team, k in (("COL", 3), ("DAL", 2)):
        idx = df.index[df.team == team]
        p = df.loc[idx, "xg"].to_numpy()
        pick = rng.choice(idx, size=k, replace=False, p=p / p.sum())
        df.loc[pick, "is_goal"] = True

    return PostgameData(
        home="COL", away="DAL", date="2026-04-10",
        home_goals=3, away_goals=2, ended="REG",
        shots=df,
        goalie_toi={"M. Blackwood": "60:00", "J. Oettinger": "58:41"},
        empty_net_goals=[],
        pregame_p_home=0.558, pregame_interval=(0.500, 0.616),
        model_label="Track B v1.1 · trackbxg_w365_20260417",
        note="EXAMPLE DATA · synthetic shots",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = example_pregame()
    fig = plot_pregame_card(pre)
    fig.savefig(OUT / "example_DAL_at_COL_pregame.png", dpi=200)

    post = example_postgame()
    fig = plot_postgame_card(post)
    fig.savefig(OUT / "example_DAL_at_COL_postgame.png", dpi=200)

    d = deserve_to_win(post.shots, post.home, post.away)
    print(f"deserved P({post.home} win) = {d['p_home']:.3f}")


if __name__ == "__main__":
    main()