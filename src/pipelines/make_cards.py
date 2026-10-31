"""Build pregame and postgame cards (PNG).

Pregame cards are made automatically by ``predict_day`` (one per game it logs,
from the same numbers it logs). To redraw one from the forward-test log:

    py -m src.pipelines.make_cards pregame  2026020180

Postgame cards need the game loaded (``load_all``) and its shots scored
(``score_xg --season ...``):

    py -m src.pipelines.make_cards postgame --pending           # every logged game now final, no card yet
    py -m src.pipelines.make_cards postgame --date 2026-10-25   # every loaded game on a date
    py -m src.pipelines.make_cards postgame 2026020180 2026020181
    py -m src.pipelines.make_cards both     2026020180

PNGs go to output/game_summaries/<date>/<away>_at_<home>_<pregame|postgame>.png.
``--redo`` redraws cards that already exist.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from config.settings import DB_PATH, OUTPUT_DIR

OUT = OUTPUT_DIR / "game_summaries"
FORWARD_DIR = OUTPUT_DIR / "forward"
POSTERIORS = OUTPUT_DIR / "posteriors"
MODELS = OUTPUT_DIR / "models"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def card_path(date: str, home: str, away: str, kind: str) -> Path:
    folder = OUT / str(date)[:10]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{away}_at_{home}_{kind}.png"


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def forward_log() -> pd.DataFrame:
    files = sorted(FORWARD_DIR.glob("predictions_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def forward_row(game_id: int, log: pd.DataFrame | None = None) -> pd.Series | None:
    """Prediction to show for this game: the last one made on or before the game
    date (the same rule score_forward uses), else None."""
    log = forward_log() if log is None else log
    if log.empty:
        return None
    rows = log[log["game_id"] == game_id]
    if rows.empty:
        return None
    made = pd.to_datetime(rows["predicted_at"]).dt.normalize()
    ok = rows[made <= pd.to_datetime(rows["game_date"])]
    rows = ok if len(ok) else rows
    return rows.sort_values("predicted_at").iloc[-1]


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _score_matrix(table, p_home_win: float) -> np.ndarray:
    """Final-score probability matrix as [home goals, away goals].

    ``final_from_sims`` returns the table; its orientation is checked against
    the exact P(home win) so a transposed table can't slip through.
    """
    m = np.asarray(table.to_numpy() if hasattr(table, "to_numpy") else table, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"final_from_sims()['table'] has shape {m.shape}, expected a 2-D score table")
    m = m / m.sum()
    i, j = np.indices(m.shape)
    as_is, transposed = m[i > j].sum(), m[i < j].sum()
    if abs(transposed - p_home_win) < abs(as_is - p_home_win):
        m = m.T
    return m


def _model_label(fit_file: str) -> str:
    stem = Path(str(fit_file)).stem
    version = "Track B v1.1" if stem.startswith("trackbxg") else "Track B"
    return f"{version} · {stem}"


# --------------------------------------------------------------------------
# pregame
# --------------------------------------------------------------------------

def pregame_card(row, res: dict, sims: pd.DataFrame, conn) -> Path:
    """Pregame card from numbers already computed (what predict_day calls).

    row:  the logged prediction (dict or Series: home, away, game_date, game_id,
          home/away_goalie, home/away_goalie_source, fit_file)
    res:  game_model.predict_matchup result (per-draw probabilities)
    sims: score-state simulation of the same matchup
    """
    from src.analysis.aggregation import final_from_sims
    from src.viz.postgame_card import player_names
    from src.viz.pregame_card import PregameData, plot_pregame_card

    home, away, date = row["home"], row["away"], str(row["game_date"])[:10]
    pr = res["probs"]
    p_draws = np.asarray(pr["home_win"], float)
    names = player_names(conn, int(row["game_id"]))

    def goalie(col):
        v = row[col]
        return names.get(int(v), f"#{int(v)}") if pd.notna(v) else ""

    data = PregameData(
        home=home, away=away, date=date,
        p_home_draws=p_draws,
        outcome={
            "home_reg": float(np.mean(pr["reg_home"])),
            "home_ot": float(np.mean(pr["ot_home"])),
            "home_so": float(np.mean(pr["so_home"])),
            "away_so": float(np.mean(pr["so_away"])),
            "away_ot": float(np.mean(pr["ot_away"])),
            "away_reg": float(np.mean(pr["reg_away"])),
        },
        p_tied_reg=float(np.mean(pr["tie_reg"])),
        final_scores=_score_matrix(final_from_sims(sims)["table"], float(p_draws.mean())),
        home_goalie=goalie("home_goalie"),
        away_goalie=goalie("away_goalie"),
        home_goalie_confirmed=row["home_goalie_source"] != "default",
        away_goalie_confirmed=row["away_goalie_source"] != "default",
        model_label=_model_label(row["fit_file"]),
    )
    return _save(plot_pregame_card(data), card_path(date, home, away, "pregame"))


def make_pregame(game_id: int, n_sims: int = 100_000) -> Path:
    """Redraw a pregame card from the forward-test log (re-runs the model with the
    logged fit file, starters and goalie offsets)."""
    from src.models.game_model import load_track_b, predict_matchup
    from src.models.stochastic.montecarlo import simulate_matchup_score_state

    row = forward_row(game_id)
    if row is None:
        raise SystemExit(f"Game {game_id} is not in the forward-test log. Run predict_day for its date first.")
    home, away = row["home"], row["away"]
    g_home, g_away = float(row["gamma_home"]), float(row["gamma_away"])

    draws, teams = load_track_b(POSTERIORS / row["fit_file"])
    late = _load_json(MODELS / "late_game.json")
    ss = _load_json(MODELS / "score_state.json")
    if late.get("fit_file") != row["fit_file"]:
        print(f"Note: late_game.json was measured on {late.get('fit_file')}, the prediction used "
              f"{row['fit_file']}; numbers may differ slightly from the log.")

    res = predict_matchup(draws, teams, late, home, away, gamma_home=g_home, gamma_away=g_away)
    p = float(np.mean(res["probs"]["home_win"]))
    if abs(p - float(row["p_home"])) > 0.003:
        print(f"Note: P(home win) {p:.4f} differs from the log ({float(row['p_home']):.4f}).")
    sims = simulate_matchup_score_state(draws, teams, late, ss, home, away,
                                        gamma_home=g_home, gamma_away=g_away, n_sims=n_sims, seed=42)
    with sqlite3.connect(DB_PATH) as conn:
        return pregame_card(row, res, sims, conn)


# --------------------------------------------------------------------------
# postgame
# --------------------------------------------------------------------------

def _cols(conn, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def make_postgame(game_id: int, log: pd.DataFrame | None = None) -> Path:
    from src.viz.postgame_card import (PostgameData, load_empty_net_goals, load_game_shots,
                                       player_names, plot_postgame_card)
    with sqlite3.connect(DB_PATH) as conn:
        state = ", game_state" if "game_state" in _cols(conn, "games") else ""
        g = pd.read_sql_query(f"""
            SELECT id AS game_id, game_date, home_team_abbrev AS home, away_team_abbrev AS away,
                   home_score, away_score, period_type{state}
            FROM games WHERE id = ?""", conn, params=(game_id,))
        if g.empty:
            raise SystemExit(f"Game {game_id} is not in the games table. Run load_all first.")
        g = g.iloc[0]
        if state and g["game_state"] not in ("OFF", "FINAL"):
            raise SystemExit(f"Game {game_id} is not final (state {g['game_state']}).")
        shots = load_game_shots(conn, game_id)
        if shots.empty:
            raise SystemExit(f"No shot_xg rows for game {game_id}. Run score_xg for its season first.")
        names = player_names(conn, game_id)
        toi = (pd.read_sql_query("SELECT player_id, toi FROM goalie_game_stats WHERE game_id = ?",
                                 conn, params=(game_id,))
               if "toi" in _cols(conn, "goalie_game_stats") else pd.DataFrame(columns=["player_id", "toi"]))
        empty_net = load_empty_net_goals(conn, game_id)

    row = forward_row(game_id, log)
    p_pre = interval = None
    model_label = ""
    if row is not None:
        p_pre = float(row["p_home"])
        interval = (float(row["p_home_lo"]), float(row["p_home_hi"]))
        model_label = _model_label(row["fit_file"])

    date = str(g["game_date"])[:10]
    data = PostgameData(
        home=g["home"], away=g["away"], date=date,
        home_goals=int(g["home_score"]), away_goals=int(g["away_score"]),
        ended=str(g["period_type"] or "REG").upper(),
        shots=shots,
        goalie_toi={names.get(int(i), f"#{int(i)}"): t for i, t in zip(toi["player_id"], toi["toi"])},
        empty_net_goals=empty_net,
        pregame_p_home=p_pre, pregame_interval=interval,
        model_label=model_label,
    )
    return _save(plot_postgame_card(data), card_path(date, g["home"], g["away"], "postgame"))


def ready_for_postgame(conn, game_ids=None, date: str | None = None) -> pd.DataFrame:
    """Regular-season games that are loaded and have xG (so a postgame card can be drawn)."""
    q = """
        SELECT g.id AS game_id, g.game_date, g.home_team_abbrev AS home, g.away_team_abbrev AS away
        FROM games g
        WHERE g.game_type = 2 AND EXISTS (SELECT 1 FROM shot_xg x WHERE x.game_id = g.id)
    """
    df = pd.read_sql_query(q, conn)
    if date is not None:
        df = df[df["game_date"].astype(str).str[:10] == date]
    if game_ids is not None:
        df = df[df["game_id"].isin(set(game_ids))]
    return df


def postgame_batch(game_ids=None, date: str | None = None, redo: bool = False,
                   report_missing: bool = True) -> list[Path]:
    """Postgame cards for many games; skips ones already drawn unless redo."""
    log = forward_log()
    with sqlite3.connect(DB_PATH) as conn:
        todo = ready_for_postgame(conn, game_ids, date)
    if game_ids is not None and report_missing:
        missing = set(game_ids) - set(todo["game_id"])
        if missing:
            print(f"Not ready (not loaded yet, or no xG; run load_all / score_xg): "
                  f"{', '.join(map(str, sorted(missing)))}")
    made = []
    for r in todo.itertuples():
        path = card_path(r.game_date, r.home, r.away, "postgame")
        if path.exists() and not redo:
            continue
        try:
            made.append(make_postgame(int(r.game_id), log))
            print(f"Saved {made[-1]}")
        except SystemExit as e:
            print(f"  {r.away} @ {r.home} ({r.game_id}): {e}")
    return made


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="Pregame / postgame cards (PNG).")
    ap.add_argument("kind", choices=["pregame", "postgame", "both"])
    ap.add_argument("game_ids", type=int, nargs="*")
    ap.add_argument("--pending", action="store_true",
                    help="postgame: every forward-logged game that is now final and has no card yet")
    ap.add_argument("--date", default=None, help="postgame: every loaded game on this date (YYYY-MM-DD)")
    ap.add_argument("--redo", action="store_true", help="redraw cards that already exist")
    a = ap.parse_args()

    if a.kind == "postgame" and (a.pending or a.date):
        ids = None
        if a.pending:
            log = forward_log()
            if log.empty:
                raise SystemExit("No forward-test log yet (run predict_day).")
            ids = list(log["game_id"].unique())
        made = postgame_batch(ids, a.date, a.redo, report_missing=not a.pending)
        print(f"{len(made)} postgame cards made")
        if a.pending:
            with sqlite3.connect(DB_PATH) as conn:
                ready = set(ready_for_postgame(conn, ids)["game_id"])
            waiting = log[~log["game_id"].isin(ready)].drop_duplicates("game_id")
            have = len(ready) - len(made)
            print(f"Forward log: {log['game_id'].nunique()} games. {have} already had a card. "
                  f"{len(waiting)} not played or not loaded yet"
                  + (f" (next: {waiting['game_date'].min()}; run load_all + score_xg after they're played)"
                     if len(waiting) else "") + ".")
        return
    if not a.game_ids:
        raise SystemExit("Give game ids, or for postgame use --pending or --date.")
    for gid in a.game_ids:
        if a.kind in ("pregame", "both"):
            print(f"Saved {make_pregame(gid)}")
        if a.kind in ("postgame", "both"):
            print(f"Saved {make_postgame(gid)}")


if __name__ == "__main__":
    main()