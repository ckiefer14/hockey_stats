"""
make_team_card.py
=================
Team cards (PNG) from the active fit, the database and the playoff odds history.

Home: src/pipelines/make_team_card.py

Usage (from the project root, after the weekly refit and playoff_odds):
    py -m src.pipelines.make_team_card BOS            # one team
    py -m src.pipelines.make_team_card BOS TOR FLA
    py -m src.pipelines.make_team_card --rankings     # only the rankings card (all teams on one card)
    py -m src.pipelines.make_team_card --all          # all teams + the rankings card (~1 min)

Writes output/teams/<date>/<TEAM>.png and rankings.png, date = the fit's ratings_as_of.
Positions: teams ordered by the posterior mean rating (every position used once); the range is
the 5th-95th percentile of the team's rank across posterior draws.

Per team:
- league rank overall / offense / defense with 90% ranges (rank of each posterior draw);
- weekly offense and defense paths over the fit window (2 seasons) with 90% bands, against
  the spread of all teams;
- what the ratings are made of now: shot quality for / against (xG part) and finishing for /
  against (phi), each as % better than average;
- goalies: share of the last 20 starts and Phase 3 rating (goals kept out vs expected);
- playoff odds through the season (playoff_odds_history_<season>.csv) and the latest projection.
"""

import argparse

import arviz as az
import matplotlib
import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from src.data.db import get_connection
from src.models.active_fit import active_fit_path, read_active, window_games
from src.models.stochastic import season_sim as ss
from src.models.stochastic.goalie import goalie_rating_as_of
from src.models.stochastic.hierarchical_tv import SEASON_GAP_DAYS, time_grid
from src.processing.goalie import goalie_game_xg
from src.processing.schedule_context import FRANCHISE

TEAM_DIR = OUTPUT_DIR / "teams"
PLAYOFF_DIR = OUTPUT_DIR / "playoffs"
Q = (0.05, 0.5, 0.95)


def season_label(d) -> str:
    d = pd.Timestamp(d)
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def load_fit():
    """Posterior arrays the card needs, pooled over chains."""
    path = active_fit_path()
    idata = az.from_netcdf(path).load()
    post = idata.posterior
    if "xg_mu_off_path" not in post:
        raise SystemExit("The active fit has no weekly rating paths (v1 fit). Team cards need a v2 (tv) fit.")

    def pool(name):
        x = post[name].values
        return x.reshape(-1, *x.shape[2:])

    fit = {k: pool(k) for k in ("xg_mu_off_path", "xg_mu_def_path", "phi_off", "phi_def", "mu_off", "mu_def")}
    fit["teams"] = str(post.attrs["teams"]).split(",")
    fit["as_of"] = str(post.attrs.get("ratings_as_of", ""))[:10]
    fit["file"] = path.name
    return fit


def step_dates(conn, fit) -> pd.DataFrame:
    """Date (step middle) and season of each weekly step, rebuilt from the fit's games."""
    gt, _ = window_games(conn)
    grid = time_grid(gt["game_date"], int(read_active().get("step_days", 7)))
    K = fit["xg_mu_off_path"].shape[1]
    if len(grid["mid"]) != K:
        raise SystemExit(f"The fit has {K} weekly steps but its games give {len(grid['mid'])} "
                         "(the database changed since the fit?). Refit, then retry.")
    mid = pd.to_datetime(grid["mid"])
    season, cur = [], season_label(mid[0])
    for m, summer in zip(mid, grid["summer"]):
        if summer:
            cur = season_label(m)
        season.append(cur)
    return pd.DataFrame({"date": mid, "season": season})


def paths(fit, steps: pd.DataFrame, ti: int) -> pd.DataFrame:
    """Weekly offense / defense as % better than average, team quantiles + league spread."""
    off = fit["xg_mu_off_path"] + fit["phi_off"][:, None, :]          # (n, K, T) goal scale
    dfn = fit["xg_mu_def_path"] + fit["phi_def"][:, None, :]
    out = steps.copy()
    for key, x in (("off", off), ("def", dfn)):
        e = np.expm1(x)
        out[f"{key}_lo"], out[f"{key}_mid"], out[f"{key}_hi"] = np.quantile(e[:, :, ti], Q, axis=0)
        means = e.mean(axis=0)                                         # (K, T)
        out[f"league_{key}_lo"], out[f"league_{key}_hi"] = means.min(axis=1), means.max(axis=1)
    # the current rating (stepped to the fit date) as a final point
    if fit["as_of"]:
        now_row = {"date": pd.Timestamp(fit["as_of"]), "season": out["season"].iloc[-1]}
        gap = (pd.Timestamp(fit["as_of"]) - out["date"].iloc[-1]).days
        if 0 < gap <= SEASON_GAP_DAYS:
            for key, now in (("off", fit["mu_off"]), ("def", fit["mu_def"])):
                e = np.expm1(now)
                now_row[f"{key}_lo"], now_row[f"{key}_mid"], now_row[f"{key}_hi"] = np.quantile(e[:, ti], Q)
                now_row[f"league_{key}_lo"] = e.mean(axis=0).min()
                now_row[f"league_{key}_hi"] = e.mean(axis=0).max()
            out = pd.concat([out, pd.DataFrame([now_row])], ignore_index=True)
    return out


def league_table(fit) -> pd.DataFrame:
    """One row per team: position 1..N overall / offense / defense (ordered by the posterior mean
    rating, so every position is used once), the 90% range of its rank across posterior draws,
    and the ratings as % better than an average team (mean, 90%)."""
    teams = fit["teams"]
    out = pd.DataFrame({"team": teams})
    for key, x in (("net", fit["mu_off"] + fit["mu_def"]), ("off", fit["mu_off"]), ("def", fit["mu_def"])):
        r = (-x).argsort(axis=1).argsort(axis=1) + 1                       # (n, T) rank per draw, 1 = best
        mean = x.mean(axis=0)
        out[f"{key}_rank"] = (-mean).argsort().argsort() + 1
        out[f"{key}_rank_lo"] = np.floor(np.quantile(r, 0.05, axis=0)).astype(int)
        out[f"{key}_rank_hi"] = np.ceil(np.quantile(r, 0.95, axis=0)).astype(int)
        e = np.expm1(x)
        out[f"{key}_pct"] = e.mean(axis=0)
        out[f"{key}_pct_lo"], out[f"{key}_pct_hi"] = np.quantile(e, [0.05, 0.95], axis=0)
        out[f"p_{key}_first"] = (r == 1).mean(axis=0)
    return out.sort_values("net_rank").reset_index(drop=True)


def ranks(table: pd.DataFrame, team: str) -> dict:
    r = table.set_index("team").loc[team]
    return {k: (int(r[f"{k}_rank"]), int(r[f"{k}_rank_lo"]), int(r[f"{k}_rank_hi"])) for k in ("net", "off", "def")}


def components(fit, ti: int) -> list:
    xg_off = fit["xg_mu_off_path"][:, -1, ti]
    xg_def = fit["xg_mu_def_path"][:, -1, ti]
    rows = []
    for label, x in (("Shot quality created (xG for)", xg_off), ("Finishing (goals per xG)", fit["phi_off"][:, ti]),
                     ("Shot quality allowed (xG against)", xg_def),
                     ("Opponents' finishing held down", fit["phi_def"][:, ti])):
        lo, m, hi = np.quantile(np.expm1(x), Q)
        rows.append((label, m, lo, hi))
    return rows


def goalies(conn, gg, team: str, day: str, n: int = 20) -> list:
    st = pd.read_sql("""SELECT g.player_id, g.name FROM goalie_game_stats g JOIN schedule s ON s.game_id = g.game_id
                        WHERE g.starter = 1 AND g.team_abbreviation = ? AND s.game_date < ?
                        ORDER BY s.game_date DESC LIMIT ?""", conn, params=(team, day, n))
    if st.empty:
        return []
    rat = goalie_rating_as_of(gg, day).set_index("goalie_id")
    out = []
    for pid, k in st["player_id"].value_counts().items():
        name = st.loc[st["player_id"] == pid, "name"].iloc[0]
        g = {"name": name, "share": k / len(st), "starts": k, "effect": None}
        if pid in rat.index:
            r = rat.loc[pid]
            g.update(effect=1 - float(r["theta_mean"]), lo=1 - float(r["theta_hi"]), hi=1 - float(r["theta_lo"]))
        out.append(g)
    return out


def playoff_history(team: str, day: str):
    files = sorted(PLAYOFF_DIR.glob("playoff_odds_history_*.csv"))
    if not files:
        return None
    h = pd.read_csv(files[-1])
    h = h[(h["team"] == team) & (h["date"].astype(str) <= day)]
    return h if len(h) else None


def latest_odds(day: str):
    """The latest playoff_odds_<d>.csv with d <= day, indexed by team (or None, None)."""
    runs = sorted(f for f in PLAYOFF_DIR.glob("playoff_odds_????-??-??.csv") if f.stem[-10:] <= day) \
        if PLAYOFF_DIR.exists() else []
    if not runs:
        return None, None
    return pd.read_csv(runs[-1]).set_index("team"), runs[-1].stem[-10:]


def make_cards(teams_wanted: list[str] | None = None, rankings: bool = False) -> list:
    from src.pipelines.make_cards import _model_label
    from src.viz.team_card import TeamCardData, plot_team_card
    import matplotlib.pyplot as plt

    fit = load_fit()
    conn = get_connection()
    steps = step_dates(conn, fit)
    day = fit["as_of"] or str(pd.Timestamp.today().date())
    info = pd.read_sql("SELECT abbreviation AS team, division, conference FROM teams", conn)
    info["team"] = info["team"].replace(FRANCHISE)
    info = info.drop_duplicates("team")
    season = str(pd.read_sql("SELECT MAX(season) AS s FROM games WHERE game_type = 2 AND home_score IS NOT NULL",
                             conn)["s"].iloc[0])
    stand = ss.current_standings(conn, season, info[info["team"].isin(fit["teams"])], FRANCHISE).set_index("team")
    gg = goalie_game_xg(conn)
    table = league_table(fit)
    wanted = [t.upper() for t in teams_wanted] if teams_wanted is not None else fit["teams"]
    unknown = [t for t in wanted if t not in fit["teams"]]
    if unknown:
        raise SystemExit(f"Not in the active fit: {unknown}. Teams: {', '.join(fit['teams'])}")
    out_dir = TEAM_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    if rankings:
        from src.viz.rankings_card import plot_rankings_card
        odds, odds_date = latest_odds(day)
        records = {t: stand.loc[t].to_dict() for t in stand.index}
        fig = plot_rankings_card(table, day, records, odds, odds_date, _model_label(fit["file"]))
        path = out_dir / "rankings.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"Saved {path}")
        print(table[["net_rank", "team", "net_rank_lo", "net_rank_hi", "off_rank", "def_rank"]]
              .rename(columns={"net_rank": "#", "net_rank_lo": "likely from", "net_rank_hi": "to",
                               "off_rank": "offense", "def_rank": "defense"}).to_string(index=False))
    for team in wanted:
        ti = fit["teams"].index(team)
        row = info[info["team"] == team]
        rec = stand.loc[team].to_dict() if team in stand.index and stand.loc[team, "gp"] > 0 else None
        d = TeamCardData(team=team, date=day,
                         division=row["division"].iloc[0] if len(row) else "",
                         conference=row["conference"].iloc[0] if len(row) else "",
                         record=rec, path=paths(fit, steps, ti), ranks=ranks(table, team), n_teams=len(fit["teams"]),
                         components=components(fit, ti), goalies=goalies(conn, gg, team, day),
                         playoff_hist=playoff_history(team, day), model_label=_model_label(fit["file"]))
        fig = plot_team_card(d)
        path = out_dir / f"{team}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"Saved {path}")
    conn.close()
    return saved


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description="Team cards from the active fit.")
    parser.add_argument("teams", nargs="*", help="team abbreviations (e.g. BOS TOR)")
    parser.add_argument("--all", action="store_true", help="every team in the fit, plus the rankings card")
    parser.add_argument("--rankings", action="store_true", help="the league rankings card (all teams on one card)")
    args = parser.parse_args()
    if not args.teams and not args.all and not args.rankings:
        parser.error("give team abbreviations, --rankings or --all")
    make_cards(None if args.all else args.teams, rankings=args.rankings or args.all)


if __name__ == "__main__":
    main()