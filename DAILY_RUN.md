# Daily Run (2026-27 season)

Run from the project root, **every morning before the first game**: finish yesterday, then set up today.

## Every morning

```powershell
# 1. Yesterday: load results, score xG, draw postgame cards
py .\scripts\load_all.py
py -m src.pipelines.score_xg --season 20262027
py -m src.pipelines.make_cards postgame --pending

# 2. Today: predictions + pregame cards, then the market's lines for the same games
py -m src.pipelines.predict_day
py -m src.pipelines.log_odds
```

- Optional: run `py -m src.pipelines.log_odds` again shortly before the first game. Scoring uses each game's **last line taken before it started**, so a later snapshot is closer to the closing line (1 credit per run; the free plan has 500 a month).

- `score_xg` must run before the postgame cards (they need each shot's xG).
- `predict_day` goes last so it uses the newest goalie history.

## Once a week (same morning, before `predict_day`)

```powershell
# 1. Yesterday (same as daily)
py .\scripts\load_all.py
py -m src.pipelines.score_xg --season 20262027
py -m src.pipelines.make_cards postgame --pending

# Weekly: grade the forward test, then refit
py -m src.pipelines.score_forward
py -m src.pipelines.fit_hierarchical --draws 2000   # v2.0 + back-to-backs (default): ~12 min
py -m src.processing.late_game
py -m src.processing.score_state
py -m src.pipelines.build_pce            # optional: refresh the Phase 8 indices

# 2. Today
py -m src.pipelines.predict_day
py -m src.pipelines.log_odds
```

The fit is Track B v2.0 + back-to-back effects (time-varying ratings, 730-day window, weekly steps, rest on by default). `--draws 2000` keeps R-hat and ESS clean; `--no-rest` fits plain v2.0, `--model xg` fits v1.1. `predict_day` looks up each team's rest itself and tags games with `[TEAM back-to-back]`. Each fit file is about 50–70 MB: delete old ones in `output/posteriors/` now and then (the active one is named in `active_fit.json`).

Refit in the **morning**, not at night: a fit made after a day's games includes them, so that day can no longer be predicted.

## Odds logging (one-time setup)

1. Get a free API key at https://the-odds-api.com.
2. Create a file named `.env` in the project root containing one line: `ODDS_API_KEY=your_key` (it is gitignored, so the key never reaches GitHub).
3. `py -m src.pipelines.log_odds` prints each game's market P(home win) (median over books, margin removed) next to the model's, and appends to `output/forward/odds_<season>.csv`. `--markets h2h,totals,spreads` also logs totals and puck lines (3 credits).

`score_forward` then adds a **model vs market** section: log loss for both on the same games, the paired difference with a 90% range, and how each did on games where they disagree by 5+ points.

## Re-predicting a game (`--replace`)

```powershell
py -m src.pipelines.predict_day --goalie COL=Blackwood --goalie DAL=Oettinger --replace
py -m src.pipelines.predict_day --replace          # re-predict every game today (e.g. after a refit)
```

- `--goalie` takes a **last name** (or part of one, any case) or a player id. The name is matched against the team's recent goalies first; if two match, it lists them with their ids. It prints the starter it used.
- See a team's goalies and ids: `py -m src.pipelines.predict_day --list-goalies COL DAL`

- `--replace` re-predicts games already in the log and redraws their pregame cards.
- Nothing is deleted: a new row is added, and scoring and the postgame card use the **last prediction made on or before game day**.
- It will **not** predict a game that's already played and loaded, or a date the active fit was trained on (the leak guards).

## Other options

| Command | What it does |
|---|---|
| `predict_day --date 2026-10-26` | Predict a specific date |
| `predict_day --days 3` | Today and the next 2 days |
| `predict_day --no-cards` | Log only, no pregame cards |
| `make_cards postgame --date 2026-10-24` | Postgame cards for every loaded game on a date (predicted or not) |
| `make_cards postgame 2026020180` | One game by id (`--redo` redraws an existing card) |
| `make_cards pregame 2026020180` | Redraw a pregame card from the log |

## If you miss a day

- **Postgame cards catch up on their own:** `--pending` draws every finished game that's missing a card.
- **Predictions can't be made up:** a missed day is a gap in the forward test. Once games are played, they can't be predicted.

## Where things go

| Output | Location |
|---|---|
| Predictions log | `output/forward/predictions_20262027.csv` |
| Scored predictions | `output/forward/scored_20262027.csv` |
| Odds log | `output/forward/odds_20262027.csv` |
| Cards | `output/game_summaries/<date>/<away>_at_<home>_<pregame\|postgame>.png` |
| Active fit | `output/posteriors/active_fit.json` |