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

**Lineups.** `predict_day` prints each team's assumed lineup and what it did to that team's expected xG, e.g.
`EDM lineup (assumed, xG shift -4.7%): out: Leon Draisaitl (Out); fills in: ...`. It starts from the last dressed lineup,
drops anyone not on today's NHL roster or listed **Out** by ESPN, keeps **day-to-day** players (flagged), brings back
healthy regulars, and fills gaps from the roster. Read those lines: if you know better (a late scratch, a call-up),
re-predict with overrides:

```powershell
py -m src.pipelines.predict_day --replace --out EDM=Draisaitl --in EDM=Savoie
py -m src.pipelines.predict_day --replace --dtd-out          # treat every day-to-day player as out
py -m src.pipelines.predict_day --replace --show-lineups     # print every team's full 18 by name
```

To check the assumptions, compare the notes (and `--show-lineups`) with ESPN's NHL injuries page, Daily Faceoff's
line combinations (projected lineups from morning skates) and team beat reporters.

If the NHL has already posted the dressed lineup (the morning roster is about 23 per team; the dressed lineup is 18
skaters), it is used as is and the line says `posted`. If a feed is down the line says so and the rest still works.

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
py -m src.pipelines.build_stints                    # new games' stints (seconds)
py -m src.pipelines.build_lineup_asof               # newest player ratings + lineup context (a few min)
py -m src.pipelines.fit_hierarchical --draws 2000   # v2.0 + back-to-backs + lineup (default): ~12-15 min
py -m src.processing.late_game
py -m src.processing.score_state
py -m src.pipelines.build_pce            # optional: refresh the Phase 8 indices
py -m src.pipelines.playoff_odds         # playoff odds from the new fit (~1-2 min) + charts:
                                         #   output/playoffs/playoff_board_<date>.png, playoff_trend_<season>.png

# 2. Today
py -m src.pipelines.predict_day
py -m src.pipelines.log_odds
```

The fit is Track B v2.0 + back-to-back effects + lineup (time-varying ratings, 730-day window, weekly steps; rest and lineup on by default). `--draws 2000` keeps R-hat and ESS clean; `--no-lineup` fits v2.0 + rest (the model before 2026-11-08), `--no-rest --no-lineup` plain v2.0, `--model xg` v1.1. The fit **stops** if the stints or player ratings are behind the loaded games, and says which command to run. `predict_day` looks up each team's rest itself and tags games with `[TEAM back-to-back]`. Each fit file is about 50–70 MB: delete old ones in `output/posteriors/` now and then (the active one is named in `active_fit.json`).

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