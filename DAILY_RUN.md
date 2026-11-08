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

- **Closing lines (every game day):** `py -m src.pipelines.log_closing` waits and takes one snapshot about 10 minutes before each group of start times (3-5 a night, 1 credit each), so every game gets a line from just before puck drop. Start it any time before the first game and leave the window open, or let Task Scheduler start it (see "Closing lines" below). `--dry-run` shows the plan without using credits. Scoring uses each game's **last line taken before it started**.

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
py -m src.pipelines.make_team_card --all  # team cards + rankings card (~1 min): output/teams/<date>/<TEAM>.png, rankings.png
py -m src.pipelines.fit_xg_layer --final  # shooter finishing (and goalie / rink) effects for the player cards (a few min)
py -m src.pipelines.make_player_card --all   # every rated skater this season (~3-6 min); or names / ids / --team BOS
                                              #   output/players/<date>/<TEAM>/<Name>_<id>.png
py -m src.pipelines.make_goalie_card --all   # every goalie who started this season (~1 min); or names / ids / --team BOS
                                              #   output/goalies/<date>/<TEAM>/<Name>_<id>.png

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

## Closing lines (one-time setup)

`log_closing` needs the computer awake at the snapshot times (it sleeps between them; if the computer was asleep and a game
already started, it says "missed" and moves on). To start it automatically every day at 11:00 am, run once in PowerShell
(one line; change the path if your project lives elsewhere):

```powershell
schtasks /Create /TN "NHL closing lines" /SC DAILY /ST 11:00 /TR "cmd /c cd /d C:\Users\ckief\personal_projects\hockey_stats_code && py -m src.pipelines.log_closing >> output\forward\log_closing.log 2>&1"
```

- Check it: Task Scheduler → Task Scheduler Library → "NHL closing lines" → Run; then read `output\forward\log_closing.log`.
- Remove it: `schtasks /Delete /TN "NHL closing lines" /F`.
- Keep the laptop plugged in with sleep off in the evenings (Settings → System → Power) or the late snapshots are missed.
- Credits: morning `log_odds` + about 4 closing snapshots a day is about 150 a month (free plan: 500). It stops itself
  below 20 credits left.

`score_forward` reports how long before puck drop the lines were taken and, once some games have them, the model vs
market comparison on closing lines only (30 minutes or less before the start). Closing lines are the sharpest version of
the market: the honest benchmark.

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