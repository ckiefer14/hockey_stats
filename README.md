# Hockey UQ Project — Architecture & Build Plan

A probabilistic hockey prediction pipeline with full uncertainty quantification, structured as a port and extension of the AEM 8290 land-surface column UQ project. The goal is calibrated predictive distributions over game outcomes, player point totals, and (eventually) season-long quantities — not point estimates.

This document is the project's reference architecture and roadmap. Come back to it when you forget what `analysis/` is supposed to contain or why `game_model.py` lives at the top level of `models/`.

---

## Project Goals

The end-state pipeline produces, for any matchup:

- $P(\text{home win})$ with credible intervals
- Predictive distributions over final score, total goals, goal differential
- Predictive distributions over individual player point totals
- Sensitivity decomposition: which inputs drive outcome variance

All of these come from the same underlying machinery — a hierarchical Bayesian model fit to league-wide data, propagated through a game model and Monte Carlo simulator, validated with proper scoring rules.

**First shipping target:** single-game win probability. Smallest pipeline that exercises every layer.

---

## Status

*Last updated: 2026-11-08*

| Phase | Status | Result |
|---|---|---|
| 1. Data foundation | **Done** | 5 seasons (2021-22 to 2025-26), 6,560 games, full play-by-play, rebound/rush context validated |
| 2. xG model | **Done** | LightGBM, test AUC 0.769, team-season xG correlates 0.97 with MoneyPuck |
| 3. Team strength & goalie aggregates | **Done** | Rolling 25-game xG team strength picks winners 59% of the time; Bayesian goalie rating (k = 800, season-adjusted) |
| 4. Track A toy MCMC | **Done** | AM sampler verified: 90% intervals cover the truth 90 and 96 times out of 100; matches the exact posterior; COL vs DAL fit saved |
| 5. Track B hierarchical model | **Done** | Full-league PyMC model (2024-25 + 2025-26, 2,624 games): team offense/defense ratings with uncertainty, home ice 0.071, tie inflation ρ = 0.51; synthetic recovery, Track A match and posterior predictive checks pass; remaining dispersion gap traced to empty-net removal (section 11) |
| 6. Game model + Monte Carlo | **Done** | `predict_game.py`: P(win) with uncertainty from an exact per-draw model (reproduces home win, REG/OT/SO shares of 2024-26); score outputs from a 10-second score-state simulator (pulled goalies, late-game caution) that matches real ties, empty-net share and margins |
| 7. Validation | **Done** | Walk-forward backtest, 2,624 held-out games (2024-25 + 2025-26). Track B v1.1 (ratings from xG + goals, 1-year window) log loss 0.6773, calibration slope 0.99: clearly beats Track B v1, Elo and home rate; **statistically tied with rolling xG** (0.6754). Recency weighting tested and not adopted. Market comparison waiting on closing-line data |
| 8. PCE / Sobol | **Done** | What makes P(home win) differ between games: team ratings 95%, home-ice uncertainty 3%, starting goalies 2%, almost no interactions. Ties are driven by tie-inflation uncertainty (55%). PCE and Saltelli agree within 0.0003. Analysis B (one matchup's uncertainty) moved to v2 |
| v2.0 Time-varying ratings | **Done, live model** | Team ratings and league levels drift week to week. Walk-forward log loss **0.6749** vs v1.1 0.6773 (**−0.0025 [−0.0040, −0.0009], clear**), ties/edges rolling xG (0.6754), calibration slope 0.98; blending with rolling xG no longer clearly helps. Weekly = 2-day steps |
| v2.0b Follow-ups | **Done** (backtest parts) | Early season fine, favorites calibrated, wrong guessed starter costs ~0.0001; **away back-to-backs badly under-called** (home won 62.0%, predicted 53.4%) → v2.0d |
| v2.0d Back-to-backs | **Done, live model** | Back-to-back effects on shot quality: log loss **0.6732**, **−0.0016 [−0.0026, −0.0006] vs v2.0 (clear)**, −0.0041 vs v1.1, slope 1.02; blend gap gone (0.0004). Home-ice drift tested, adds nothing. ARI → UTA merged |
| v2.0e / v2.0f Rest on finishing, travel | **Tested, not adopted** | Rest on goals per xG −0.0004 [−0.0009, +0.0001] vs live (within luck); travel, time zones, 3-in-4 +0.0000. The away back-to-back gap (−0.056) is not about travel, backups or a goalie on zero rest (v2.0g check); remaining value at most ~0.001 log loss → chapter closed |
| v2.1 Hierarchical xG layer | **Built, not adopted for games** | Shot level: shooter finishing helps every held-out season (−0.0002 to −0.0003 per shot), goalie skill once 4+ seasons of history; rink and season × type dropped. Game level vs live: layer goalie −0.0003 [−0.0016, +0.0011], + shooter-adjusted xG −0.0004 [−0.0019, +0.0012], within luck, calibration 0.94–0.96 |
| v2.2 Lineup model | **Done, live model since 2026-11-08** | Shift data fixed and backfilled, stints, ridge and Bayesian RAPM validated. Game level vs live (3,936 games): −0.0009 [−0.0020, +0.0002], within luck (about 94% likely better), leans better in 2 of 3 seasons. **Adopted** (a judgment call: 6% risk accepted) with live lineups from ESPN injuries + the NHL game-day roster; recheck with 2026-27 |
| v2 (remaining) | In progress | Forward test and odds logging running (now v2.0 + rest + lineup); calibration checked (fine). Playoff odds done (v2.4). Next: **v2 visuals** (lineups/rest/ranges on cards, team rating trends, player cards, playoff chart), then v2.0c analysis B and v2.3 extras; v2.2 recheck in spring 2027 |

**Check current state:** `py .\scripts\check_state.py` lists which files and tables exist and whether `shot_context` matches the current derivation code.

---

## The Six-Step Arc

The full pipeline, end-to-end:

1. **Build component models** — xG (gradient boosting), team strength aggregates, goalie ratings. Each produces estimates with uncertainty.
2. **Combine via Poisson/NB game model** — maps team-level inputs to $\lambda_{\text{home}}, \lambda_{\text{away}}$.
3. **Monte Carlo simulation** — propagates uncertainty from every layer into a predictive distribution over outcomes.
4. **Aggregate sims** into the distributions you actually care about: win probabilities, score distributions, player point totals.
5. **Validate** with proper scoring rules (log-loss, Brier, CRPS) and calibration plots on held-out games.
6. **Fit PCE surrogate** over the simulator's input space — enables fast sensitivity analysis and analytical Sobol decomposition.

---

## Mapping to the AEM 8290 Soil-Atmosphere Project

The hockey UQ pipeline is structurally the same as your AEM 8290 land-surface column project, with the forward model swapped and a hierarchical layer added. Most of the conceptual machinery already exists in working code.

| Hockey project component | Soil project equivalent |
|---|---|
| Component models (xG, team strength, goalie) | Forward model: soil heat + Richards' + SEB |
| Game model ($\lambda_{\text{home}}, \lambda_{\text{away}}$) | Forward model mapping $\theta \to$ QoIs |
| Monte Carlo over game outcomes | Forward propagation, screening, Sobol Monte Carlo |
| Aggregation to predictive distributions | Posterior moments, predictive distributions |
| Held-out validation, calibration plots | LOO-CV $Q^2$, held-out RMSE, trace plots, ACF |
| PCE surrogate for Sobol indices | PCE surrogate for MCMC efficiency |

**What transfers directly:**
- The Adaptive Metropolis sampler (random-walk MH, burn-in / adapt / sample, $(2.38)^2/d$ scaling, $\xi \in [-1,1]^d$ standardized space)
- Additive Gaussian likelihood structure (mechanics — independent QoI noise summed in log-space)
- Objective prior strategy (uniform-bounded for location, log-uniform for scale spanning decades)
- PCE-as-surrogate trick for expensive forward models
- Validation discipline ($Q^2$, trace plots, acceptance rates, ACF, posterior recovery vs truth)

**What changes:**
- **Dimensionality.** Soil: 5 calibration parameters. Hockey hierarchical model: ~70–100+ parameters. Random-walk MH does not scale; NUTS (HMC variant via PyMC/NumPyro) required.
- **Hierarchy.** Soil model is flat. Hockey model has population-level hyperpriors with partial pooling across teams and positions.
- **Likelihood form.** Gaussian becomes Poisson (goal counts) or Bernoulli (win/loss).
- **Forward model speed.** Soil: ~1 s/eval, surrogate essential. Hockey game model: milliseconds/eval, surrogate optional. PCE earns its keep in hockey for the **Sobol analysis**, not for inference speed.
- **Data volume.** Soil: synthetic 24–72h time series of 2 QoIs. Hockey: ~1200 games/season × multiple observables per game.

---

## Architecture

### Folder Structure

```
hockey-stats/
│
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
│
├── config/
│   ├── settings.py             # paths, current season, DB location
│   └── teams.py                # team metadata, colors, abbreviations
│
├── data/                       # SQLite database lives here
│   ├── hockey_stats.db
│   └── external/               # files downloaded by hand (e.g. MoneyPuck teams.csv)
│
├── src/
│   │
│   ├── data/                   # data acquisition & storage
│   │   ├── __init__.py
│   │   ├── nhl_api.py          # NHL API client
│   │   ├── odds.py             # DONE: The Odds API client, team-name map, no-vig probabilities, consensus line
│   │   ├── db.py               # SQLite operations
│   │   ├── scraper.py          # Natural Stat Trick / Evolving-Hockey scrapers
│   │   └── cache.py            # caching layer
│   │
│   ├── processing/             # raw data → analysis-ready
│   │   ├── __init__.py
│   │   ├── shot_context.py     # DONE: rebound / rush flags → shot_context table
│   │   ├── xg_processing.py    # DONE: xG feature table, one row per shot
│   │   ├── shots.py            # shot data processing
│   │   ├── corsi.py            # Corsi/Fenwick calculations
│   │   ├── game_score.py       # game score breakdown
│   │   ├── team_strength.py    # DONE: team_game_xg, rolling_strength, strength_as_of
│   │   ├── stints.py           # v2: shifts → stints (constant on-ice sets)
│   │   ├── rapm.py             # RAPM / WAR (future)
│   │   ├── goalie.py           # DONE: goalie_game_xg (per-goalie, per-game xGA / GA)
│   │   ├── game_table.py       # DONE: one row per game for Track B (goals, starters, goalie γ)
│   │   ├── late_game.py        # DONE: OT factor, shootout, empty-net table (late_game.json)
│   │   └── score_state.py      # DONE: score effects, pull timing, 6-on-5 / empty-net rates (score_state.json)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic: Game, Player, Shot, ScheduleGame
│   │   ├── game_model.py       # DONE: exact outcome probabilities per posterior draw
│   │   ├── active_fit.py       # DONE: which saved fit the live pipeline uses (active_fit.json)
│   │   │
│   │   ├── deterministic/
│   │   │   ├── __init__.py
│   │   │   ├── xg.py           # DONE: ExpectedGoalsModel (LightGBM)
│   │   │   └── rapm.py         # ridge regression RAPM (future)
│   │   │
│   │   └── stochastic/
│   │       ├── __init__.py
│   │       ├── am_sampler.py   # Adaptive Metropolis (ported from soil)
│   │       ├── goalie.py       # DONE: goalie_rating_as_of (Gamma-Poisson, season-adjusted)
│   │       ├── toy_mcmc.py     # Track A: hockey-flavored AM MCMC
│   │       ├── hierarchical.py # DONE: Track B v1 PyMC model, sample_scores, simulate_league
│   │       ├── hierarchical_xg.py # DONE: Track B v1.1 (ratings from xG + goals), simulate_league_xg
│   │       ├── hierarchical_tv.py # DONE: Track B v2.0 (time-varying ratings, live model), simulate_league_tv
│   │       ├── xg_layer.py        # v2.1: hierarchical layer on top of LightGBM xG (Laplace fit)
│   │       ├── lineup_rapm.py  # v2: Bayesian RAPM player effects
│   │       ├── montecarlo.py   # DONE: tie-inflated and score-state game simulators
│   │       └── pce.py          # DONE: PCE (ported from soil), Sobol' from coefficients, Saltelli, Ishigami test
│   │
│   ├── analysis/               # NEW: reduce sim output to distributions
│   │   ├── __init__.py
│   │   ├── aggregation.py      # DONE: outcome tables, likely scores, totals, puck lines, fair odds
│   │   └── sensitivity.py      # DONE: Phase 8 inputs/ranges, vectorized exact game model, group indices
│   │
│   ├── validation/             # scoring, calibration, backtesting
│   │   ├── __init__.py
│   │   ├── scoring.py          # DONE: log loss, Brier, calibration slope/table, paired bootstrap, RPS for totals
│   │   ├── baselines.py        # DONE: home rate, tuned Elo, rolling-xG logistic
│   │   └── backtest.py         # DONE: walk-forward harness (weekly refits, resumable)
│   │
│   ├── pipelines/              # NEW: high-level orchestration
│   │   ├── __init__.py
│   │   ├── train_xg.py         # DONE: build features, train, evaluate, save
│   │   ├── score_xg.py         # DONE: score every shot into shot_xg (leave-one-season-out)
│   │   ├── fit_hierarchical.py # DONE: fit Track B (default v1.1, last 365 days), save NetCDF + ratings, set active fit
│   │   ├── fit_toy.py          # fit Track A toy MCMC for comparison
│   │   ├── fit_xg_layer.py     # v2: fit hierarchical xG layer, save NetCDF
│   │   ├── fit_lineup.py       # v2: build stints, fit RAPM, save NetCDF
│   │   ├── predict_game.py     # DONE: py -m src.pipelines.predict_game HOME AWAY [--date ...]
│   │   ├── backtest_season.py  # DONE: py -m src.pipelines.backtest_season --season S [--model xg] [--window D] [--half-life D]
│   │   ├── run_baselines.py    # DONE: py -m src.pipelines.run_baselines --seasons S1 S2
│   │   ├── predict_day.py      # DONE: forward test, log every game on a date before it's played
│   │   ├── log_odds.py         # DONE: log sportsbook moneylines next to the forward test
│   │   ├── score_forward.py    # DONE: score the forward-test log against played games
│   │   ├── make_cards.py       # DONE: pregame cards (called by predict_day) and postgame cards (--pending, --date, ids)
│   │   └── build_pce.py        # DONE: py -m src.pipelines.build_pce [--degree 4]
│   │
│   ├── viz/                    # visualization (existing)
│   │   ├── __init__.py
│   │   ├── theme.py
│   │   ├── rink_plot.py        # DONE: draw_rink, plot_game_events, plot_game
│   │   ├── card_style.py       # DONE: card theme, team-colour pairing (colour-blind checked), odds/pct helpers
│   │   ├── pregame_card.py     # DONE: P(win) posterior strip, fair odds, how the game ends, score grid, totals
│   │   ├── postgame_card.py    # DONE: shot heat map, deserve-to-win meter, xG worm, model vs result, shooters, goalies
│   │   ├── shot_heatmap.py
│   │   ├── game_summary.py     # (future)
│   │   ├── player_card.py      # (future)
│   │   └── components/
│   │
│   └── export/
│       ├── __init__.py
│       └── save.py
│
├── notebooks/                  # exploration and checks
│   ├── data_sanity.ipynb       # raw data checks (sections 1-13)
│   ├── xg_features_check.ipynb # xG feature table checks
│   ├── xg_model_dev.ipynb      # baseline, LightGBM, evaluation, MoneyPuck comparison
│   ├── phase3_check.ipynb      # team strength + goalie rating checks and settings
│   ├── phase4_toy.ipynb        # Track A function checks + recovery/coverage tests
│   ├── phase5_check.ipynb      # Track B checks (recovery, Track A match, predictive)
│   ├── phase6_check.ipynb      # game model checks (exact vs simulated, history, calibration preview)
│   ├── phase7_check.ipynb      # backtest scoring: models vs baselines, paired tests, calibration, blends, market
│   └── phase8_check.ipynb      # PCE/Sobol checks: Ishigami, degree, accuracy, PCE vs Saltelli, groups, robustness
│
├── output/                     # generated images, posterior samples
│   ├── game_summaries/
│   ├── player_cards/
│   ├── posteriors/             # NetCDF posterior samples
│   └── models/                 # pickled trained models (xG); keep out of git
│
├── tests/
│   ├── test_corsi.py
│   ├── test_game_score.py
│   └── fixtures/
│
└── scripts/                    # CLI entry points
    ├── load_all.py                   # teams, rosters, schedule, games, shots, events, shifts; seasons from SEASONS in config/settings.py; 1 s between requests, retries on 429, skips complete schedules
    ├── backfill_game_events.py       # one-time: full play-by-play for loaded games
    ├── rebuild_shot_context.py       # re-derive shot_context after a code change
    ├── check_state.py                # project state + shot_context sync check
    ├── fix_goal_shooters.py          # one-time repair (goal scorer IDs)
    ├── check_block_attribution.py    # diagnostic: who owns a blocked-shot event
    ├── check_block_zone_frame.py     # diagnostic: which frame blocked-shot zones use
    ├── nst_spotcheck.py              # compare rush/rebound counts with Natural Stat Trick
    └── rush_event_audit.py           # list rush-flagged shots for one game
```

### Why Certain Things Live Where They Do

**`models/game_model.py` at the top level of `models/`** — neither purely deterministic nor purely stochastic. It's the bridge that takes deterministic components and lifts them into a stochastic combiner. Forcing it into one subfolder would be a category error. It belongs alongside `schemas.py` as a top-level model concern.

**`analysis/` separate from `models/`** — `models/` is about computation that produces parameters or predictions. `analysis/` is about *reducing* simulation output to the distributions you care about. Different responsibility: aggregation, not modeling.

**`validation/` separate from everything** — scoring rules and calibration plots are infrastructure that applies to any model. Keeping them separate means they can be reused if you swap component models.

**`pipelines/` for orchestration** — high-level entry points that chain components together. Each pipeline file should read like a recipe: load this, fit that, save the output. No business logic, just glue.

**Two tracks under `stochastic/`** — `toy_mcmc.py` (Track A, your soil-pipeline port) and `hierarchical.py` (Track B, PyMC) coexist. Track A is the reference implementation; Track B is the production model. They share input contracts so the rest of the pipeline doesn't care which produced the samples.

---

## Data Contract

### Tables

| Table | Rows | One row per | Notes |
|---|---|---|---|
| `shot_events` | 773,259 | shot attempt (incl. blocked, shootout) | Raw; filter at query time (below) |
| `game_events` | 2,060,996 | play-by-play event of any type | Raw; ordered by `sort_order`, never `event_id` |
| `shot_context` | 566,545 | unblocked, non-shootout shot | Derived from `game_events` by `shot_context.py` |
| `shot_xg` | 562,374 | scored shot (unblocked, periods 1–4, goalie in net) | Derived by `score_xg.py`; each season scored by a model that did not train on it |
| `shifts` | 4,578,542 | player shift | |
| `games`, `schedule` | 6,560 | game | 1,312 per season × 5 |

`shot_context` joins to `shot_events` on `(game_id, event_id)`. After any change to
`src/processing/shot_context.py`, run `scripts/rebuild_shot_context.py`;
`check_state.py` flags a stale table.

### Filtering Rules

`shot_events` is stored **raw and complete** — no filtering is applied at the
database level. Shootout attempts (period 5) and blocked shots are all present.
This is deliberate: the raw table stays a faithful record, and every consumer
applies the filters it needs at query time.

**Two rules, applied at query time:**

1. `period <= 4` — excludes shootout (period 5) events; keeps OT (period 4).
2. `event_type IN ('shot-on-goal', 'goal', 'missed-shot')` — excludes
   blocked shots (degenerate coordinates) and shootout attempts.

Together these yield **566,545 clean unblocked non-shootout shots**. The xG
feature table then drops empty-net shots (opposing goalie digit = 0 in
`situation_code`), leaving **562,374 training shots** (Phase 2).

**Why this matters per consumer:**

- **xG model (Phase 2):** apply both rules, then drop empty-net shots. Blocked
  shots lack a true shot location, shootout attempts are mislabeled as goals,
  and empty-net shots score regardless of geometry; any of them would poison
  training.
- **Team model (Phase 5) and game model (Phase 6):** fit **regulation goals**
  (`period <= 3`, empty-net goals excluded). Overtime is sudden death, which a
  per-game scoring rate can't describe, so Phase 6 simulates OT and the
  shootout separately after a tied regulation. (Phase 4's Track A used
  `period <= 4` goals; the matched check uses the same.)

**Background:** the data-sanity notebook found that period-5 shootout attempts
are stored as `event_type` `'goal'` / `'shot-on-goal'` / `'missed-shot'`
(1,065 shootout "goals", ~2.6% of all goal events). All score-reconciliation
residuals after filtering are off-by-exactly-1 (the shootout-winner credit
added to the official score with no corresponding real goal event), zero
off-by-more-than-1 — fully characterized and expected, not a bug.

### Known Data Quirks (verified against this database)

- **Goal scorers:** the API stores the scorer under `scoringPlayerId`, not
  `shootingPlayerId`. `parse_shot_events` reads both. 41,536 goals were repaired
  once with `scripts/fix_goal_shooters.py`.
- **Blocked-shot owner:** `eventOwnerTeamId` is the **shooting** team (90/10,
  `check_block_attribution.py`).
- **Blocked-shot zone:** `zoneCode` is written from the **defending** team's side
  even though the owner is the shooter (99.9% of team-periods,
  `check_block_zone_frame.py`). `shot_context.py` flips it before the rush rule.
- **Zone codes** are relative to the event owner (O / N / D). Compare two events
  only after translating both into the shooter's frame.
- **`situation_code`** is 4 digits: away goalie, away skaters, home skaters, home
  goalie (`1551` = 5v5). Decode from the shooter's point of view.
- **Shot-type labels shifted in 2025-26** (many wrist → snap, deflected → tip-in).
  The xG features group them as `wrist-snap` and `redirect`.
- **Missing shot type** occurs almost only on goals; filling it with "unknown"
  leaks the outcome. The xG features fill it with "wrist".
- **Coordinates:** center ice is (0, 0), goal lines at x = ±89, blue lines at
  x = ±25. Teams switch ends each period; normalize per game, period and team.

### Rebound and Rush Definitions (`shot_context.py`)

NST conventions, with the no-stoppage rule enforced (faceoffs, whistles,
penalties and goals break the chain):

- **Rebound:** unblocked shot within 3 s of a prior attempt by the same team.
  9.4% of shots; converts at 12.4% vs 6.6% for other shots.
- **Rush:** unblocked shot within 4 s of any event in the neutral zone or the
  shooter's defensive zone. 2.7% of shots; weak as a yes/no flag (7.4% vs
  7.1%), but strong when split by the previous event: takeaway 10.1%,
  blocked shot 11.1%, giveaway 9.0%, hit 4.0%.
- NST spot-check: 1 of 3 rush flags matched in game 2025020993. Both misses had
  an opponent's shot as the precursor, at physically impossible 1–2 s gaps
  (about 17% of rush flags, 0.5% of training data).

---

## Parallel Build: Track A vs Track B

The honest version of the model is the PyMC hierarchical NUTS sampler. But porting your AEM 8290 AM MCMC structure to a tiny hockey problem first buys real value: a reference implementation you trust, and an independent cross-check on the PyMC model.

### Track A — Toy AM MCMC (Soil Pipeline Port)

**Scope:** one team pair at a time. Parameter vector:

$$\theta = (\mu_{\text{off,A}}, \mu_{\text{def,A}}, \mu_{\text{off,B}}, \mu_{\text{def,B}})$$

**Likelihood:** Poisson on observed goal counts in head-to-head games:

$$p(y^{\text{obs}} \mid \theta) = \prod_{g} \text{Poisson}(G_{\text{home},g} \mid \lambda_{\text{home}}(\theta)) \cdot \text{Poisson}(G_{\text{away},g} \mid \lambda_{\text{away}}(\theta))$$

**Priors:** bounded uniform in standardized $\xi \in [-1, 1]^4$ space, same as soil.

**Identifiability — only two of the four parameters are learnable.** With $\lambda_{A} = \exp(\alpha_0 + \mu_{\text{off},A} - \mu_{\text{def},B})$ and $\lambda_{B} = \exp(\alpha_0 + \mu_{\text{off},B} - \mu_{\text{def},A})$, adding the same constant to $\mu_{\text{off},A}$ and $\mu_{\text{def},B}$ (or to $\mu_{\text{off},B}$ and $\mu_{\text{def},A}$) leaves the likelihood unchanged. The data pin down only the two differences

$$d_A = \mu_{\text{off},A} - \mu_{\text{def},B}, \qquad d_B = \mu_{\text{off},B} - \mu_{\text{def},A}$$

and the rest of the posterior is the bounded prior. Sample $(d_A, d_B)$ directly ($d = 2$).

**Fixed league terms.** Both come from the whole league, computed at fit time by `league_baseline()`, not estimated from the pair:

$$\log \lambda_A = \alpha_0 + \eta \cdot \text{home}_A + d_A, \qquad \log \lambda_B = \alpha_0 + \eta \cdot \text{home}_B + d_B$$

- $\alpha_0 = \log(\text{league away goals per game})$ = 1.0291 (2.799 goals)
- $\eta = \log(\text{home rate} / \text{away rate})$ = 0.0715 (3.006 vs 2.799 goals)

One pair's ~15 games would give $\eta$ an uncertainty of about ±0.20, three times the effect; the league gives about ±0.010. Fixing it uses the precise estimate, keeps the model at two parameters, and avoids $\eta$ trading off against $d_A, d_B$. The ignored ±0.010 is tiny next to the ~±0.15 width of each $d$. (A tight prior $\eta \sim \mathcal{N}(0.0715, 0.010^2)$ would give the same answer.)

**Goals target:** `gf_xg` / `ga_xg` — periods 1–4, **empty-net goals excluded** (5.9% of goals). Empty-net goals are scored by the team already leading, so they never change the winner; they reflect game state rather than team strength; and excluding them matches the xG model. Track B uses the same target.

**Expect wide posteriors.** Two teams meet three or four times a season, so even five seasons give only about 15 games per pair. Track A is a correctness check, not a source of useful team ratings.

**Sampler:** Adaptive Metropolis, three phases:
1. **Burn-in** ($n_b = 5000$): isotropic proposal $\Sigma_p = \sigma_0^2 I$ with $\sigma_0 = 0.1$
2. **Adapt:** estimate posterior covariance from burn-in, set $\Sigma_p = s \cdot \hat\Sigma$ with $s = (2.38)^2 / d$
3. **Sample:** main run with fixed adapted covariance

**Reuses:** your soil sampler module, standardization machinery, diagnostics (trace plots, ACF, acceptance rate, $R$-hat).

**Purpose:** proof of life, reference posteriors, bug-catching for Track B.

### Track B — Hierarchical PyMC Model

**Scope:** full league simultaneously, on a recent window (v1: `FIT_SEASONS` = 2024-25 and 2025-26, static team ratings), fit to **regulation goals** (periods 1–3, no empty-netters).

**Model:**

$$
\begin{aligned}
\mu_{\text{off},t} &\sim \mathcal{N}(0, \sigma_{\text{off}}^2) \quad \text{for each team } t \\
\mu_{\text{def},t} &\sim \mathcal{N}(0, \sigma_{\text{def}}^2) \quad \text{for each team } t \\
\alpha_0 &\sim \mathcal{N}(\log 3,\ 0.5^2) \quad \text{league baseline scoring rate} \\
\alpha_{\text{home}} &\sim \mathcal{N}(0, 1) \\
\log \lambda_{\text{home}} &= \alpha_0 + \alpha_{\text{home}} + \mu_{\text{off},h} - \mu_{\text{def},a} - \gamma_{a} \\
\log \lambda_{\text{away}} &= \alpha_0 + \mu_{\text{off},a} - \mu_{\text{def},h} - \gamma_{h} \\
\rho &= e^{\tau} - 1, \quad \tau \sim \mathcal{N}(0,\ 0.5^2) \quad \text{tie inflation} \\
P(G_{\text{home}} = h, G_{\text{away}} = a) &= \frac{\text{Pois}(h \mid \lambda_{\text{home}})\,\text{Pois}(a \mid \lambda_{\text{away}})\,(1 + \rho\,[h = a])}{1 + \rho\, T}, \quad T = \sum_k \text{Pois}(k \mid \lambda_{\text{home}})\,\text{Pois}(k \mid \lambda_{\text{away}})
\end{aligned}
$$

**Tie inflation.** Two independent Poisson scores end regulation tied about 17% of the time; real NHL games are tied after 60 minutes 23.7% of the time (2024-26). Late in tied games both teams play safe for the guaranteed point, and trailing teams press to equalize, so scores are not independent. $\rho$ multiplies every tied score by $1 + \rho$ and renormalizes ($T$ is the tie probability under plain Poisson). $\rho = 0$ is independent Poisson. In PyMC the two Poisson terms stay as observed variables and the correction $\log(1 + \rho[h=a]) - \log(1 + \rho T)$ is added with `pm.Potential` ($T$ summed over $k = 0..15$). `sample_scores` draws from this model exactly (a mixture: with probability $\rho T / (1 + \rho T)$ force a tie with $k \propto \text{Pois}(k \mid \lambda_h)\text{Pois}(k \mid \lambda_a)$, otherwise independent Poisson). Phase 6's exact win-probability model uses the same tie-inflated table; its score outputs come from the score-state simulator instead (see Phase 6). Why it matters: it sets how often games reach OT, which moves regulation-win probabilities and, above all, the number of loser points in simulated standings.

**Hyperpriors:** $\sigma_{\text{off}}, \sigma_{\text{def}} \sim \text{HalfNormal}(0.3)$ (team effects on the log scale are about ±0.1, so 0.3 is loose but rules out silly values).

**Goalie term is a fixed input, not estimated.** $\gamma_a$ and $\gamma_h$ are known numbers per game: $\gamma = -\log \theta$, where $\theta$ is the Phase 3 goalie rating (`goalie_rating_as_of`, posterior mean) of the goalie who **started** that game, computed as of that game's date so no game uses later information. Positive $\gamma$ = better goalie = fewer goals against. A goalie with no rating history gets $\gamma = 0$ (league average). This works like $\alpha_0$ and $\eta$ in Track A: a precisely known term held fixed.

Why not estimate a goalie effect inside Track B: from goal counts alone, a goalie who plays most games for one team is always seen alongside that team's defense, so the model can't tell good defense from good goaltending (only backups and trades separate them). It would also rate goalies a second time, in a place that can't see shot quality. With $\gamma$ fixed, $\mu_{\text{def}}$ means team defense *apart from* goaltending, and each goalie is rated once, by the model that sees shot quality.

**Cost:** Track B treats the goalie ratings as exact and ignores their uncertainty. The Phase 3 posterior sd is about ±0.03 on the log scale, small next to team effects, so this is acceptable for v1. The v2.1 hierarchical xG layer plans a fully Bayesian goalie effect whose posterior draws can replace the fixed values and carry that uncertainty through.

**Why the global intercept $\alpha_0$ is required.** Every team effect has a zero-mean prior (and the fixed goalie offsets average about 0), so without $\alpha_0$ the model is centered at $\log \lambda = 0$, i.e. about 1 goal per team per game instead of about 3. The team effects would have to shift together to make up the difference, fighting their own priors. $\alpha_0$ carries the league scoring level; the team effects and goalie offsets are then deviations from it. Its prior is centered on $\log 3 \approx 1.1$ and is wide enough to let the data decide.

**Remaining soft identifiability.** $\alpha_0$ can still trade off against the average of the $\mu_{\text{off}}$'s and the $\mu_{\text{def}}$'s. (The fixed $\gamma$'s average about 0, since the Phase 3 ratings are season-adjusted to league GA ÷ xGA = 1.) The zero-mean priors resolve this softly. On the real fit the correlation is strong ($r = -0.81$) but harmless: $\alpha_0$ still has ESS 2,239. Switch to sum-to-zero team effects (`pm.ZeroSumNormal`) only if $\alpha_0$'s ESS ever drops.

**Sampler:** NUTS. Write every random effect in non-centered form from the start ($\mu_{\text{off},t} = \sigma_{\text{off}}\, z_t$, $z_t \sim \mathcal{N}(0,1)$), because centered hierarchical models with small $\sigma$ produce funnel geometry and divergences.

**Output:** ArviZ NetCDF with the full joint posterior across all teams' offense and defense, plus $\alpha_0$, $\alpha_{\text{home}}$, the $\sigma$'s and $\rho$. The team indices and fixed $\gamma$ values used are stored in `constant_data`, the goals in `observed_data`.

**Purpose:** the actual production model.

**v1.1 (Phase 7, adopted).** The model above is v1 (goals only). Phase 7 replaced it with v1.1, which learns the team ratings mainly from regulation xG (Gamma likelihood) and fits goals on top with small finishing effects, on a 1-year window. Same outputs under the same names; see Phase 7 for the model and results.

### Cross-Validation Between Tracks

The two tracks only *should* agree when they are fit to the same data with the same likelihood. Comparing Track A on head-to-head games with full Track B on the whole league compares different datasets and different priors, so a gap between them proves nothing. The check is therefore done in three steps, from strict to loose.

**1. Synthetic recovery (both tracks).**
Simulate a full league season from Track B's generative model with known parameters.
- Fit Track B and check that the posteriors cover the true team effects (simulated with the same fixed goalie offsets).
- Fit Track A on 2–3 team pairs from the same simulated games and check that its $d_A, d_B$ intervals cover the true differences.

**2. Like-for-like on real data (the strict agreement test).**
For 2–3 team pairs, fit Track B in a *matched* configuration on the same $(A, B)$ head-to-head games Track A sees:
- no pooling ($\sigma_{\text{off}}, \sigma_{\text{def}}$ fixed at 1),
- goalie term and tie inflation switched off (Track A has neither),
- $\alpha_0$ and the home term fixed to the same values as Track A,
- the same goals Track A used (periods 1–4).

The priors differ slightly (Track A: uniform on $[-1, 1]$ for each $d$; matched Track B: $\mathcal{N}(0, 1)$ on each team term, so $\mathcal{N}(0, 2)$ on each $d$), which moves the means by about 0.001 with 18 games. A truly flat prior would leave NUTS wandering along the two directions the data can't pin down.

The two posteriors on $d_A, d_B$ should then match to within Monte Carlo error. A real disagreement here means a bug in the likelihood, the parameterization or one of the samplers.

**3. Pooled Track B vs Track A (qualitative only).**
Fit full Track B on league data and compute its posterior on the same $d_A, d_B$. It should be much narrower and shrunk toward zero, and it should usually sit inside Track A's wide interval. A pooled interval far outside Track A's is a flag to investigate, not an automatic failure, since pooling and the extra league data are supposed to move it.

---

## Build Sequence

Realistic for a side project with focused sessions a few times a week. Roughly two months to a usable single-game predictor.

### Phase 1 — Data Foundation (Weeks 1–2) — DONE

- ~~Expand `scripts/load_shots.py` to iterate over the full league schedule.~~ Done as `scripts/load_all.py`: all 32 teams, 5 seasons.
- ~~Verify shot data integrity.~~ `notebooks/data_sanity.ipynb`, sections 1–13, all passing.
- ~~Backfill 2–3 seasons.~~ 2021-22 through 2025-26 (6,560 games).

**Added beyond the original plan:**
- `game_events` table (full play-by-play, every event type), needed for rush detection.
- `shot_context` table with rebound / rush flags, prior event type and time since prior event.
- Fixes found by the checks: goal-shooter repair, blocked-shot zone frame, no-stoppage rule (see Data Contract → Known Data Quirks).
- External check against Natural Stat Trick (`scripts/nst_spotcheck.py`, `scripts/rush_event_audit.py`).

### Phase 2 — xG Model (Week 3) — DONE

**Files:**
- `src/processing/xg_processing.py` — `build_xg_features(conn)`: one row per shot, 13 features
- `src/models/deterministic/xg.py` — `ExpectedGoalsModel` with `.fit()`, `.predict()`, `.evaluate()`, `.calibration()`, `.feature_importance()`, `.save()`, `.load()`
- `src/pipelines/train_xg.py` — run with `py -m src.pipelines.train_xg [--final]`
- `notebooks/xg_features_check.ipynb`, `notebooks/xg_model_dev.ipynb`

**Features (13):** distance, angle, shot type (grouped), rebound flag, rush flag,
previous event type, time since previous event, skater difference, total skaters,
own goalie pulled, distance from previous event, speed from previous event,
rebound angle change.

**Model:** LightGBM, log-loss objective. `num_leaves=31`, `min_child_samples=200`,
`learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`. A 9-combination grid
search found no better setting (top six within 0.00005 log loss; 63 leaves worse).

**Season roles:** train 2021-22 to 2023-24, validate 2024-25 (early stopping picks
413 trees), then refit on 2021-22 to 2024-25 and test once on 2025-26.
Recency weights 1.0 / 1.25 / 1.5 / 2.0, oldest to newest. Random seed 42.

**Results on the 2025-26 test season:**

| | Baseline (distance + angle, logistic) | Final LightGBM |
|---|---|---|
| AUC | 0.714 | **0.769** |
| Log loss | 0.2307 | **0.2167** |
| Predicted vs actual goals | +1.4% | +2.5% (7,770 vs 7,578) |
| Calibration | Mid-range under, top over | Within 0.8 points in every decile |

What moved the score, in order: LightGBM over logistic regression; previous-event
features; recency weighting; shot-type grouping (fixed the goal total).

**MoneyPuck comparison (2025-26, 5-on-5, team-season totals, 32 teams):**
xGF r = 0.970, xGA r = 0.940, total xG ratio 1.000. MoneyPuck blocks scripted
downloads; save `teams.csv` by hand to `data/external/`.

**Saved models (`output/models/`), version v1:**
- `xg_lgbm_v1_20212022_20242025.pkl` + `_metrics.json` — the evaluated model
- `xg_lgbm_v1_live.pkl` + `_metrics.json` — same settings, trained on all five seasons, for scoring new games
- `XG_MODEL_PATH` in `config/settings.py` names the active model; load it with `load_active_model()`.
- Metrics files record version, seed, settings, features, training seasons, data cutoff and library versions.
- `train_xg.py` refuses to overwrite an existing file; bump `MODEL_VERSION` in `xg.py` when features or settings change.

**Gaps against the original plan:**
- MoneyPuck validation was done on team-season totals, not per-game totals. Per-game is stricter and still to do.
- Method is named `.predict()`, not `.predict_proba()`.

### Phase 3 — Team Strength & Goalie Aggregates (Week 4) — DONE

**Files:**
- `src/pipelines/score_xg.py` — scores every shot into the `shot_xg` table. Run with `py -m src.pipelines.score_xg [--season YYYYYYYY]`.
- `src/processing/team_strength.py` — `team_game_xg(conn)`, `rolling_strength(tg, window=25)`, `strength_as_of(tg, date)`
- `src/processing/goalie.py` — `goalie_game_xg(conn)`
- `src/models/stochastic/goalie.py` — `goalie_rating_as_of(gg, date, k=800, lookback_days=730, season_adjust=True)`
- `notebooks/phase3_check.ipynb` — correctness checks plus the tests that chose the window and `k`

**Scoring every shot (`shot_xg`).** Each season 2021-22 to 2025-26 is scored by a
leave-one-season-out model (trained on the other four seasons, 413 trees, same settings
as v1), so no shot is scored by a model that saw it. Seasons the live model never
trained on (2026-27 onward) are scored by the live model. xG / goals by season:
1.032, 1.031, 0.974, 0.969, 1.024 (see *Season-level xG offset* in Open Questions).

**Team strength.** One row per team per game (`team_game_xg`): xGF, xGA, goals for and
against (periods 1–4, empty-net goals included) and game length in minutes (overtime
counted). `rolling_strength` adds each team's estimate going *into* every game from its
previous 25 games: xGF/60 and xGA/60 as total xG ÷ total minutes × 60, and their variance
(game-to-game variance ÷ games). Windows span seasons. `strength_as_of` gives the same
estimate for every team on any date.

**Goalie rating.** `goalie_game_xg` credits each shot to the goalie in net, so a pulled
starter and his replacement each carry only the shots they faced. The rating is
Gamma-Poisson on goals allowed per expected goal:

$$
GA \sim \text{Poisson}(\theta \cdot xGA), \qquad \theta \sim \text{Gamma}(k, k), \qquad
\theta \mid \text{data} \sim \text{Gamma}(k + GA,\; k + xGA)
$$

$\theta < 1$ is better than average. Each season's xGA is first rescaled so league
GA ÷ xGA = 1.0 within that season (using only games before the rating date). Output per
goalie: posterior mean and variance of $\theta$, a 90% credible interval, raw GSAx and
games. This is the swappable goalie interface; on Track B's additive log scale,
$\gamma_g \approx -\log \theta_g$ (positive = better).

**Results (`phase3_check.ipynb`):**

Correctness: two rows per game (13,120), league xGF = league xGA, xGF and goalie xGA per
season match `shot_xg` exactly, every goalie-game named, and a hand-computed pre-game
estimate matches `rolling_strength` (no leakage).

Team strength predicting the next game (6,381 games where both teams had 10+ games of history):

| Window | Correlation with goal diff (xG) | (goals) | Picks winner (xG) |
|---|---|---|---|
| 10 | 0.233 | 0.168 | 58.2% |
| 25 | **0.251** | 0.206 | **59.3%** |
| 50 | **0.251** | 0.240 | 59.1% |

xG-based estimates beat goals-based ones at every window. 25 and 50 tie; 25 is kept
because it reacts faster.

Goalie rating predicting the next season (2022-23 to 2025-26 pooled, 253 goalie-seasons
with 20+ games, correlation margin of error about ±0.12):

| k | Weighted error (adjusted) | Correlation (adjusted) | Correlation (unadjusted) |
|---|---|---|---|
| 1 (raw record) | 0.0284 | 0.037 | 0.043 |
| 100 | 0.0138 | 0.100 | 0.094 |
| 400 | 0.0120 | 0.120 | 0.105 |
| **800** | **0.0119** | 0.124 | 0.106 |
| 1600 | 0.0119 | 0.125 | 0.106 |

The season adjustment improves error and correlation from k = 100 up. Error is lowest at
k = 800 and flat from about 400 to 1,600. The signal is real but weak: a goalie's past two
seasons are a modest guide to his next one, consistent with public hockey analytics. With
k = 800, ratings for big-sample goalies span roughly 0.96 to 1.04.

**Settings kept:** team window 25 games; goalie `k = 800`, `season_adjust=True`,
`lookback_days = 730`.

**Notes:**
- `games` in goalie output counts appearances, so a short relief stint counts as one game. The rating itself is weighted by xGA faced, not games.
- Utah (`UTA`) has a different abbreviation from Arizona, so its team-strength history starts fresh in 2024-25.
- Goalie names come from `goalie_game_stats` (boxscores), which covers every goalie who played, not only current rosters.

### Phase 4 — Track A Toy MCMC (Week 5) — DONE

**Files:**
- `src/models/stochastic/am_sampler.py` — `adaptive_metropolis`, `autocorrelation`, `plot_traces`, `plot_acf`, ported unchanged from the soil project
- `src/models/stochastic/toy_mcmc.py` — `league_baseline(tg)`, `pair_games(tg, team_a, team_b)`, `make_log_posterior(games, alpha0, eta)`, `simulate_pair_games(n_games, d_A, d_B, alpha0, eta, seed)`
- `src/pipelines/fit_toy.py` — fits one pair, 4 chains. Run with `py -m src.pipelines.fit_toy COL DAL [--overwrite]`.
- `notebooks/phase4_toy.ipynb` — function checks and the four recovery tests

**Model.** Two parameters, uniform prior on $[-1, 1]^2$, Poisson likelihood on `gf_xg` / `ga_xg`:

$$\log \lambda_A = \alpha_0 + \eta \cdot \text{home}_A + d_A, \qquad \log \lambda_B = \alpha_0 + \eta \cdot \text{home}_B + d_B$$

with $\alpha_0 = 1.0291$ (2.799 away goals/game) and $\eta = 0.0715$ (home teams score 7.4% more, 3.006 goals/game) fixed from league data by `league_baseline`. Empty-net goals are 5.9% of all goals.

**Exact posterior (used as a check).** With a flat prior on $d$, the posterior of $e^{d}$ is $\text{Gamma}(Y, B)$, where $Y$ = total goals and $B = \sum_g e^{\alpha_0 + \eta \cdot \text{home}_g}$. So $E[d] = \psi(Y) - \log B$ and $\text{sd}(d) = \sqrt{\psi_1(Y)}$. The sampler is compared against this, not just against itself.

**Results (`phase4_toy.ipynb`, truth $d_A = 0.15$, $d_B = -0.10$):**

| Test | Result |
|---|---|
| `log_post` vs scipy Poisson (differences) | Match to 6 decimals |
| 1. One fit, 15 games, 4 chains | Truth inside both 90% intervals; acceptance 0.35–0.37; R-hat 1.0001; ACF below 0.05 by lag 11 |
| 2. 500 games | Posterior narrows 5.7× and 6.1× (expected about 5.8×) |
| 3. Coverage, 100 seeds × 15 games | 90/100 and 96/100 (target 90, normal range 84–96) |
| 4. MCMC vs exact | Means within 0.002, sds within 1% |

In Test 1 the truth sits about 0.9 sd from the posterior mean of $d_B$. That comes from the 15 simulated games (B scored below its true rate by chance), not from the sampler: MCMC matches the exact posterior for that data. Coverage is what tests calibration.

**Real pair, COL vs DAL** (18 games, 2021-11-26 to 2026-04-04, COL home in 9; `fit_toy.py`, 4 × 50,000 draws):

| | Mean | 90% interval | Rate vs league average | Exact mean |
|---|---|---|---|---|
| $d_A$ (COL scoring vs DAL) | +0.114 | [−0.106, +0.325] | ×1.12 | +0.113 |
| $d_B$ (DAL scoring vs COL) | −0.055 | [−0.296, +0.173] | ×0.95 | −0.054 |

R-hat 1.0001–1.0002, ESS about 26,000–28,000, acceptance 0.34–0.36. Both intervals cover 0: 18 games can't separate the two teams, as expected.

**What Track A is for downstream.** Nothing in Phases 6–8 reads Track A's numbers; they use Track B. Track A's value is what it proves and what it provides as a reference:
- **Phase 5 (Track B) is checked against it.** The matched Track B fit on the same head-to-head games, with the same fixed $\alpha_0$ and $\eta$, must reproduce the `toy_<A>_<B>.nc` posteriors. A gap means a bug in Track B's likelihood or parameterization, found before Track B is trusted league-wide.
- **The simulator and its exact answer are reusable.** `simulate_pair_games` and the Gamma check extend to Track B's synthetic-recovery test (Cross-Validation step 1).
- **It fixed the shared conventions** every later model uses: the goals target (`gf_xg`), the fixed-league-term approach, the posterior file format (NetCDF with observed data and settings attached) and the diagnostics (R-hat, ESS, acceptance).
- **The sampler is verified,** so if it is ever needed again (for example, PCE-based inference if a slow possession-level simulator replaces the Poisson game model) it is ready.

**Settings kept:** 4 chains, 5,000 burn-in + 50,000 draws, $\sigma_0 = 0.1$, seed 42 (+ chain number), starts spread over the prior box.

**Notes:**
- **ArviZ 1.x** (current; installed with Python 3.14) returns an xarray `DataTree` instead of `InferenceData`, and `az.from_dict` takes one nested dictionary. `fit_toy.py` handles both 0.x and 1.x. Version 1.x drops `attrs` passed to `from_dict`, so model settings are written to `posterior.attrs` directly.
- **Writing `.nc` files needs `h5netcdf` and `h5py`:** `py -m pip install arviz h5netcdf h5py`.
- Summary statistics in `fit_toy.py` (mean, sd, equal-tailed 90% interval) are computed with numpy so they match the notebook; R-hat and ESS come from ArviZ.
- Acceptance of about 0.35 is right for two parameters (0.234 is the many-dimension optimum).

### Phase 5 — Track B Hierarchical PyMC (Week 6) — DONE

**Files:**
- `src/processing/game_table.py` — `starting_goalies(conn)`, `regulation_goals(conn)`, `goalie_gamma(gg, dates)`, `build_game_table(conn, seasons)` → `(gt, teams)`: one row per game with team indices, goals (periods 1–4 and regulation), starters and their fixed $\gamma$
- `src/models/stochastic/hierarchical.py` — `build_model(gt, teams, matched=False, ..., goal_cols=GOAL_COLS, tie_inflation=True)`, `sample(model, seed)`, `sample_scores(lam_home, lam_away, rho, rng)`, `simulate_league(...)`, `team_ratings(idata, teams)`
- `src/pipelines/fit_hierarchical.py` — `py -m src.pipelines.fit_hierarchical [--seasons ...] [--overwrite]`; writes `hier_<first>_<last>.nc`, `_meta.json`, `_ratings.csv`, `_trace.png` to `output/posteriors/`
- `notebooks/phase5_check.ipynb` — eleven sections of checks (below); section 11 tests the score-effects explanation
- `scripts/test_pymc.py` — install check: tiny Poisson fit against the exact answer

**Install:** `py -m pip install pymc` works on Python 3.14 (PyMC 6.3.2, PyTensor 3.3.4). PyTensor found the MSYS2 g++ compiler (`C:\msys64\ucrt64\bin`), so sampling runs compiled. Sampling uses `cores=1` (chains one after another) on Windows.

**Game table** (2024-25 + 2025-26): 2,624 games, 32 teams, 164 games each. Starter found for every side. Regulation goals per game: home 2.867, away 2.679. **23.7% of games tied after regulation**; 15.3% decided by an OT goal (the rest by shootout). Goalie $\gamma$ from −0.050 to +0.066; 0.3% of sides have no rating history ($\gamma = 0$). Recomputing one starter's $\gamma$ from earlier games only matches exactly (no leakage).

**Results (`phase5_check.ipynb`):**

| Check | Result |
|---|---|
| Synthetic recovery (3 fake leagues × 2,480 games, true $\rho = 0.5$) | Team effects: 88.5% of 192 90% intervals cover the truth; league terms 15/15; $\rho$ estimated 0.61, 0.44, 0.54; 0 divergences |
| Matched check vs Track A (COL–DAL, 18 games) | $d_A$ +0.1113 vs +0.1141, $d_B$ −0.0525 vs −0.0553; sds within 1% |
| Sampler (real fit, about 50 s) | R-hat ≤ 1.0037, ESS ≥ 1,570, 0 divergences, BFMI 0.62–0.71 |
| Home ice | $\alpha_{\text{home}}$ = 0.0713 [0.043, 0.099]; direct league ratio on the same games 0.0676 |
| Scoring level | $\alpha_0$ = 0.992 [0.958, 1.025] (regulation away goals ≈ 2.70) |
| Team spread | $\sigma_{\text{off}}$ = 0.068, $\sigma_{\text{def}}$ = 0.064: a typical team is about ±7% on offense and on defense |
| Tie inflation | $\rho$ = 0.51 [0.40, 0.63]: regulation ties about 51% more likely than independent scoring |
| Sanity | Net rating vs goal differential r ≈ 0.998; vs **xG differential r ≈ 0.79** (never seen by the model) |

Top teams by net rating: COL (about +0.18), TBL, DAL, CAR, VGK; bottom: SJS (about −0.17), CHI, VAN, NSH, ANA. 90% intervals are about ±0.09 for every team because every team gives the model the same data (164 games, about 450 goals each way); the middle of the league overlaps heavily. Extreme goal differentials are shrunk (SJS −0.77 goals/game → −0.17; COL +0.77 → +0.18).

**Posterior predictive check** (every game re-simulated from 500 posterior draws; regulation goals):

| Statistic | Actual | Model 90% | Plain Poisson 90% |
|---|---|---|---|
| Goals per team-game | 2.773 | [2.724, 2.822] ✓ | [2.744, 2.843] ✓ |
| Variance ÷ mean | 0.922 | [0.961, 1.031] ✗ | [0.995, 1.066] ✗ |
| Share 0 goals | 0.052 | [0.054, 0.066] ✗ | [0.058, 0.070] ✗ |
| Share 5+ goals | 0.138 | [0.137, 0.155] ✓ | [0.144, 0.164] ✗ |
| Home win in regulation | 0.420 | [0.393, 0.433] ✓ | [0.426, 0.468] ✗ |
| Tied after regulation | 0.237 | [0.219, 0.256] ✓ | [0.159, 0.182] ✗ |

Tie inflation fixes ties and regulation home wins. What remains: each team's regulation goals are about 8% less spread than Poisson (more 2–3 goal games, fewer shutouts), and one-goal margins are more common than the model gives (home +1: 18.3% actual vs 14.7%).

**Why (section 11, from play-by-play):**

| Test | Result |
|---|---|
| Variance ÷ mean by periods played (no empty-netters) | Period 1: 0.981 (inside Poisson range 0.975–1.037) → periods 1–2: 0.960 → regulation: 0.922. The squeeze builds up during the game. |
| Scoring vs each team's own Track B pace, by score state | Trailing 2+: 1.11 · trailing 1: 1.07 · tied: 0.96 · leading 1: 0.94 · leading 2+: 0.93 (each ±0.02–0.04) |
| Regulation variance ÷ mean with empty-net goals put back | **1.010, inside the Poisson range** (0.986–1.053); removing empty-netters causes about 90% of the gap |

So **score effects are real** (trailing teams score about 7–11% above their normal pace, leaders about 6–7% below), but **most of the "too bunched" result comes from the goal definition**: empty-net goals go to leading teams, so leaving them out trims the high end of the counts. With them included, regulation goals are Poisson-like.

**Consequences:** no change to Track B (ratings measure normal-play strength, and empty-net goals never change the winner).

**Correction (found in Phase 6).** Reading this as "mostly a counting artifact" was wrong. Real regulation goals without empty-netters are compressed (0.92) by score effects, and empty-net goals then add spread back (1.01); the two are the same late-game behavior. Track B's regulation goals are Poisson-like, so adding empty-net goals on top over-spread final scores (1.06–1.07 vs 0.98). Phase 6 therefore built the score-state simulator for score outputs (see Phase 6).

**How the model got here** (each change found by a check):
1. First fit on periods 1–4 goals: the predictive check showed 8.5% "ties" (only shootout games) vs 16–18% simulated, because OT goals were counted as ordinary scoring. → Fit regulation goals; Phase 6 simulates OT.
2. Regulation fit: ties 23.7% actual vs 16–18% simulated, home regulation wins too high. → Tie inflation $\rho$.
3. Dispersion still below Poisson. → Section 11 traced it to score effects plus the empty-net exclusion. → No change to Track B; Phase 6 score-state simulator for score outputs.

**Settings kept:** window 2024-25 + 2025-26; priors as in *Track B*; NUTS 4 chains × (1,000 tune + 1,000 draws), `target_accept = 0.9`, seed 42.

**Notes:**
- **Windows file lock.** `az.from_netcdf` keeps the file open; a notebook kernel that loaded a posterior locks it and `--overwrite` fails (`GetLastError() = 33`). The notebook's `load_posterior()` reads into memory and closes the file. Restart any old kernel (or close VS Code) to release a lock. A safer save (write to `.tmp`, then swap) is written but not yet installed in `fit_hierarchical.py`.
- `fit_hierarchical.py` sets matplotlib's file-only backend inside `main()`, not at import, so notebooks that import from it keep inline plots.
- An OT goal after an *untied* regulation can happen when a removed empty-net goal hid a real regulation tie (leader scores into the empty net, trailer scores twice). The check allows up to 1% of games.
- Sampling took about 13 s with plain Poisson and about 50 s with tie inflation (the tie sum runs over 16 scores per game).

### Phase 6 — Game Model + Monte Carlo (Week 7) — DONE

**Files:**
- `src/processing/late_game.py` — `game_results`, `mean_rates`, `fit_ot_factor`, `shootout_home`, `empty_net_table`, `late_game_params`. `py -m src.processing.late_game` → `output/models/late_game.json`
- `src/processing/score_state.py` — `load_events`, `measure`, `summarize`, `score_state_params`. `py -m src.processing.score_state` → `output/models/score_state.json` (includes the simulator's calibrated `scale`)
- `src/models/game_model.py` — `load_track_b`, `default_starter`, `goalie_offset`, `matchup_rates`, `regulation_table`, `outcome_probs`, `final_score_table`, `predict_matchup`
- `src/models/stochastic/montecarlo.py` — tie-inflated simulator (`simulate_games`, `simulate_matchup`, `summarize_sims`) and score-state simulator (`simulate_regulation_score_state`, `simulate_games_score_state`, `simulate_matchup_score_state`, `calibrate_scale`)
- `src/analysis/aggregation.py` — `fair_american_odds`, `outcome_table`, `likely_scores`, `totals_and_lines`, `final_from_sims`, `format_prediction`, `result_to_json`
- `src/pipelines/predict_game.py` — `py -m src.pipelines.predict_game HOME AWAY [--date YYYY-MM-DD] [--home-goalie ID] [--away-goalie ID] [--score-sims N] [--check-sims N] [--save]`; saved predictions go to `output/predictions/<date>_<away>_at_<home>.json`
- `notebooks/phase6_check.ipynb` — nine sections (below)

**Rebuild order** whenever the data or the fit changes (each step uses the previous one's output; since Phase 7 the fit is v1.1 on the last 365 days and becomes the active fit):
0. After loading new games: `py .\scripts\load_all.py`, then `py -m src.pipelines.score_xg --season <season>` (the fit stops if this was skipped)
1. `py -m src.pipelines.fit_hierarchical` (add `--overwrite` to refit the same end date)
2. `py -m src.processing.late_game`
3. `py -m src.processing.score_state`

**In-season routine (2026-27):**
- **Every morning:** `load_all.py` → `score_xg --season 20262027` → `make_cards postgame --pending` (cards for last night's games) → `predict_day` (forward-test log + a pregame card per game; add `--goalie TEAM=NAME --replace` (last name or id) when starters are confirmed, which redraws that card).
- **Weekly:** `load_all.py` → `score_xg --season 20262027` → `fit_hierarchical` → `late_game` → `score_state` → `build_pce` (optional) → `score_forward`.

**Two models, each for what it does best:**

| Output | Source | Why |
|---|---|---|
| P(win), regulation / OT / shootout split, with 90% intervals | **Exact model** (`game_model.py`): tie-inflated regulation table (16 × 16) per posterior draw, then OT and shootout formulas | Analytic and fast; one P(win) per draw, so the mean is the prediction and the spread is the uncertainty about team strength (game randomness is already inside each P) |
| Likely final scores, totals, puck lines, margins | **Score-state simulator** (100,000 games per prediction) | Reproduces real score shapes (section 9) |

The prediction header prints the simulator's win share next to the exact one as a running consistency check.

**Game flow:**
1. **Regulation.** Rates from Track B: $\lambda = \exp(\alpha_0 [+ \alpha_{\text{home}}] + \mu_{\text{off}} - \mu_{\text{def,opp}} - \gamma_{\text{opp goalie}})$, tonight's starters' $\gamma$ from the Phase 3 rating as of game day (default starter: most starts in the team's last 10 games).
2. **Overtime** if tied: $P(\text{OT goal}) = 1 - e^{-k(\lambda_h + \lambda_a)\cdot 5/60}$, home scores it with probability $\lambda_h / (\lambda_h + \lambda_a)$.
3. **Shootout** if still tied: 50/50.
4. **Official final score:** empty-net goals plus 1 for the OT or shootout winner.

**Late-game parameters (`late_game.py`, 2024-26):**

| Quantity | Value |
|---|---|
| How games ended | Regulation 77.2%, OT 15.3%, shootout 7.5%; home won 54.2% |
| OT factor $k$ | **2.39** [2.20, 2.61]: 3-on-3 scores at about 2.4× the regulation rate. Fitted share of OT games decided in OT 67.1% vs actual 67.2% |
| Who scores in OT | Home 52.1% vs 51.9% expected from ratings (no separate OT home edge needed) |
| Shootout | 196 games, home 49.5% [43.6, 55.3], p = 0.94 vs 50/50 → 50/50 |
| Leader's empty-net goals by regulation margin (0 / 1 / 2+) | Margin 1: 51.0 / 40.9 / 8.1% · margin 2: 29.5 / 68.7 / 1.8% · margin 3+: 79.1 / 20.9 / 0% (used by the tie-inflated simulator) |

**Score-state parameters (`score_state.py`, from `situation_code` goalie digits):**

| Situation | Value |
|---|---|
| Normal play (both goalies in), goals vs own Track B pace | Trailing 2+: 1.057 · trailing 1: 1.036 · tied: 0.974 · **tied, last 5 min of P3: 0.768** · leading 1: 0.958 · leading 2+: 0.956 |
| Goalie pulled, trailer's 6-on-5 factor | 2.72 (down 1) · 2.74 (down 2) · 2.63 (down 3+) |
| Goalie pulled, leader's empty-net goals | 0.358 / 0.388 / 0.319 per minute (down 1 / 2 / 3+) |
| Share pulled by 2:00 / 1:00 left (no goals in between) | Down 1: 44% / 98% · down 2: 92% / 98% · down 3+: 29% / 31% |
| Empty-net goals outside a late trailing pull | 113 (about 11%; delayed penalties, early pulls; not simulated) |

**Score-state simulator.** Regulation in 360 slices of 10 seconds. Each slice: (1) a trailing team in the last 6 minutes may pull its goalie (measured hazard by margin and 30-second bin); (2) rates: pulled trailer = own rate × 6-on-5 factor, leader facing an empty net = empty-net rate, otherwise own rate × score-state factor; (3) goals; after any goal both goalies return. **No tie inflation**: extra ties come from "tied late" caution and 6-on-5 equalizers. One overall `scale` (calibrated on the fit games) keeps average goals with a goalie in net equal to Track B's rates. 10-second slices keep late pulls realistic; two goals in one slice happen about once in 15,000 slices.

**Results (`phase6_check.ipynb`):**

| Check | Result |
|---|---|
| Exact vs simulated (20 matchups × 50,000 games) | All outcomes agree, worst \|z\| < 4; probabilities sum to 1; final scores have no ties |
| History: home win | 54.2% actual vs 53.4% expected [51.8, 55.0] |
| History: regulation / OT / shootout | 77.3 / 15.3 / 7.5% actual vs 76.3 / 15.9 / 7.8% expected, all inside 90% ranges (regulation about 1 point low: the "hidden tie" effect in the exact model) |
| Calibration preview (in-sample) | Log loss 0.6758 vs 0.6896 for "home team 54%"; Brier 0.2414 vs 0.2482; picks the winner 57.3%; 7 of 10 deciles inside their range |
| Sanity | Home ice +3.3 points for two average teams; COL vs SJS 69.3% at home, 63.2% away; a ±0.04 goalie offset swings P(win) 3.7 points |
| Exact vs score-state P(win), 40 games | Mean absolute difference 0.5 points (simulation noise 0.35) |

**Score shape, all 2,624 games (section 9):**

| | Actual | Tie-inflated sim | Score-state sim |
|---|---|---|---|
| Past regulation | 22.75% | 23.50% | **22.79%** |
| Variance ÷ mean, regulation without empty-netters | 0.922 | 1.005 | **0.938** |
| Variance ÷ mean, regulation with empty-netters | 1.010 | 1.115 | **1.048** |
| Variance ÷ mean, final goals | 0.979 | 1.072 | **1.011** |
| One- / two- / three-goal games | 40.7 / 19.7 / 22.9% | 38.1 / 17.5 / 27.9% | **41.2 / 19.6 / 24.3%** |
| Empty-net share of regulation goals | 6.62% | 6.34% | **6.62%** |
| Mean final total | 6.167 | 6.155 | 6.198 |
| Home win | 54.2% | 53.0% | 52.9% |

The score-state simulator is closer on 9 of 11 rows, hits the past-regulation share exactly without a tie parameter (which also resolves the empty-net "hidden tie" mismatch for score outputs), and closes about 70% of the final-score spread gap. Part of the remaining 1.011 vs 0.979 is expected: each simulated game draws its ratings from the posterior, which adds a little spread.

**Example** (`predict_game COL DAL --date 2026-04-10`, Blackwood vs Oettinger by default): COL 55.8% (90% 50.0–61.6%), fair odds COL −126 / DAL +126; tied after regulation 23.8%.

**How Phase 6 got here:**
1. Exact model + tie-inflated simulator with an empty-net table: win and OT/SO shares matched history, but final scores were too spread (1.06 vs 0.98) with too many 3-goal margins.
2. That showed score effects matter for score outputs (correction to the Phase 5 reading). → Measured score effects, pull timing and 6-on-5 / empty-net rates; built the score-state simulator; switched score outputs to it.
3. A tie parameter on top of the score-state simulator was rejected: ties already match, and it would double-count them.

**Notes:**
- `predict_game.py` with today's date looks for the most recent starters; until 2026-27 games are loaded it uses the last starters on record. To predict this season's games: load 2026-27, then run the rebuild order above.
- The exact model's past-regulation share is 23.5% vs 22.8% real (empty-net "hidden ties"); harmless for P(win), and score outputs no longer use it.
- Calibration is too cautious at the extremes in-sample (predicted 43.2% → actual 37.3%; 59.6% → 69.1%; 63.4% → 68.8%). Unusual in-sample, where over-confidence is normal; leading suspect is static two-season ratings blurring teams that changed. First question for Phase 7. **Answered in Phase 7: it did not hold out of sample.** On held-out games the two-season v1 was *too confident* (slope 0.79).

### Phase 7 — Validation (Week 8) — DONE

**Files:**
- `src/validation/backtest.py` — `fit_window`, `predict_block`, `walk_forward(conn, season, window_days, refit_days=7, score_sims=5000, model="goals"|"xg", half_life=None, ...)`, `model_label` (e.g. `trackb_w730`, `trackbxg_w365`, `trackbxg_w365_hl90`)
- `src/validation/baselines.py` — `home_rate`, `elo` (tuned), `rolling_xg`, `all_baselines`
- `src/validation/scoring.py` — `log_loss`, `brier`, `accuracy`, `calibration_slope`, `calibration_table`, `score_models`, `paired_difference`, `by_group`, `total_goals_rps`, `score_outputs`
- `src/models/stochastic/hierarchical_xg.py` — Track B v1.1 (below): `build_model_xg(gt, teams, half_life_days=None, ref_date=None)`, `game_weights`, `simulate_league_xg`
- `src/processing/game_table.py` — added `regulation_xg(conn)`; the game table now carries `home_xg_reg` / `away_xg_reg`
- `src/pipelines/backtest_season.py` — `py -m src.pipelines.backtest_season --season 20252026 [--model goals|xg] [--window 365] [--half-life 90] [--refit-days 7] [--restart]` → `output/backtests/<label>_<season>.csv` (appended after every refit, so a stopped run resumes) and `<label>_<season>_scored.csv`
- `src/pipelines/run_baselines.py` — `py -m src.pipelines.run_baselines --seasons 20242025 20252026` → `baselines_<season>.csv` + `_info.json`
- `notebooks/phase7_check.ipynb` — finds every `trackb*_scored.csv` on its own; sections: data checks, model table per season and combined, paired head-to-head, calibration, by month/season, interval widths, score outputs, blends, market

**Backtest design (leak-free).** Test seasons 2024-25 and 2025-26 (1,312 games each). Before opening night: fit Track B on the `window_days` before it and measure the late-game and score-state parameters on those games only. Then refit every 7 days on the window before the refit date and predict the next week's games. Goalie offsets use the actual starters (known by game time, as they are for closing lines) rated as of the game date; each season's xG comes from a model that never trained on it. About 50 s per weekly refit (v1.1 about 2–3 min), about 26 refits per season.

**Competitors:**

| Model | How it predicts |
|---|---|
| Home rate | Home win share of earlier seasons, same for every game |
| Elo | Grid-tuned on earlier seasons only (k, home edge, off-season carry-over); first season is burn-in |
| Rolling xG | Logistic regression on the difference in each team's last-25-game xG rate (Phase 3), needs 10+ games |
| Track B v1 | Goals only, 1-year (`w365`) or 2-year (`w730`) window |
| Track B v1.1 | xG + goals, 1- or 2-year window, optional recency half-life |

**Track B v1.1 — ratings from xG and goals** (`hierarchical_xg.py`). Built in Phase 7 because v1 lost to rolling xG. Each game gives two observations per side (regulation, no empty-netters):

$$
\begin{aligned}
\log \nu_{\text{home}} &= \beta_0 + \beta_{\text{home}} + \mu^{xG}_{\text{off},h} - \mu^{xG}_{\text{def},a}, \qquad xG_{\text{home}} \sim \text{Gamma}(\text{mean } \nu_{\text{home}},\ \text{shape } \kappa) \\
\log \lambda_{\text{home}} &= \log \nu_{\text{home}} + \delta_0 + \delta_{\text{home}} + \phi_{\text{off},h} - \phi_{\text{def},a} - \gamma_a \\
(G_{\text{home}}, G_{\text{away}}) &\sim \text{tie-inflated Poisson}(\lambda_{\text{home}}, \lambda_{\text{away}};\ \rho)
\end{aligned}
$$

and the mirror image for the away side. $\mu^{xG}$ are shot-quality ratings, learned mostly from xG (the steadier signal). $\phi$ are finishing / team-save effects beyond xG and the goalie, tightly pooled ($\sigma_\phi \sim \text{HalfNormal}(0.05)$) so they stay near 0 unless the data clearly support them. The goalie $\gamma$ acts on goals only (goalies change conversion, not shot quality). Priors: $\beta_0 \sim \mathcal{N}(\log 2.7, 0.5^2)$, $\beta_{\text{home}} \sim \mathcal{N}(0, 0.5^2)$, $\delta \sim \mathcal{N}(0, 0.2^2)$, $\sigma \sim \text{HalfNormal}(0.3)$, $\log \kappa \sim \mathcal{N}(\log 9, 1)$, tie prior as v1. Both likelihoods are added with `pm.Potential`. The model also saves goal-scale ratings under the v1 names ($\alpha_0 = \beta_0 + \delta_0$, $\alpha_{\text{home}} = \beta_{\text{home}} + \delta_{\text{home}}$, $\mu = \mu^{xG} + \phi$), so the game model, simulators and backtest work unchanged. Optional recency weights $0.5^{\text{age}/\text{half-life}}$ multiply each game's log-likelihood.

Synthetic recovery (fake league, 2,480 games): correlation with the true goal-scale ratings 0.93 / 0.94 (offense / defense) vs 0.82 / 0.88 for goals-only v1 on the same data; 0 divergences. PyTensor sometimes prints "overflow encountered in dot" during tuning; those steps are rejected and harmless (no divergences).

**Results: P(home win), 2,624 held-out games** (log loss, lower is better; coin flip 0.6931):

| Model | 2024-25 | 2025-26 | Combined | Calibration slope [90%] |
|---|---|---|---|---|
| Home rate | 0.6870 | 0.6929 | 0.6900 | n/a |
| Elo (tuned) | 0.6721 | 0.6922 | 0.6822 | 0.64 (too confident) |
| **Rolling xG** | **0.6669** | **0.6838** | **0.6754** | 0.83 (too confident) |
| Track B v1, 2-yr | — | — | 0.6811 | 0.79 (too confident) |
| Track B v1, 1-yr | — | — | 0.6798 | 1.06 |
| **Track B v1.1, 1-yr** (adopted) | 0.6670 | 0.6877 | 0.6773 | **0.99** [0.80, 1.16] |
| Track B v1.1, 2-yr, half-life 180 d | 0.6673 | 0.6854 | 0.6763 | 1.20 [0.97, 1.41] |
| Track B v1.1, 1-yr, half-life 90 d | 0.6683 | 0.6841 | 0.6762 | 1.50 [1.24, 1.73] (too cautious) |
| Track B v1.1, 1-yr, half-life 60 d | 0.6700 | 0.6841 | 0.6771 | 1.61 [1.32, 1.89] (too cautious) |

Slope 1 = calibrated; below 1 = too confident; above 1 = too cautious.

**Paired comparisons** (same games, bootstrap 90% range; negative = first model better):
- Every v1.1 version clearly beats v1 and Elo.
- v1.1 vs rolling xG: 1-yr +0.0020 [−0.0020, +0.0058]; hl180 +0.0010; hl90 +0.0008 [−0.0031, +0.0046]; hl60 +0.0017. **All tied**: every range spans 0.
- hl90 vs hl60: −0.0009 [−0.0016, −0.0002]. The half-life versions vs the unweighted 1-yr: not significant.
- Averaging any v1.1 version with rolling xG beats that v1.1 alone by 0.0023–0.0031 (best blend 0.6738, with hl90). The two models make different mistakes.

**Score outputs** (over 5.5 / 6.5, puck lines, total-goals RPS from the score-state simulator): slightly better than predicting each market's historical rate. Totals are limited by season-to-season drift in league scoring, which a window fit can't see coming.

**What Phase 7 found:**
1. **The Phase 6 "too cautious" result did not hold out of sample.** On held-out games, v1 with two seasons was too confident (0.79): static ratings over two seasons treat old games as current. One season fixed calibration (1.06) and improved log loss.
2. **Goals alone are not enough.** v1 lost to rolling xG. Putting xG inside Track B (v1.1) closed the gap to a statistical tie, kept calibration (0.99) and beat v1 and Elo clearly.
3. **Recency helps only when teams change, and costs calibration.** Half-lives of 60–90 days matched rolling xG in 2025-26 (0.6841 vs 0.6838) but were worse in the steadier 2024-25 (0.6700 vs 0.6670), and made predictions too cautious: down-weighting games leaves fewer effective games, so every team's rating is less certain, and averaging over that pulls predictions toward 50%. A half-life forgets every team at the same rate whether it changed or not.
4. **No Track B version beats rolling xG,** and the blend still helps. What is still missing is recent form, handled properly: ratings that drift over time, at a rate the model estimates (v2 item 1).

**Adopted:** Track B v1.1, 1-year window, no half-life. Ties the best baseline, best calibrated model tested, no tuning knob. The rolling-xG blend is kept as a **benchmark, not a production feature**: it would give P(win) with no honest interval, ignore the starting goalie in half the average, and disagree with the simulator's P(win). If time-varying ratings work in v2, blending should stop helping; that is v2's test.

**How Phase 7 got here:**
1. Backtest v1 (1- and 2-year windows) against the baselines: v1 lost to rolling xG; 2-year was too confident. → One-year window; xG into Track B.
2. Built v1.1 (xG + goals): tied rolling xG, calibrated. Still behind in 2025-26. → Tested recency (half-lives 180, 90, 60 days).
3. Recency tied 2025-26 but hurt 2024-25 and calibration. → Kept v1.1 1-yr; time-varying ratings to v2.

**Not done in Phase 7:**
- **Market comparison.** Notebook section 10 is ready; it needs closing moneylines in `data/external/closing_lines.csv` (columns `date, home, away, home_ml, away_ml`). Remove the margin by dividing each implied probability by their sum (about 1.04–1.05), then score like the models. A model that doesn't beat the closing line loses money once the margin is taken out, so this is information, not betting advice.
- Per-subset comparisons, the cost of a wrong default starter, the held-out home-win share and a forward test: all listed with the market comparison in *v2.0b Phase 7 follow-ups*.

**Switching production to v1.1.** `fit_hierarchical.py` now fits the same model and window as the backtest: default `--model xg --window 365`, games from `end − 365 days` up to `end` (default: the day after the last game in the database), saved as `<label>_<YYYYMMDD>.nc` (e.g. `trackbxg_w365_20260417.nc`). After a successful save it writes `output/posteriors/active_fit.json` (fit file, model, window dates, seasons, teams). `late_game.py`, `score_state.py` and `predict_game.py` read that file (`src/models/active_fit.py`), so all three use the same fit and the same games; `predict_game.py` warns if `late_game.json` or `score_state.json` were measured on a different fit. `--model goals --window 730` still fits v1 for comparison. The old `hier_20242025_20252026.nc` is left in place for the Phase 5/6 notebooks.

**Settings kept:** weekly refits, 5,000 score-state sims per game in the backtest, seed 42 (+ refit number), 1,000 posterior draws per prediction.

**Notes:**
- **Missing-xG check.** `fit_hierarchical.py` stops if any loaded regular-season game in the window (or after it, with the default end) has no rows in `shot_xg`, and prints the `score_xg --season` command to run. Without it, unscored games drop out of the game table and the fit quietly uses older data and goalie ratings. `--skip-xg-check` overrides.
- The latest `backtest.py` returns early ("All games already predicted; nothing to fit.") when a results file is complete, instead of redoing the pre-season fit.
- The notebook discovers model files by label, so any new backtest (new window or half-life) appears without code changes.
- Calibration slope: logistic regression of the outcome on logit(p). For a model that predicts the same value every game (home rate) it is undefined.

### Phase 8 — PCE for Sobol (Week 9+) — DONE (analysis A)

**Question (analysis A):** across the league, what makes one game's prediction different from another's? Team offense, team defense, the starting goalies, or the league-wide terms? Analysis B (which posterior uncertainties drive one matchup's 90% interval) is in v2 (see *v2.0c*).

**Files:**
- `src/models/stochastic/pce.py` — port of the soil driver's PCE code: `theta_to_xi`, `xi_to_theta`, `total_order_indices`, `legendre_eval`, `build_design_matrix`, `lhs_samples`, `fit_pce`, `pce_predict`, `loocv_q2`, `holdout_test`, `sobol_from_pce`, `saltelli_sobol`, `build_pce_surrogate`, `plot_pce_validation`, `plot_sobol`. Added: `sobol_groups`, `ishigami` / `ishigami_exact` (known-answer test), a vectorized option in `build_pce_surrogate`, a faster `build_design_matrix` (same result), and a recursive `total_order_indices` (same set and order; the soil version loops over $(P+1)^d$ ≈ 10 million combinations for $d = 10$).
- `src/analysis/sensitivity.py` — `INPUT_NAMES`, `GROUPS`, `input_specs(draws, gt, late)`, `game_qoi(theta, late)` (exact game model for arrays of inputs), `truth_predictor`, `sd_contributions`
- `src/pipelines/build_pce.py` — `py -m src.pipelines.build_pce [--degree 4] [--n-train N] [--saltelli 16384] [--overwrite]` → `output/models/pce_<fit stem>.pkl`, `_summary.json`, `_validation.png`, `_sobol.png` (about 40 s)
- `notebooks/phase8_check.ipynb` — eleven sections (below); the degree scan takes a few minutes

**Truth model:** the exact game model (`outcome_probs`), about 60,000 games per second, so the PCE design and the Saltelli check both run on the real model. PCE is not needed for speed here; it earns its keep by giving Sobol' indices analytically from its coefficients (see *Why PCE Earns Its Keep*).

**Inputs (10), each uniform with the real mean and sd** (range = mean ± √3 sd, Legendre basis):

| Input | Varies over | Source |
|---|---|---|
| `off_home`, `off_away` | the 32 teams | sd of the teams' posterior-mean offense ratings (active fit) |
| `def_home`, `def_away` | the 32 teams | same, defense |
| `goalie_home`, `goalie_away` | starters in the fit window | sd of the starters' $\gamma$ |
| `alpha0`, `alpha_home`, `rho` | posterior uncertainty (same in every game) | posterior mean and sd |
| `ot_factor` | its uncertainty | `late_game.json` 90% range ÷ 3.29 |

Sobol' indices are shares of variance, so how much each input varies is the main modeling choice; variance matching keeps every input at its real spread. Section 10 repeats the analysis with full observed ranges (worst to best team, 1st–99th percentile goalie) to check that the conclusion doesn't depend on that choice.

**QoIs:** P(home win), P(tied after regulation), expected regulation goals (both teams).

**PCE settings:** total degree $P = 4$ (1,001 terms), 3 training points per term (3,003 Latin hypercube points), 2,000 hold-out points, seed 42. Grouped indices: team ratings / starting goalies / league terms.

**Checks (`phase8_check.ipynb`):**

| Section | Check |
|---|---|
| 2. Known answer | Ishigami function: PCE ($P = 10$) and Saltelli reproduce the exact indices, including $S_3 = 0$, $S_{T,3} = 0.244$ (test run: PCE within 0.0003) |
| 4. Truth model | Vectorized `game_qoi` = `outcome_probs` game by game; average matchup favors home a little; directions right (home offense up → P up; away goalie up → P down) |
| 5. Degree | Hold-out error for $P = 2..5$; indices stable from $P = 3$ |
| 6. Accuracy | Saved PCE: hold-out RMSE < 1% of the output sd; built on the active fit |
| 7. Main result | Sobol' $S_i$, $S_{T,i}$ per QoI; PCE vs Saltelli on the exact model (16,384 × 12 runs) within 0.01 |
| 8. Groups | Team ratings vs goalies vs league terms |
| 9. Units | Main-effect curves: P(home win) as each input moves across its range |
| 10. Robustness | Largest group unchanged and group shares move < 0.10 with full-range inputs |

**Pipeline test (fake database, 180 games; not real results):** PCE hold-out R² ≥ 0.99999 for all three QoIs at $P = 4$ (RMSE 1.7e-4 on P(home win)); indices identical to 3 decimals from $P = 3$ to $5$; PCE and Saltelli agree within 0.0005; first-order sums 0.93–0.995 (little interaction). (The fake fit's shares are not meaningful; real results below.)

**Results (active v1.1 fit, 365-day window; all notebook checks PASS).** PCE and Saltelli on the exact model agree within 0.0003 on every index.

| Input | P(home win) $S_i$ | sd alone (prob.) | P(tied after reg.) $S_i$ / $S_{T,i}$ | Reg. goals $S_i$ |
|---|---|---|---|---|
| `off_home` | 0.248 | 0.032 | 0.122 / 0.154 | 0.221 |
| `def_away` | 0.238 | 0.031 | 0.118 / 0.148 | 0.213 |
| `off_away` | 0.234 | 0.031 | 0.040 / 0.070 | 0.199 |
| `def_home` | 0.225 | 0.030 | 0.038 / 0.068 | 0.192 |
| `alpha_home` | 0.031 | 0.011 | 0.015 / 0.020 | 0.027 |
| `goalie_away` | 0.012 | 0.007 | 0.006 / 0.008 | 0.011 |
| `goalie_home` | 0.011 | 0.007 | 0.002 / 0.004 | 0.010 |
| `alpha0` | 0.000 | 0.000 | 0.044 / 0.044 | 0.125 |
| `rho` | 0.000 | 0.000 | **0.549** / 0.550 | 0.000 |
| `ot_factor` | 0.000 | 0.000 | 0.000 | 0.000 |
| **Output mean / sd** | 0.524 / 0.064 | | 0.259 / 0.016 | 5.64 / 0.44 |
| Sum of first-order | 0.999 | | 0.934 | 0.997 |

Grouped (first order; total in brackets where different):

| Group | P(home win) | P(tied after reg.) | Reg. goals |
|---|---|---|---|
| Team ratings | **0.946** | 0.375 (0.384) | **0.826** |
| Starting goalies | 0.023 | 0.008 (0.011) | 0.020 |
| League terms | 0.031 | **0.608** (0.614) | 0.152 |

Robustness (full observed ranges instead of variance-matched): P(home win) team ratings 0.942, goalies 0.035, league terms 0.023; ties 0.458 / 0.014 / 0.514; goals 0.851 / 0.032 / 0.116. Largest group unchanged everywhere; largest move 0.095 (ties).

**What Phase 8 found:**
1. **Team ratings decide who wins: 95% of the variation in P(home win) between games.** P(home win) has sd 0.064 across matchups (typical games 46–59%, a narrow league). Offense and defense matter equally (0.48 vs 0.46); each team input alone moves P(win) by about ±3 points (1 sd). Inputs pair as the model's structure says: home offense ≈ away defense (both set the home scoring rate), away offense ≈ home defense.
2. **Almost no interactions** (first-order sum 0.999): P(win) is close to additive in the ratings. Each input's effect barely depends on the others.
3. **Starting goalies: 2%** (about ±0.7 points each). Small because the Phase 3 goalie ratings vary little between starters (strongly shrunk, k = 800; see the input table in notebook section 3) next to team ratings, which matches Phase 3's weak year-to-year goalie signal. This is how much goalies matter *in the model*, not necessarily in hockey: if the rating is over-shrunk, real goalie effects are larger. The v2.0b wrong-starter test and the v2.1 goalie effect are the checks.
4. **Of the league terms, only home ice matters for the winner** (3%, about ±1.1 points from its posterior uncertainty). Scoring level, tie inflation and the OT factor have essentially zero effect on P(win). Pinning down home ice (more seasons, season-varying home ice in v2.3) is the only league-term improvement that would move P(win).
5. **Ties are driven by tie-inflation uncertainty** ($\rho$: 55%), then the home side's scoring rate. The league-term indices here measure *our uncertainty* about one shared number, so a sharper $\rho$ (more data, or the score-state simulator's tie mechanism in the exact model) would tighten OT and standings predictions; it won't change who wins. Ties are the only output with real interactions (7%): a tie depends on how *balanced* the two scoring rates are, which involves both teams together.
6. **Totals:** team ratings 83%, league scoring level 12%. Season-to-season drift in scoring ($\alpha_0$) is the next lever after ratings, consistent with Phase 7 (totals limited by scoring drift).

**What this means for v2:** improving team ratings is where nearly all the leverage on P(win) is, which supports v2.0 (time-varying ratings) as the first model item. Goalie and home-ice work can at most move P(win) a few percent of its variance unless the goalie ratings are too shrunk (test before building v2.1's goalie effect).

**Run order:** rebuild order (`fit_hierarchical` → `late_game` → `score_state`), then `py -m src.pipelines.build_pce`, then the notebook. Rebuild after each new fit; the indices shift slightly as ratings change.

**Expected from the model's structure:** P(win) depends on the inputs only through the two scoring rates, $\lambda_h$ (home offense − away defense − away goalie + home ice) and $\lambda_a$. So offense and defense shares split by how much each varies across teams, goalies get a share set by their (small) spread, and $\alpha_0$, $\rho$ and the OT factor should barely matter for P(win) but dominate ties and totals. Because the model only sees offense minus the opponent's defense, "offense vs defense" credit depends on how ratings vary across teams, not on a separate effect.

## v2 — Planned Improvements

Everything in this section comes **after Phase 7**. The baseline each item must beat is now **Track B v2.0 + back-to-backs (v2.0d)** (held-out log loss 0.6732, slope 1.02); before that v2.0 (0.6749, slope 0.98) and v1.1 (0.6773, slope 0.99), tied with rolling xG (0.6754). Each item is kept only if it improves held-out log loss in the Phase 7 walk-forward backtest (`backtest_season.py` + `phase7_check.ipynb`).

| Item | What it adds | Main payoff |
|---|---|---|
| **v2.0 Time-varying team ratings — DONE (live)** | Ratings and league levels that drift over time at an estimated rate, on top of v1.1 | Beat v1.1 clearly, tied/edged rolling xG, calibrated; see *v2.0* below |
| **v2.1 Hierarchical xG layer** | Shooter, goalie, rink and season × shot-type effects on top of LightGBM xG | Finishing talent, a shrunk xG-based goalie rating, fixes for arena bias and redirect drift |
| **v2.2 Lineup model** | Per-skater offensive/defensive effects (Bayesian RAPM), aggregated over tonight's lineup | Predictions that react to injuries, scratches and trades |
| **v2.0c Phase 8 analysis B** | Sensitivity of one matchup's P(win) interval to each posterior uncertainty (correlated inputs: Shapley effects) | Explains what makes a prediction uncertain; points to which ratings need more data |
| **v2.0b Phase 7 follow-ups — DONE (backtest parts)** | Market comparison, subset results, wrong-starter cost, home-win share, forward test on 2026-27 | Found the back-to-back gap; market and forward test still running |
| **v2.0d Back-to-backs — DONE (live)** | Rest effects on shot quality for a team on the second night of a back-to-back; ARI → UTA as one franchise | Clear gain over v2.0; see *v2.0d* below |
| **v2.3 Other candidates** | In-game win probability, travel effects, rest on finishing/goaltending, team- and season-specific home ice, better OT/SO, etc. (xG in Track B and score-state simulation are done) | Smaller calibration gains |

### v2.0c — Phase 8 Analysis B: What Drives One Matchup's Uncertainty

Phase 8 did analysis A (what makes games differ across the league). Analysis B asks, for one game: which posterior uncertainties make its 90% P(win) interval wide (e.g. "COL–DAL's interval comes mostly from DAL's defense rating")? Inputs are that game's posterior draws (team ratings, home ice, $\rho$; goalie uncertainty once v2.1 makes it a draw instead of a fixed value). Those inputs are **correlated** (offense with defense, $\alpha_0$ with the ratings, $r = -0.81$), and Sobol' indices assume independent inputs, so the options are: decorrelate first (Rosenblatt / Nataf transform, or a PCE on whitened inputs; indices then belong to the transformed inputs), or Shapley effects, which share variance fairly among correlated inputs. Start with Shapley effects on the exact model (cheap); report next to the posterior-correlation matrix.

### v2.0b — Phase 7 Follow-Ups (validation, not model changes)

Left open when Phase 7 closed. None changes the model; each tells v2 where to aim. All use the Phase 7 backtest files (`output/backtests/*_scored.csv`) and `phase7_check.ipynb` unless noted.

1. **Market comparison (most important). Forward version RUNNING (started 2026-10-31):** `src/data/odds.py` + `src/pipelines/log_odds.py` (`py -m src.pipelines.log_odds [--markets h2h,totals,spreads]`) pull current NHL odds from The Odds API (free plan; key in `.env` as `ODDS_API_KEY`; 1 credit per moneyline snapshot) and append them to `output/forward/odds_<season>.csv` (one row per game × book × snapshot: moneylines, margin, no-vig P(home)). `score_forward` scores the market as the median no-vig P(home) across books from the **last snapshot taken before each game's start time** (exact: the API gives start times), next to the model: log loss, paired model-minus-market difference with a 90% range, and results on games where they differ by 5+ points. Tested with simulated API responses (matching, margin removal, the before-start rule, scoring). The historical (backtest) version below still needs past closing lines.
   Historical version: The only test against the real benchmark: v1.1 ties rolling xG, but its gap to the betting market is unknown. Needs historical closing moneylines for 2024-25 and 2025-26 saved as `data/external/closing_lines.csv` (columns `date, home, away, home_ml, away_ml`; team abbreviations as in the database). Notebook section 10 then removes the margin (divide each implied probability by their sum, about 1.04–1.05) and scores the market like the models: log loss, Brier, calibration, paired difference vs v1.1. Realistic target: behind the market; each v2 model item is then measured by how much of that gap it closes. Also compare by subset to see where model and market disagree.
2. **Results by subset.** Log loss, calibration and paired differences vs rolling xG for: the first month of the season (ratings carried over the summer), back-to-backs and rest days, games started by backup goalies, and games after trades. Shows which v2 item would help most (v2.0 for early season and trades, rest covariates for back-to-backs, v2.1 goalie effect for backups).
3. **Cost of a wrong default starter.** The backtest used actual starters. Live predictions guess the starter (most starts in the last 10 games) unless `--home-goalie` / `--away-goalie` are given. Rerun the backtest predictions with the default starter instead of the actual one and compare log loss; the difference is what a wrong guess costs, and whether waiting for the confirmed starter is worth it.
4. **Home-win share on held-out games.** In-sample the model gave 52.9–53.4% vs 54.2% actual. Check mean predicted vs actual home-win share per season in the backtest; if it is low out of sample too, look at the home-ice prior or season-varying home ice (v2.3).
5. **Forward test on 2026-27 — RUNNING (started 2026-10-24).** Nothing was tuned on these games, so it is the cleanest check of the live setup (weekly refits, guessed starters, real-time data).
   - `src/pipelines/predict_day.py`: `py -m src.pipelines.predict_day [--date D] [--days N] [--goalie TEAM=NAME|ID ...] [--list-goalies TEAM ...] [--replace]` predicts every regular-season game on the date(s) from the `schedule` table and appends to `output/forward/predictions_<season>.csv` (P(win) with 90% range, regulation/OT split, rates, fair odds, totals, puck lines, total-goals distribution, starters and whether they were given or guessed, historical home rate as a baseline, fit file, timestamp). It refuses games already loaded as played and dates inside the active fit's window. Run it in the morning: the schedule has dates, not puck-drop times.
   - `src/pipelines/score_forward.py`: `py -m src.pipelines.score_forward [--season S]` scores, per game, the last prediction made on or before the game date: log loss, Brier, accuracy, calibration slope (90% ranges) vs the historical home rate and the backtest of the live model (0.6732 for v2.0 + rest; was 0.6773 for v1.1); calibration table; by week; score outputs; and **how often the default starter was right** (with log loss on wrong-guess games), which also covers item 3 going forward. Writes `output/forward/scored_<season>.csv`.
   - Expect a wide range for months: about ±0.03 in log loss after 300 games; differences of 0.005 need 1,000+ games.
   - Test on the fake database (15 games): skips, `--replace` with given starters, leak guards and scoring all worked.
   - **Game cards.** `predict_day` draws a pregame card for every game it logs, from the same numbers it logs (no second model run, so card and log always match); `--no-cards` turns it off, and a card that fails never stops the logging. Postgame cards: `py -m src.pipelines.make_cards postgame --pending` draws one for every logged game that is now loaded and scored (skips cards already drawn; `--redo` redraws), `--date D` for every loaded game on a date (logged or not), or give game ids. The postgame card shows the pregame P(win) the forward test scores (last prediction made on or before game day). Its "deserve to win" meter replays the game's shots 10,000 times, each scoring with its xG (`src/analysis/deserved.py`; tied replays go to the team with the larger xG share, the Phase 6 OT rule). `scripts/demo_cards.py` draws both cards on example data; `scripts/inspect_for_cards.py` prints the tables and signatures the cards use.

### v2.0 — Time-Varying Team Ratings — DONE (live model since 2026-11-01)

**Why:** Phase 7 showed v1.1 missed recent form (rolling xG tied it, and blending helped), and a recency half-life was the wrong fix (it forgets every team at the same rate and made predictions too cautious, slopes 1.5–1.6). Phase 8 showed team ratings drive 95% of P(win), so better ratings are the biggest lever.

**Files:**
- `src/models/stochastic/hierarchical_tv.py` — `time_grid`, `build_model_tv(gt, teams, step_days=7)` → `(model, grid)`, `sample_tv` (stores only the variables needed; `target_accept` 0.95), `add_current_ratings(idata, grid, end)`, `game_rates`, `add_paths`, `compact_paths`, `simulate_league_tv`
- `fit_hierarchical.py` — `--model tv` (now the default), `--step-days`, `--draws`, `--tune`; 730-day window for tv
- `backtest.py` / `backtest_season.py` — `model="tv"`, `--step-days`; labels `trackbtv_w730`, `trackbtv_w730_s2`
- `late_game.py` (`mean_rates`), `montecarlo.py` (`calibrate_scale`), `game_model.py` (`load_track_b`) — for a v2.0 fit, past games use the ratings **as of their own date**
- `notebooks/phase7_check.ipynb` — v2.0 labels and section 11, the v2.0 verdict

**Model.** v1.1 (xG Gamma + tie-inflated Poisson goals, finishing effects, fixed goalie offsets, home ice, tie inflation), with:

$$\mu^{xG}_{\text{off},t,k} = \mu^{xG}_{\text{off},t,k-1} + s_k\,\varepsilon_{t,k}, \qquad \textstyle\sum_t \varepsilon_{t,k} = 0, \qquad s_k = \begin{cases}\tau_{\text{off}}\sqrt{\text{gap}_k/7} & \text{in season}\\ \tau_{\text{off,summer}} & \text{across an off-season (gap > 45 days)}\end{cases}$$

the same for defense, and both league levels ($\beta_0$: xG per game; $\delta_0$: goals per xG) as their own random walks. Steps are bins of `step_days` days that contain games. The zero-sum team steps send league-wide changes to $\beta_0$ / $\delta_0$ instead of moving all 32 ratings together. Priors (half-normal sd): team $\tau$ 0.02/week, team summer 0.10, league 0.01/week, league summer 0.05. Non-centered steps. For prediction, `add_current_ratings` steps every draw from the last step to the window end (an off-season step for a pre-season fit) and saves the result under the v1 names, so everything downstream is unchanged.

**Synthetic recovery** (12 teams, 2 seasons): drifting teams + league → $\tau$ recovered (0.014–0.017 vs 0.015), 90% path coverage 91% / 89%; nothing drifting → $\tau$ shrinks to ~0.005 and ratings stay static; only the league drifting → drift goes to the league level, team $\tau$ near 0. No divergences.

**Real fit** (730 days to 2026-04-17, 2,634 games): weekly team drift $\tau$ ≈ 0.012 (offense and defense) — over a season about ±0.06, as large as the spread between teams ($\sigma$ ≈ 0.05–0.06); summer steps 0.019 / 0.040 (wide: one off-season in the window); league xG level drifts 0.0135/week; conversion steady in season (0.005) but moves over the summer (0.043), matching Phase 3's season offsets. Home ice 0.072, $\rho$ 0.52 (same as v1.1). About 6 min per fit (1,000 draws); the lowest-ESS parameters are the summer steps (318–377); team paths ≥ 575. Live fits use `--draws 2000`.

**Per-game rates (a needed fix):** late_game (OT factor) and score_state (simulator scale) compare real games with each game's expected rate. With time-varying ratings that rate must come from the ratings on the game's date: on the test data, per-game rates averaged 2.695 home goals vs 2.706 actual, while today's ratings applied to every past game gave 2.595.

**Walk-forward backtest** (2,624 held-out games, same protocol as Phase 7):

| Model | 2024-25 | 2025-26 | Combined | Calibration slope |
|---|---|---|---|---|
| **v2.0, weekly steps (adopted)** | **0.6652** | 0.6845 | **0.6749** | **0.98** [0.80, 1.15] |
| v2.0, 2-day steps | — | — | ≈0.6750 | 0.98 [0.80, 1.14] |
| v1.1 (1-yr) | 0.6670 | 0.6877 | 0.6773 | 0.99 |
| v1.1, half-life 90 d | 0.6683 | **0.6841** | 0.6762 | 1.50 |
| Rolling xG | 0.6669 | 0.6838 | 0.6754 | 0.83 |

| Test | Result |
|---|---|
| v2.0 vs v1.1 | **−0.0025 [−0.0040, −0.0009]: clear win** (first change since v1.1 outside luck) |
| v2.0 vs rolling xG | −0.0005 [−0.0038, +0.0027]: ahead, within luck (v1.1 trailed by 0.0019) |
| Blend (v2.0 + rolling xG) minus v2.0 | −0.0011 [−0.0026, +0.0006]: no longer clearly helps (was −0.0023 to −0.0031 with v1.1) |
| Weekly minus 2-day steps | −0.0001 [−0.0002, +0.0001]: identical → weekly kept |

**What v2.0 found:**
1. **Teams change within a season as much as they differ from each other,** and modelling that change at an estimated rate is what v1.1 was missing. It gains in both seasons: best of all in the steadier 2024-25, and in 2025-26 it matches the half-life versions while staying calibrated.
2. **Step length doesn't matter** (weekly = 2-day). The remaining small blend gain (0.0011, within luck) is not about reacting a few days sooner.
3. In 2025-26 alone the slope is 0.82 [0.52, 1.07] (slightly over-confident); rolling xG shows the same there (0.73), so it looks like a hard season rather than a v2.0 problem.

**Open items:**
- **ARI / UTA — fixed in v2.0d:** the game table now maps ARI → UTA (one franchise), so Utah inherits Arizona's history instead of starting from scratch.
- **Summer steps are loosely identified** with one off-season in the window; a 3-season window gives two (test in the backtest).
- **Fit size:** about 50–70 MB per saved fit (paths stored as float32); delete old weekly fits now and then.
- **League scoring level for totals:** now drifting; recheck the Phase 7 score outputs (totals) with v2.0.

### v2.0b — Follow-Up Results (2026-11-01)

Run in `notebooks/v2_followups.ipynb` on the v2.0 walk-forward backtest files.

- **Early season:** no worse than the rest of the season; carrying ratings over the summer works.
- **Favorites:** calibrated; no sign of over- or under-confidence at the extremes.
- **Wrong default starter:** the guess (most starts in the last 10 games) is wrong in 38–45% of team-games, but costs only about +0.0001 log loss, because goalie ratings differ little (Phase 8: goalies 2% of P(win) variance). Waiting for confirmed starters is optional.
- **Home-win share:** 2024-25 actual 56.2% vs 53.3% predicted.
- **Back-to-backs (the real gap):** with the **away team on the second night**, home won 62.0% (± 3.9) vs 53.4% predicted (bias −0.086). With the **home team** on a back-to-back the model was too high (+0.053). This explains much of the low home share above and led to v2.0d.

### v2.0d — Back-to-Back (Rest) Effects — DONE (live model since 2026-11-02)

**Files:**
- `src/processing/game_table.py` — `team_rest(conn)` (days since each team's previous game, from the schedule, so it works for future games), `rest_on_date(conn, team, date)`; the game table gains `rest_home`, `rest_away`, `b2b_home`, `b2b_away`. `FRANCHISE = {"ARI": "UTA"}` merges Arizona into Utah after the goalie merges.
- `src/models/stochastic/hierarchical_tv.py` — `build_model_tv(..., rest=True, home_drift=False)`; `rest_shift(draws, b2b_home, b2b_away)`; `game_rates` applies rest (and the home path when drifting).
- `fit_hierarchical.py` — `--rest` is **on by default for `--model tv`** (`--no-rest` turns it off); `--home-drift` optional. Labels `trackbtv_w730_r` (`_h`, `_rh` for home drift).
- `backtest.py` / `backtest_season.py` — `--rest`, `--home-drift`; backtest outputs keep the rest and back-to-back columns.
- `predict_day.py` / `make_cards.py` — look up rest for each team on the game date, apply `rest_shift`, log `rest_*` / `b2b_*` / `gamma_*_eff`, and print a `[TEAM back-to-back]` tag.
- `phase7_check.ipynb` section 11 (each variant vs plain v2.0) and `v2_followups.ipynb` (bias by subset per variant).

**Model.** v2.0 plus four effects on the log xG rate (each Normal(0, 0.1)): a team on the second night of a back-to-back creates less xG (`rest_off_home`, `rest_off_away`) and allows more (`rest_def_home`, `rest_def_away`). Home and away get separate effects because the bias differed by side. Prediction uses posterior means of the effects as a shift on the log rates. **Home drift** (tested): home ice as a random walk that steps only across summers, $\tau_{\text{home,summer}} \sim$ HalfNormal(0.05).

**Walk-forward backtest** (2,624 held-out games, ARI → UTA merged):

| Model | Combined log loss | vs plain v2.0 | Calibration slope |
|---|---|---|---|
| **v2.0 + rest (adopted)** | **0.6732** | **−0.0016 [−0.0026, −0.0006], clear** | **1.02** |
| v2.0 + rest + home drift | 0.6732 | −0.0016 [−0.0027, −0.0006] | 1.02 |
| v2.0 (weekly) | 0.6749 | — | 0.98 |
| v2.0, 2-day steps | 0.6750 | +0.0001 | 0.98 |

| Test | Result |
|---|---|
| v2.0 + rest vs v1.1 | −0.0041: clear |
| v2.0 + rest vs rolling xG | −0.0021: ahead, within luck |
| Blend (v2.0 + rest + rolling xG) minus v2.0 + rest | 0.0004 [−0.0021, +0.0013]: no gain left (was 0.0011 for v2.0, ~0.0027 for v1.1) |
| Home drift on top of rest | identical: not adopted |

**Did it fix the bias?**

| Subset | Games | Actual home win | v2.0 | v2.0 + rest | Bias before → after |
|---|---|---|---|---|---|
| Away team on back-to-back | 416 | 62.0% ± 3.9 | 53.4% | 56.4% | −0.086 → **−0.056** |
| Home team on back-to-back | 152 | 48.7% ± 6.7 | 54.0% | 50.5% | +0.053 → **+0.018** |
| Neither | 1,926 | | | | +0.010 → +0.006 |

Home back-to-backs are essentially fixed; away back-to-backs about a third fixed.

### v2.0e — Rest on Finishing and Goaltending — TESTED, NOT ADOPTED (2026-11-02)

**Idea:** a tired team may also finish worse, and its goalie (often the backup) lets in more of the same shots; v2.0d only lets rest change xG. **Model:** four more effects on goals per xG (`rest_fin_home/away`, `rest_gk_home/away`, Normal(0, 0.1)); `--rest-goals` in `fit_hierarchical` / `backtest_season`, label `_rg`. Synthetic recovery: effects recovered when present (wider ranges than the xG terms; goals are noisier), all include 0 when absent.

**Result:** vs the live model (v2.0 + rest) −0.0004 [−0.0009, +0.0001], **within luck → not adopted** (kept in the code; retest when 2026-27 adds a season). It moved the bias the most of anything tried: away back-to-back −0.056 → −0.042, home back-to-back +0.018 → −0.001. Calibration 1.01.

### v2.0f — Travel, Time Zones, 3 Games in 4 Nights — TESTED, NOT ADOPTED (2026-11-03)

**Idea:** the road team on a back-to-back usually flew in overnight. **Files:** `src/processing/schedule_context.py` (new): arena locations (`ARENAS`, city-level, standard UTC offsets), per game and side `rest`, `b2b`, `km` (since the last game), `trav` (= b2b × km / 1000), `tz` (time zones crossed, rest ≤ 2 days), `dense` (3rd game in 4 nights); neutral-site games found from `games.venue` (travel counts 0). All from the schedule, so known before puck drop. The game table, backtest outputs and the predictions log carry these columns. **Model:** six effects on log xG rate shared by home and road (`trav_off/def`, `tz_off/def`, `dense_off/def`); `--travel`, label `_rt`. `context_shift(draws, context)` replaces `rest_shift` for all fits. Synthetic recovery passed (18 of 20 terms in their 90% ranges across both cases; b2b, travel and 3-in-4 overlap, so credit is split imperfectly, the total is well determined).

**Result:** vs the live model +0.0000 [−0.0003, +0.0004] → **not adopted.** By trip length the away back-to-back bias is the same for short (−0.048, under 500 km) and medium (−0.059, 500–1,200 km) trips; crossing 2+ time zones shows no gap (−0.027); 3-in-4 small (−0.025). **The away back-to-back gap is not travel.** Since v2.0e (which includes goaltending) helped most, the next check is whether it sits in games the backup started (`v2_followups.ipynb`, "Back-to-backs × backup goalies"): if so, goalie ratings are too shrunk and the fix is the v2.1 goalie layer.

**Also found:** `predict_day` silently dropped new log columns (it reindexed to the file's old columns), so since v2.0d the rest columns were not logged (the logged probabilities were right). Fixed: new columns are added to the file. **`scripts/check_backtests.py`** checks every backtest file (one row per game, full coverage, no duplicates) and the game table; both notebooks now drop duplicated games and leave out models without a prediction for every game, with a message. Run it before the notebooks.


### v2.0g — Goalie on Zero Rest — CHECKED, NOT BACKTESTED (2026-11-03)

**Why:** splitting away back-to-backs by goalie, the bias sat in games the **usual starter** played (140 games, −0.121 vs the live model) while backup starts were fine (276 games, −0.024; rested backups +0.027). So it looked like goalie fatigue, not under-rated backups. **Built:** `goalie_rest(conn)` / `goalie_rest_on_date` in `game_table.py` (days since the starter's previous start; `g2b` = he also started yesterday), game-table columns `grest_*`, `g2b_*`, model term `gk_b2b` (opponent's goals per xG, Normal(0, 0.15)), `--goalie-rest`, label `_rk`; `predict_day` tags such games. Synthetic recovery passed.

**Check on the exact feature (no backtest needed):** goalies almost never start both nights: **40** away games in two seasons, bias **+0.065** (wrong sign, noise ±0.13). Away back-to-backs with a fresh goalie: 491 games, −0.070. The "usual starter" games are mostly the starter returning rested after the backup played the first night. **Not goalie fatigue → not backtested;** the code stays, off by default.

**Back-to-back chapter closed.** Five ideas tested (rest, home drift, rest on finishing, travel, goalie rest); only rest (v2.0d) was adopted. The remaining away back-to-back miss (−0.056 on 416 of 2,624 games) is worth at most about 0.001 log loss even if fixed perfectly (bias² / 2p(1−p) × share of games), and ~15 subsets were examined, so part of it may be chance. Further gains must come from items that touch every game (v2.1, v2.2).

### v2.1 — Hierarchical xG Layer — BUILT, NOT ADOPTED FOR GAMES (2026-11-06)

**Files:**
- `src/models/stochastic/xg_layer.py` — `load_shots`, `fit_laplace` (production fitter), `build_xg_layer` (PyMC, for checking), `adjusted_logit`, `effect_draws`, `simulate_goals`.
- `src/pipelines/fit_xg_layer.py` — `--holdout` (walk-forward shot test, per-term ablation, goals vs expected by shot type and rink), `--final` (fit all, top/bottom lists, predictive check, save `output/posteriors/xg_layer_<season>[_bg]*`), `--terms` (subset, refit without the others).
- `src/pipelines/build_xg_asof.py` — weekly as-of refits on earlier shots only → `shot_xg_layer` (shooter-adjusted xG, `xg_base` for staleness), `goalie_layer_asof` (goalie gamma on Track B's scale), `xg_layer_fits`. Incremental: rebuilds the latest week, new weeks, and weeks with missing or rescored shots.
- Game model options: `build_game_table(..., xg_layer, goalie_layer)`; `--xg-layer` / `--goalie-layer` in `fit_hierarchical` and `backtest_season` (labels `_lx`, `_lg`, `_lxg`); `active_fit.json` carries them so `late_game`, `score_state`, `predict_day` follow; `fit_hierarchical` stops if the as-of tables are behind.

**Fitting: Laplace, not NUTS.** Plain NUTS on 581k Bernoulli rows did not finish 200 draws in 10 minutes (many hours for a full fit). The layer is fit like a mixed-effects logistic model (as `lme4::glmer`): Newton steps on a sparse design for the effects, sigmas at the mode of the Laplace marginal likelihood times their half-normal priors, effects Gaussian N(mode, H⁻¹). 9–30 s on the full data. **Checked against NUTS** on a 60k-shot subsample: effect means correlate 0.99–1.00, sds within 5–7% (Laplace slightly wider). **Synthetic recovery** (560k shots, realistic volumes): shooter sigma 0.154 vs 0.151 true, rink 0.053 vs 0.047, season × type 0.079 vs 0.076, goalie 0.100 vs 0.084; 90–100% of true effects inside their 90% ranges; all sigmas near 0 when there are no effects.

**Shot-level results** (`--holdout`, fit on earlier seasons, score the next; per-shot log loss, negative = better):

| Test season (training seasons) | All four terms | Shooter + goalie | Shooter | Goalie | Rink | Season × type |
|---|---|---|---|---|---|---|
| 2023-24 (2) | +0.00018 | −0.00004 | −0.00023 | +0.00014 | +0.00007 | +0.00024 |
| 2024-25 (3) | −0.00032 | **−0.00030** | −0.00033 | −0.00001 | 0.00000 | +0.00001 |
| 2025-26 (4) | −0.00023 | **−0.00025** | −0.00024 | −0.00007 | +0.00004 | −0.00002 |

(The per-term columns for shooter and goalie come from the shooter + goalie fit; rink and season × type from the four-term fit.)

- **Shooter finishing** helps every season. **Goalie skill** needs history: hurts with 2 seasons, neutral with 3, helps with 4. **Rink** recording bias is negligible once LightGBM is in (sigma 0.026, no rink beyond about ±3%, within chance): the Open Question on arena bias is answered. **Season × shot type**: the redirect drift is real (+0.18, +0.11, +0.03, −0.01, −0.22, −0.10 by season) but each season goes its own way, so carrying it forward hurts; both dropped.
- Full fit (shooter + goalie, 581k shots): sigma_beta 0.153 [0.140, 0.167], sigma_gamma 0.073 [0.060, 0.089]. Top finishers Draisaitl (+0.44; 241 goals on 157 xG), Thompson, Robertson, Makar, Kaprizov; top goalies Shesterkin, Sorokin, Hellebuyck, Vasilevskiy, Ullmark. Predictive check: 98% of shooters and 96% of goalies inside their 90% intervals (slightly wide, the cautious side).

**Game-level results** (walk-forward backtest, 2,624 games, vs the live model v2.0 + rest):

| Variant | vs live | Calibration slope |
|---|---|---|
| + layer goalie (`_r_lg`) | −0.0003 [−0.0016, +0.0011] | 0.96 |
| + layer goalie + shooter-adjusted xG (`_r_lxg`) | −0.0004 [−0.0019, +0.0012] | 0.94 |
| live model | — | 1.02 |

**Not adopted:** within luck, with ranges three times as wide as the rest variants (a new goalie rating moves many games both ways), and calibration slips below 1: the layer's goalie spread is wider than Phase 3's, so game predictions get more extreme than results justify. Phase 3's heavy shrinkage is about right for games even though the layer predicts shots better. The shooter signal at game level is mostly already in Track B's team finishing term. **Kept for v2.2:** shooter finishing pays off when the model knows who dresses tonight; the as-of tables are built for it. No weekly `build_xg_asof` needed until then.

#### Original plan

**Why a layer, not a fully Bayesian xG model.** LightGBM already pins down the shot-geometry function: with 562k shots, uncertainty in the distance/angle/shot-type curve is negligible next to game randomness and team-strength uncertainty. A fully Bayesian xG model would be slow on 562k shots and would need its nonlinear interactions written by hand, likely losing AUC. What LightGBM *can't* do is tell shooters, goalies and rinks apart, or let shot-type conversion drift by season. A small hierarchical model on top of it adds exactly that.

**Model.** LightGBM's prediction enters as a fixed offset; only the random effects are estimated:

$$
\begin{aligned}
\text{logit}\, p_i &= \text{logit}(xG^{\text{lgbm}}_i) + \beta_{s[i]} - \gamma_{g[i]} + \delta_{r[i]} + \varepsilon_{k[i]} \\
\beta_s &\sim \mathcal{N}(0, \sigma_\beta^2) \quad \text{shooter finishing talent} \\
\gamma_g &\sim \mathcal{N}(0, \sigma_\gamma^2) \quad \text{goalie skill (positive = better, same sign as Track B)} \\
\delta_r &\sim \mathcal{N}(0, \sigma_\delta^2) \quad \text{rink / arena recording effect} \\
\varepsilon_k &\sim \mathcal{N}(0, \sigma_\varepsilon^2) \quad \text{season} \times \text{shot-type group} \\
y_i &\sim \text{Bernoulli}(p_i)
\end{aligned}
$$

Half-normal hyperpriors on the four $\sigma$'s. Partial pooling shrinks low-volume shooters and goalies toward zero; high-volume ones keep their own estimate.

**What each term addresses:**
- $\beta$ — finishing talent. Feeds player point distributions later.
- $\gamma$ — a shrunk, xG-adjusted goalie rating. Replaces the Phase 3 conjugate rating if it backtests better.
- $\delta$ — the arena recording bias from Open Questions. A logit intercept per rink is an approximation: the real bias is in recorded *location*, so a distance-adjustment version is a later refinement. Home teams play 41 games in their own rink, so $\delta$ is partly confounded with home shooters and goalies; fitting all terms jointly is what separates them.
- $\varepsilon$ — the redirect location drift. Lets conversion by shot type move season to season instead of relying only on retraining.

**Inputs — the one hard rule.** The offset must come from the walk-forward `shot_xg` table (Phase 3), never from in-sample LightGBM predictions. In-sample xG is overfit, its residuals look too small, and the random effects would be shrunk toward zero.

**Fitting.**
- NumPyro NUTS (or PyMC with the NumPyro backend). Plain PyMC on 562k Bernoulli rows is slow.
- Non-centered parameterization for every random effect from the start.
- Parameter count is a few thousand (shooters dominate) — fast once the offset does the heavy lifting.

**Files:**
- `src/models/stochastic/xg_layer.py` — model definition (renamed from the planned xg_hierarchical.py, too close to hierarchical_xg.py, Track B v1.1).
- `src/pipelines/fit_xg_layer.py` — load `shot_xg` + IDs, fit, save `output/posteriors/xg_layer_<season>.nc`.
- Adjusted per-shot xG (posterior mean and sd) written to a `shot_xg_v2` table; `shot_xg` stays untouched.

**Validation — keep the layer only if it earns it:**
1. **Synthetic recovery.** Simulate shots on top of real offsets with known $\beta, \gamma, \delta, \varepsilon$; check recovery, same discipline as the soil project.
2. **Held-out shot log loss.** Fit on seasons up to $t-1$, score season $t$; must beat the LightGBM offset alone.
3. **Posterior predictive checks.** Goals per shooter, per goalie and per rink against their predictive intervals.
4. **Sanity of the spreads.** $\sigma_\beta$ and $\sigma_\gamma$ should be small (finishing and goaltending talent are real but modest on the logit scale).
5. **Game-level payoff.** Swap $\gamma$ into the game model in place of the Phase 3 goalie rating and rerun the Phase 7 backtest. Keep whichever gives better game log loss.

**Goalie handoff (resolved).** Track B has no estimated goalie term; it takes goalie offsets as fixed inputs from Phase 3. To use the v2.1 goalie effect, swap it in as that input: one $\gamma$ posterior draw per Track B posterior draw, which also carries the goalie uncertainty Track B v1 ignores. Goalie skill still enters only once.

### v2.2 — Lineup (Player-Level) Model — DONE (live model since 2026-11-08)

**Data fixes first.**
- `scripts/check_shifts.py` — per season: games with shift charts, skater shift time vs box-score TOI, dressed-skater match, on-ice skater counts, goalie gaps; the worst games. It found 2025-26 missing 505 of 1,312 shift charts.
- Cause: `load_all` swallowed shift-request failures (`except: shift_count = 0`) and never retried, because the game counted as loaded. Fixed in `scripts/load_all.py`: failures are logged; `shift_fetch_log` records every try; every run retries games without a chart (failures each run, empty answers after `EMPTY_RETRY_DAYS` = 1, since an empty answer can be a fluke); `--shifts-only` runs just that; progress every 10 games.
- `src/data/nhl_api.py`: every request has a 30 s timeout (a stalled request used to hang forever, silently), and `call()` retries timeouts and connection errors; shift rows with no player are skipped instead of failing the whole game; goal markers (typeCode 505) are dropped.
- Result: all 527 retried games filled. Only 57 games (2024-25) genuinely have no chart in the NHL feed, under 1% of games, skipped. A few games store every shift twice; the stint builder de-duplicates.
- Windows note: several hung `load_all` runs at once kept the database locked; `Stop-Process -Name python -Force` clears them.

**Stints** (`src/processing/stints.py`, `src/pipelines/build_stints.py`; tables `stints`, `stint_games`). Stretches with no line change: skaters and goalie per side, strength, score at start, faceoff zone start (home point of view), shots, xG and goals per side. Conventions: a shot or goal at second t goes to the stint with s < t ≤ e (the scorer whose shift ends at the goal gets it); a faceoff at t sets the zone of the stint starting at t; duplicate shift rows and goal markers dropped. A game is skipped if TOI is off by more than 3%, the skater count is outside 3–6 for more than 1% of the time, there's no goalie for more than 6% of the time, or shots fall outside any stint. Checked on a hand-worked game (exact match). Real build: 6,669 games ok, 60 skipped (under 1%), 2.5M stints, 2.2M at 5v5; 48.6 min of 5v5 and 9.2 min of special teams per game, 5v5 xG 27,501 vs goals 27,288.

**Ridge RAPM reference** (`src/models/deterministic/rapm.py`, `src/pipelines/fit_rapm.py`). 5v5 stints with both goalies, two rows per stint (each team on offense), target xG/60 weighted by minutes; player offense and defense columns; controls for home, score state, zone start; players under 100 min share "other" columns. Ridge from the normal equations (matches scikit-learn's `Ridge` to 3e-7); lambda by 5-fold cross-validation grouped by game. Real fit (2 seasons, 891k stints): lambda 1,000 (interior), players cut held-out error 0.18% vs controls only (stint xG is mostly noise). Controls are textbook: home +0.13, trailing +0.12 / +0.21, leading −0.09 / −0.14, offensive-zone faceoff +0.67, defensive-zone −0.90, call-ups' offense −0.35. MacKinnon tops offense; Seider, Pelech, Tanev and Slavin top defense; system players (Spence, Hagel) high, as usual for RAPM.

**Bayesian RAPM** (`src/models/stochastic/lineup_rapm.py`, `src/pipelines/fit_lineup_rapm.py`). The same regression with spreads for forward/defense × offense/defense learned by maximizing the exact marginal likelihood (a Gaussian likelihood with Gaussian priors gives an exact Gaussian posterior: no MCMC). Synthetic recovery: all four spreads recovered, 91–92% interval coverage. Real fit: spreads F off 0.206, D off 0.177, F def 0.184, D def 0.172 (ridge's lambda implied one spread of 0.132, so ridge over-shrinks); agreement with ridge 0.99; held-out error the same as ridge. Team-season "system" effects (`--teams`) can't be separated from the roster (posterior sds 75–85% of their spread, no held-out gain), so they're dropped.

**Into the game model** (`src/processing/lineup_context.py`, `src/pipelines/build_lineup_asof.py`; tables `player_rating_asof`, `lineup_fits`, `lineup_context`).
- The Bayesian RAPM is refit every 14 days on the 730 days of stints before that date.
- Each game's dressed skaters are weighted by expected TOI (average of their previous 10 games), and the lineup's strength is compared with the team's usual lineup (previous 20 games, rated with the same ratings). Unrated players get the "other" rating.
- `lin` = (change in offense − opponent's change in defense) × 5v5 share of xG / league 5v5 xG/60, entered in Track B as `kappa_lineup × lin` on the log xG rate (κ ~ N(1, 0.5), learned). `--lineup` in `fit_hierarchical` / `backtest_season`, label `_r_lu`.
- Tests: a planted scratch of a team's top forward matched the hand-computed change exactly; κ recovered (0.96 for a true 1, −0.05 for a true 0); no leakage.
- The weakest real lineups are the last week of 2025-26 (teams resting regulars, 3–5 call-ups) and opening weeks. Typical change ±3.6% xG, the weakest −20%.
- Limit: linemates who always play together can't be separated, so the shift is strongest for players the model can isolate.

**Backtest** (`scripts/compare_vs_live.py`: variants vs the live model on shared games, per season and pooled). With ratings rebuilt from 2022-10 (a first test had lineup shifts only from 2024-10, so most training games had none) and 2023-24 added:

| vs live (v2.0 + rest) | 2023-24 | 2024-25 | 2025-26 | Pooled (3,936 games) |
|---|---|---|---|---|
| + lineup (`_r_lu`) | −0.0011 | −0.0017 | +0.0002 | **−0.0009 [−0.0020, +0.0002]** |
| + rest on finishing (`_rg`) | +0.0002 | −0.0008 | −0.0000 | −0.0002 [−0.0007, +0.0003] |

**Decision: adopted (2026-11-08), a deliberate exception to the "whole range below 0" rule.** Within luck by that rule, but about 94% likely better, and it grew (−0.0006 → −0.0009) rather than faded with more data; the downside if it is luck is at most about +0.0002. The backtest uses the actual dressed lineup (the best case); live lineups come from the feeds below. Rest on finishing shrank toward 0 with the third season (−0.0004 → −0.0002): closed. Recheck with 2026-27 complete (checklist below); if it is then clearly worse, go back to `--no-lineup`.

**Live lineups** (`src/data/lineup_sources.py`, `src/processing/live_lineup.py`, in `predict_day` whenever the active fit has the lineup term). Sources, checked with `scripts/probe_lineup_sources.py`:
- **ESPN injuries** (unofficial public feed, no key): all 32 teams, statuses Injured Reserve / Out / Day-To-Day, names and positions. Abbreviations match the NHL's except LA, NJ, SJ, TB.
- **NHL game-day roster** (`gamecenter/<id>/play-by-play` → `rosterSpots`): on game day about 23 per team, the active roster *without* injured-reserve players but with healthy scratches and Out / day-to-day players.

Assumed lineup per team:
1. If the NHL roster already lists 18 or fewer skaters, that is the dressed lineup; use it ("posted").
2. Otherwise, start from the last dressed lineup; drop players not on today's roster or listed Out; keep day-to-day players, noted (`--dtd-out` drops them); bring back regulars who are on the roster, healthy and dressed for 10+ of the last 20 games.
3. Apply `--out TEAM=NAME` / `--in TEAM=NAME`.
4. Balance to 12 F / 6 D: drop the least-used extras, fill from the roster by usage, or with an unrated call-up (the fit's "other player" rating).

The shift uses the same calculation as the backtest (expected TOI from the previous 10 games, vs the team's previous 20 lineups, the latest rating fit). Each game prints the lineup source, notes and the xG shift; the log keeps `lineup_source_*`, `lineup_notes_*`, `lineup_players_*`, `d_off_*`, `lin_*`, `lineup_fit`. Tested on each case (Out player, off the roster, returning regular, posted lineup, both feeds down, day-to-day, overrides) and end to end. `fit_hierarchical` turns `--lineup` on by default for v2.0 and stops if stints or ratings are behind; weekly: `build_stints`, `build_lineup_asof` before the fit (DAILY_RUN.md).

**Recheck checklist (spring 2027):**
1. `py -m src.pipelines.build_stints`
2. `py -m src.pipelines.build_lineup_asof --rebuild --start 2022-10-01`
3. Backtest 2026-27 with `--rest` and with `--rest --lineup`; rerun the earlier `--rest --lineup` seasons with `--restart` so they use the newest ratings.
4. `py .\scripts\check_backtests.py`, then `py .\scripts\compare_vs_live.py --variants trackbtv_w730_r_lu` (about 5,200 games).
5. Pooled range entirely below 0: confirmed. Clearly above 0: switch back with `--no-lineup`. Otherwise keep it. The forward test (`score_forward`) adds live evidence against the market.

#### Original plan

**Idea.** The v1 game model rates *teams*. It can't tell when a top player is injured, scratched or traded. The lineup model rates each *skater's* effect on xG for and against, then builds tonight's team strength from tonight's dressed lineup. It is essentially Bayesian RAPM, fit with the same hierarchical/NUTS machinery as Track B.

**Data unit: the stint.** A stint is a stretch of time during which the set of players on the ice doesn't change; any line change by either team ends it. `stints.py` builds them by overlaying the 12 on-ice players' intervals from `shifts` and cutting at every change. Each stint records duration, the 10 skaters and 2 goalies, strength, score state, zone start, and shots and summed xG for each side. Expect a few hundred stints per game, on the order of a couple million over five seasons.

**Model (sketch).** For stint $j$ with duration $t_j$:

$$
\text{rate}_{\text{home},j} = \alpha_0 + \alpha_{\text{home}} + \sum_{p \in \text{home skaters}} o_p - \sum_{q \in \text{away skaters}} d_q + \text{(score, zone terms)}
$$

and the mirror image for the away side. Each skater gets an offensive effect $o_p$ (raises his team's xG rate) and a defensive effect $d_p$ (suppresses the opponent's). Partial pooling learns the prior widths instead of tuning a ridge penalty, and the joint posterior shows when linemates can't be told apart (their effects come out negatively correlated).

**From ratings to a game prediction.**
1. Take tonight's dressed lineup (known about an hour before puck drop).
2. Project each player's ice time from **past games only**.
3. Team xGF/60 and xGA/60 ≈ ice-time-weighted sums of the dressed players' $o_p$ and $d_p$.
4. Convert xG to goals with finishing and goaltending effects (tonight's starting goalie; later the v2.1 shooter $\beta$ and goalie $\gamma$).
5. Feed $\lambda_{\text{home}}, \lambda_{\text{away}}$ into the same Phase 6 game model and Monte Carlo. Nothing downstream changes.

**Choices and tentative defaults.** None of these need deciding until Phase 7 is done; the defaults are starting points, not decisions.

*Before starting*

| # | Choice | Options | Tentative default |
|-|----------|------------------|------------------|
| 1 | Order within v2 | Lineup model first, xG layer first, or both | xG layer first — smaller, and its $\beta, \gamma$ feed step 4 |
| 2 | Ridge RAPM reference | Yes / no | Yes — fit `deterministic/rapm.py` with scikit-learn first and check the Bayesian means against it |

*Model design*

| # | Choice | Options | Tentative default |
|-|----------|------------------|------------------|
| 3 | Likelihood | Weighted Gaussian on xG/60; Tweedie on summed xG; two-part (shot counts + shot quality) | Weighted Gaussian (variance ∝ 1/duration); upgrade later |
| 4 | Scale of player effects | Linear on the xG rate, or log scale (effects multiply) | Linear — comparable to public RAPM |
| 5 | Game situations | 5v5 only, or separate PP/PK player effects | 5v5 only; special teams stay team-level |
| 6 | Stint covariates | Score state, zone start, home ice, back-to-backs | Score state, zone start, home ice |
| 7 | Prior structure | One shared $\sigma$; separate forward/defense $\sigma$; prior centered on last season | Separate F/D $\sigma$; no carryover in the first version |
| 8 | Data window | Rolling 2–3 seasons, or all 5 with recency weights | Rolling 2 seasons |
| 9 | Goalies in the stint model | In or out | Out — goalies affect whether shots go in, not shot quality; they enter at step 4 |
| 10 | Fitting method | NumPyro NUTS, or variational / Laplace approximation | NumPyro NUTS, sparse design matrix; approximate only if too slow |

*Making a prediction*

| # | Choice | Options | Tentative default |
|-|----------|------------------|------------------|
| 11 | Ice-time projection | Last-$N$-games average share, or line-based deployment | Last 10 games, rescaled so the dressed lineup sums to a full game |
| 12 | Lineup source in the backtest | Actual dressed lineup vs. projected lineup | Actual dressed lineup (known pregame) with **projected** TOI — never that game's real TOI |
| 13 | xG → goals | League-average finishing + Phase 3 goalie rating, or v2.1 $\beta, \gamma$ | League average + Phase 3 goalie first; swap in v2.1 later |

*Evaluation*

| # | Choice | Options | Tentative default |
|-|----------|------------------|------------------|
| 14 | What counts as winning | Better overall game log loss, or better on games with major lineup changes | Score both. Fix the "major change" threshold in advance (e.g. a top-6 F or top-4 D out) |

**Validation.** Same Phase 7 walk-forward harness as the team model: game log loss and Brier, overall and on the lineup-change subset. Plus synthetic recovery on simulated stints with known player effects. If the lineup model only wins on lineup-change games, consider a cheaper hybrid: the team model adjusted for known missing players.

**Files.** `src/processing/stints.py`, `src/models/stochastic/lineup_rapm.py`, `src/pipelines/fit_lineup.py`, posteriors to `output/posteriors/lineup_<window>.nc`.

### v2.4 — Playoff Odds — DONE (2026-11-08)

**Files:** `src/models/stochastic/season_sim.py` (`current_standings`, `simulate_season`, `qualify`, `summarize`), `src/pipelines/playoff_odds.py` (`py -m src.pipelines.playoff_odds [--sims 20000] [--date …] [--no-drift] [--no-check]`).

**Method:**
- **One posterior draw per simulated season,** used for every remaining game, so a team better than its average rating is better all season.
- **Weekly rating drift:** each team's offense and defense random-walk at the fit's own τ_off / τ_def, zero-sum.
- **Exact game outcomes:** regulation, OT and shootout for either side, from `outcome_probs`. Rest and back-to-back shifts from the schedule; each team's goalie offset is its goalies' Phase 3 γ weighted by the share of the last 20 starts; lineups at full strength.
- **NHL rules:** win 2, OT/SO loss 1; ties broken by points, RW, ROW, wins, then a coin flip (head-to-head and goal differential not simulated). Top 3 per division plus 2 wild cards per conference.
- **Standings** come from the database and are checked against the NHL standings feed.

**Tests** (synthetic 32-team league, balanced 82-game schedule):
- equal teams 43–56% to make the playoffs (about 50%) and 10–16% to win the division (12.5%);
- a +0.45 team 99.9%, a −0.45 team 0.1%;
- exactly 16 playoff teams and 4 division winners in every season;
- 2.23 points per game; about ±15 points of luck over a season;
- drift widens the ranges; a team 30 points clear with few games left is 100%.

**Output:** printed per division, saved as `output/playoffs/playoff_odds_<date>.csv`, and appended to `playoff_odds_history_<season>.csv` for charts. First real run (2026-11-08, 276 games played): the standings matched the NHL's for all 32 teams.

**Open:**
- Stanley Cup / round-by-round odds (bracket simulation, best-of-7, playoff OT without shootouts).
- A historical check: run as of past dates with fits ending then, and score P(playoffs) against what happened.

### v2.5 — Visuals for v2 — DONE (2026-11-08)
Order: (1) pregame card upgrade, (2) playoff odds chart, (3) team card, (4) player card.

**(1) Pregame card — DONE (2026-11-08).** `src/viz/pregame_card.py`, `src/viz/card_style.py`,
`src/pipelines/make_cards.py`. A new band under the header shows, per team, what the model knew:
- playoff odds from the latest `output/playoffs/playoff_odds_<date>.csv` on or before the game
  date (points, games played, projected points with 90% range; the run date if older);
- schedule chips: days off / back-to-back (with km overnight) / 3rd game in 4 nights / goalie
  also started last night (tiring spots in orange);
- lineup: source (NHL posted or assumed), xG change vs the usual lineup (`expm1(lin)`), and the
  live_lineup notes grouped (Out, Back, Fills in, Scratched, Day-to-day playing).
Old log rows without these columns are drawn as the v1 card. The rest of the card is unchanged
(the 90% win-probability range was already there). No change to `predict_day`; redraw any
logged game with `py -m src.pipelines.make_cards pregame <game_id>`.

**(2) Playoff odds charts — DONE (2026-11-08).** `src/viz/playoff_chart.py`,
`src/pipelines/make_playoff_charts.py`; `playoff_odds` draws them at the end (`--no-charts` to skip).
- `playoff_board_<date>.png`: four division panels; per team one bar = P(playoffs), split by how
  it gets in (division win in the team colour, 2nd/3rd lighter, wild card lightest), points now
  and projected points (90% range).
- `playoff_trend_<season>.png`: P(playoffs) per team by run date, one panel per division,
  labelled at the line ends. Grows with each weekly run.
Team colours: a team whose colour is too close to a division rival's switches to its next team
colour; if none works its line is dashed. Labels always name the team.

**(3) Team card — DONE (2026-11-08).** `src/viz/team_card.py`, `src/pipelines/make_team_card.py`
(`py -m src.pipelines.make_team_card BOS` or `--all`; output/teams/<fit date>/<TEAM>.png). Per team:
- league rank overall / offense / defense, with 90% ranges (rank in each posterior draw);
- weekly offense and defense paths over the fit window (xg_mu_*_path + phi_*, as % better than an
  average team) with 90% bands, against the spread of all teams; off-seasons squeezed to 3 weeks.
  The week dates are rebuilt from the fit's games (time_grid); the card stops if they no longer
  match the fit (database changed: refit first);
- inside the ratings now: shot quality for / against (xG part) and finishing for / against (phi);
- goalies: share of the last 20 starts and Phase 3 rating (1 − theta, 90% range). The defense
  rating excludes the starter, who enters the model separately;
- playoff odds through the season and the latest projection (from the playoff_odds history).

**(3b) Rankings card — DONE (2026-11-08).** `src/viz/rankings_card.py`; `make_team_card --rankings`
(also made by `--all`; output/teams/<fit date>/rankings.png). All teams on one card: position,
record, net rating (% better than average, 90% range), rank range, offense and defense positions,
playoff odds and projected points. Fix: the team cards used the median rank across posterior
draws, which can give two teams the same rank and skip another (no "2nd"). Positions now come
from one league ordering by the posterior mean rating (`league_table`), shared by both cards, so
every position is used once; the "likely" range is still the 5th-95th percentile of the rank.

**(4) Player card — DONE (2026-11-08).** `src/viz/player_card.py`, `src/pipelines/make_player_card.py`
(`py -m src.pipelines.make_player_card McDavid`, ids, `--team EDM`, or `--all` for every rated skater who
played this season, about 3-6 min; output/players/<rating date>/, one folder per team with --team / --all).
Percentile bars among regulars at the player's position (forwards / defensemen with 300+ 5v5
minutes in the 2-season window): dot = where the mean rating ranks, whisker = where the ends of
its 90% range rank.
- Offense / defense / net: v2.2 `player_rating_asof` (newest fit date). Net sd assumes offense
  and defense independent (no covariance stored).
- Finishing: v2.1 xG layer shooter effect (`xg_layer_<season>_beta.csv` from `fit_xg_layer
  --final`, all four terms, all loaded seasons), among shooters with 150+ shots; shows goals vs
  xG. Now refreshed weekly (DAILY_RUN). The card notes it if the fit is more than 10 days behind
  the scored shots. v2.1 is still not part of the game model; this is display only.
- Usage: games and TOI this season, 5v5 minutes in the window; net impact over time (every
  rating fit) against the middle 80% of regulars at the position.
Not included: linemate quality (would need a separate stint summary).

**(5) Goalie card — DONE (2026-11-08).** `src/pipelines/make_goalie_card.py` (draws with the player
card's layout; `player_card.py` gained trend title / % axis / card-kind options). Names, ids,
`--team BOS` or `--all` (every goalie who started this season); output/goalies/<date>/.
Percentile bars among regular goalies (1,500+ shots faced in the 2-season window):
- Goals kept out: Phase 3 goalie rating (1 − theta, 90% range), the game model's goalie input;
- Shooter-adjusted: v2.1 xG layer goalie effect (1 − exp(−gamma)), display only;
- This season only: 1 − GA / xGA (league-adjusted), no shrinkage, Poisson range; 200+ shots.
Tiles: starts and share of the team's starts, save % (shots on goal, periods 1-4), goals saved
above expected this season, goals per game vs an average goalie at the window's average
workload. Trend: Phase 3 rating weekly (in-season weeks) against the middle 80% of regulars.

### v2.6 — Closing Lines — DONE (2026-11-08)
For the market comparison, the benchmark is the closing line (the last price before puck drop),
the sharpest the market gets. `src/pipelines/log_closing.py` gets the day's start times from
The Odds API events list (free), groups games starting within 20 minutes, sleeps, and takes one
moneyline snapshot 10 minutes before each group (1 credit each, 3-5 a night), appending to
`output/forward/odds_<season>.csv` like `log_odds`. It re-reads start times before each snapshot,
reports games missed while the computer slept, and stops below 20 credits. Windows Task Scheduler
setup in DAILY_RUN.md. `consensus()` now returns `minutes_before` (start minus snapshot);
`score_forward` prints the median and the model vs market difference on closing lines only
(snapshots 30 minutes or less before the start). Historical closing lines (past seasons) would need
the paid historical odds endpoint, so the comparison builds up from now on.

### v2.7 — Stanley Cup Odds — DONE (2026-11-08)
`season_sim.simulate_playoffs` plays each simulated season's bracket (called by `playoff_odds`).
- Bracket (NHL since 2014): the better division winner plays the 2nd wild card, the other the 1st;
  2nd vs 3rd in each division; round 2 inside each half; conference final; Cup Final.
- Home ice: higher seed in rounds 1-2 (division winner, 2nd, 3rd, wild card), better record after.
- Games: best of 7, 2-2-1-1-1; exact series probability by enumeration (checked against brute
  force). A game tied after regulation goes to sudden-death overtime, won by the team that scores
  first (its share of the combined scoring rate); no shootouts.
- Ratings: each team's end-of-season rating in that simulated season (same posterior draw plus
  the season's drift), drifting 2 weeks per round. Goalie offsets: the regular-season starter mix.
- Checks on a 32-team test league (4,000 seasons): 16 / 8 / 4 / 2 / 1 teams per round in every
  season, 2nd and 3rd of a division never both advance, P(Cup) sums to 1; the strongest teams come
  out at 12-14%, about where real favourites sit. A league that cannot form the bracket is skipped.
Output: columns p_round2, p_conf_final, p_final, p_cup in playoff_odds_<date>.csv (and history),
`cup_odds_<date>.png` (make_playoff_charts), a Cup column on the rankings card, Cup odds on the team
card's playoff tile. Not modelled: playoff-specific home ice or scoring (regular-season values used),
re-seeding (the NHL does not re-seed), injuries during the playoffs.

### v2.8 — Playoff Odds Backtest — DONE (2026-11-09); drift x2 adopted
`src/pipelines/backtest_playoff_odds.py`: were past seasons' playoff and Cup odds, computed only from
what was known on each date, calibrated? Default: the 3 latest complete seasons x Nov 15, Dec 15,
Jan 15, Feb 15, Mar 15 (15 checkpoints, about 1-2 hours; resumes).
- Each checkpoint: the live model (v2.0 + rest + lineup) fit on the 730 days before the date
  (`backtest.fit_window`), OT / shootout parameters from those games, standings as of that morning
  (`current_standings(..., before=)`), goalie mix and rest shifts as of the date, 5,000 simulated
  seasons + playoffs. Baseline "standings only": same simulation, every team (and goalie) equal.
- Truth: playoff teams and Cup winner from the NHL playoff-bracket feed (falls back to final
  standings, no Cup); division winners and final points from the database.
- Report: Brier / log loss of P(playoffs) by checkpoint vs the baseline (paired, 90% range over
  team-seasons); calibration by bin; final points inside the 90% projected range (target 90%);
  division winners; the champion's probability on each date (sanity check only: 3 champions).
What it tests: the rating uncertainty and the weekly drift (tau) over months, which the
single-game backtest cannot see. If points ranges are too narrow or wide, tau is the lever.

**First results** (2023-24 to 2025-26, 15 checkpoints, drift x1.0):
- Model beats standings-only at every date (log loss 0.496 vs 0.564 on Nov 15 ... 0.216 vs 0.232 on
  Mar 15); pooled -0.0296 [-0.0799, +0.0299]: very likely helps, not proven on 96 team-seasons.
- Too confident, mostly early: final points inside the 90% range 85% in Nov-Dec, 87.5% overall (misses
  lean low: 7.5% below, 5.0% above); teams under 5% made it 6.9% of the time (predicted 1.0%);
  40-60% bin made it 38%, 60-80% bin 58%. Division winners: model slightly worse than standings-only.
- Cup sanity: the champion's mean log score -2.26 (standings-only -2.97, all equal -3.47).
**Fix under test:** `season_sim.TAU_SCALE` multiplies the fit's weekly drift in season simulations only
(game predictions unchanged); `playoff_odds --tau-scale` overrides it. The backtest now saves each
checkpoint's fit and inputs (output/backtests/playoff_cache/) and `--tau-scales 1 1.25 1.5 2`
re-simulates them at each scale without refitting. Pick the scale with the lowest log loss, preferring
coverage near 90% when log losses are within about 0.003; then set TAU_SCALE.

**Sweep result (2026-11-09)**, 15 checkpoints x 5,000 seasons per scale:

| drift x | log loss | Brier | points inside 90% | inside Nov-Dec | <5%: predicted / made | champion log score |
|---|---|---|---|---|---|---|
| 1.00 | 0.3560 | 0.1122 | 87.5% | 85.4% | 1.0% / 6.9% | -2.257 |
| 1.25 | 0.3557 | 0.1123 | 87.5% | 85.4% | 1.0% / 7.0% | -2.283 |
| 1.50 | 0.3558 | 0.1123 | 88.1% | 87.0% | 1.0% / 7.0% | -2.297 |
| 2.00 | 0.3553 | 0.1123 | 89.0% | 88.5% | 1.0% / 6.0% | -2.323 |

Decision: **TAU_SCALE = 2.0** (lowest log loss, though all within luck; coverage closest to 90%).
Drift barely changes accuracy; it makes the points ranges honest. Game predictions are unaffected.
Long shots: the under-5% teams that made it were 3 team-seasons (5 rows): MTL 2024-25 (0.6-1.6% at
three dates; a second-half run to the last wild card at 91 pts), STL 2024-25 (3.9% in Feb; a run after
a coaching change, 96 pts), WSH 2023-24 (3.8% in Feb, 91 pts). About one was expected. Mild evidence
at most; drift does not change it (their gap is points already lost). **Recheck after 2026-27** with
four seasons: if under-5% teams keep making it, look at how the ratings treat bad starts.

### v2.9 — Why This Number (Shapley breakdown) — DONE (2026-11-09)
`src/models/explain.py` (`win_prob_breakdown`), called by `predict_day` for every game; logged as
`shap_base`, `shap_total`, `shap_home_ice`, `shap_ratings`, `shap_goalies`, `shap_rest`, `shap_lineup`.
- The scoring rates are a sum of log-scale pieces, each 0 when neutral: home ice (alpha_home), skater
  ratings (off/def of both teams), goalies (the starters' Phase 3 ratings), rest & travel (every
  schedule-context term), lineups (kappa x lin). Switching a factor off = its pieces to 0 for both teams.
- Base = equal teams, average goalies, neutral ice (about 50%; the rest is the late-game model's own
  OT / shootout home edge). Shapley value = average change in P(home win) when the factor is switched
  on, over all 120 orders (32 combinations, exact per-draw probabilities, all posterior draws).
- The five values add up exactly to P(home win) - base; checked to 6 decimals against the logged
  p_home. Signs checked on a planted game (away back-to-back, tired goalie, weaker lineup -> home).
- Pregame card: a "Why BOS 56.1%" strip under the context band, from the favourite's side: one tile
  per factor (points of win chance and which team it helps, with a bar on a shared scale) and the sum
  line "even game 50.3% +2.1 -1.8 ... = 56.1%". Rows logged before this have no strip.
This was v2.0c analysis B.

### v2.3 — Other Candidate Improvements

Smaller items already listed under *Open Questions*, gathered here so v2 has one list:
- **Score effects, travel** as game-model covariates. **Back-to-backs: done in v2.0d** (shot quality); rest on finishing/goaltending is the open follow-up.
- **OT/SO model** weighted by goalie quality instead of a flat 50/50.
- **Goal dependence:** done in v1 for ties (Track B tie inflation, $\rho = 0.51$). Dixon-Coles or a bivariate model remain alternatives if Phase 7 shows margin-specific miscalibration.
- **Score-state simulation: built in Phase 6** for pre-game score outputs. Natural extensions: **in-game win probability** (start the simulator from the current score, time and goalie state), period-specific score effects, and simulating the 11% of empty-net goals from delayed penalties and early pulls.
- **xG in Track B: done in Phase 7** (Track B v1.1, xG as a second observed quantity sharing the team effects).
- **Arena bias, location version:** adjust recorded shot distance per rink, refining the v2.1 rink intercept.
- **Rebound and rush windows:** test 2/3/4 s by retraining xG and comparing validation log loss.
- **Player-level point distributions:** built on the v2.2 player effects plus the v2.1 shooter finishing effects.
- **Team-specific home ice:** $\alpha_{\text{home}} + \eta_t$, with $\eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$ partially pooled toward the league value (same idea as the goalie shrinkage). Candidates are rinks with real reasons to differ (altitude in Colorado, travel to Winnipeg or Vancouver). Published work generally finds most team-to-team spread in home advantage is noise, so expect $\sigma_\eta$ to come out small and most teams to land near the league value.
- **Season-varying home ice** (tested in v2.0d as a summer-only random walk: no gain on top of rest, not adopted): $\alpha_{\text{home},s}$ per season, either partially pooled or a slow random walk, because home advantage has drifted down over time across sports. League value on 2021-22 to 2025-26: $\eta = 0.0715$ (about 7.4% more scoring at home, uncertainty about ±0.010).
- Both home-ice items add parameters (32 for teams, 1 per season), so each must improve held-out log loss in the Phase 7 backtest to be kept.
- **Time-varying team ratings:** moved up to **v2.0** after Phase 7.

---

## Key Conceptual Notes

### Cheap vs Honest Uncertainty Propagation

Both versions produce joint vector samples. The difference is *where the correlations live.*

**Independent marginal sampling (cheap, fragmented).** Each component fit independently, sampled independently. Joint distribution is the product of marginals → diagonal covariance. Misses correlations between parameters that exist in the data (e.g., team offense and defense are correlated; team strength and goalie skill are negatively correlated for bad-team-good-goalie scenarios).

**Joint hierarchical posterior (honest).** All parameters fit jointly in a single hierarchical model. Sample the whole vector at once → full covariance, possibly non-Gaussian. This is what Track B produces.

Cheap version typically miscalibrates the tails. Won't catastrophically break win probabilities but will produce wrong probabilities for extreme outcomes.

### Why PCE Earns Its Keep for Hockey

For the soil project, PCE was essential because the truth model was ~1 s/eval. For the hockey team-level game model, evaluation is in microseconds — direct MCMC on the truth model is viable, surrogate not strictly needed for inference.

PCE earns its keep in hockey for **sensitivity analysis**: analytical Sobol indices directly from PCE coefficients via the variance-decomposition identity

$$\text{Var}[\hat Y] = \sum_{|\alpha| > 0} c_\alpha^2 \mathbb{E}[\Psi_\alpha^2]$$

This gives a clean decomposition of outcome variance into contributions from each input (team xGF/60, team xGA/60, goalie GSAx, etc.) — without running Saltelli on millions of samples.

If you later upgrade to a possession-by-possession simulator that's slow to run, PCE returns as an inference-speed tool.

### Sources of Uncertainty (Ranked by Sobol Later)

Three categories injected at different layers:

1. **Aleatoric (irreducible).** Game randomness — Poisson sampling in the game model. Even with perfect knowledge of $\lambda$, the puck still bounces.
2. **Parameter uncertainty.** Posterior width on team/goalie/player parameters from finite-sample inference.
3. **Model misspecification.** The Poisson form, the independence assumption between home/away goals, the missing factors (rest, travel, injuries). Hardest to quantify; partially addressed by tie inflation (Phase 5), posterior predictive checks and held-out validation.

Sobol analysis at the end ranks which of these matter most for which outcomes.

---

## Persistence & Reproducibility

- **xG model:** pickle under `output/models/`, named with `MODEL_VERSION` (`xg_lgbm_v1_*`), plus a `_metrics.json` with version, seed, settings, data cutoff and library versions. `XG_MODEL_PATH` in `config/settings.py` points to the active model.
- **Per-shot xG:** the `shot_xg` table, rebuilt by `score_xg.py`; each row records the model version and which model scored it.
- **Posterior samples:** ArviZ NetCDF under `output/posteriors/`. Both tracks write to the same format so downstream code is sampler-agnostic. Track A writes `toy_<A>_<B>.nc` (posterior + observed games, settings in `posterior.attrs`) plus `_meta.json` (seed, sampler settings, game ids, diagnostics, library versions) and trace/ACF plots. Like `.pkl` files, these stay out of git; rebuild with `fit_toy.py`. Track B writes `<label>_<YYYYMMDD>.nc` (e.g. `trackbxg_w365_20260417.nc`; label = model, window, half-life; date = window end; seasons, window dates and team list in `posterior.attrs`), `_meta.json` (priors, window, sampler settings, diagnostics, versions), `_ratings.csv` and `_trace.png`; rebuild with `fit_hierarchical.py`. `active_fit.json` names the fit the live pipeline uses. (Phase 5–6 fits are `hier_<first>_<last>.nc`.)
- **Late-game and score-state parameters:** `output/models/late_game.json` and `output/models/score_state.json`, each recording the fit file they were measured against; rebuild with `late_game.py` / `score_state.py` after every new fit.
- **Game cards:** `output/game_summaries/<date>/<away>_at_<home>_<pregame|postgame>.png` (pregame from `predict_day`, postgame from `make_cards`).
- **Forward test:** `output/forward/predictions_<season>.csv` (append-only log, one row per prediction, timestamped) and `scored_<season>.csv`.
- **Predictions:** `output/predictions/<date>_<away>_at_<home>.json` (`predict_game.py --save`): inputs, goalies used, outcome probabilities with intervals, likely scores, totals and puck lines.
- **PCE surrogate:** `output/models/pce_<fit stem>.pkl` (coefficients, multi-indices, input specs, Sobol' indices, hold-out set) and `_summary.json` (input ranges and sources, accuracy, single and grouped indices, Saltelli check); rebuild with `build_pce.py` after each new fit.
- **Backtest results:** `output/backtests/<label>_<season>.csv` (one row per held-out game: teams, goalies, P(win) with 90% range, regulation/OT probabilities, rates, score outputs, refit date, training games, divergences) and `_scored.csv` (plus final score, how it ended, `home_won`); baselines in `baselines_<season>.csv` + `_info.json` (tuned Elo settings). Labels name the model, window and half-life (`trackbxg_w365`).

Reproducibility checklist for any fit:
- Random seed logged
- Data cutoff date logged
- Model version logged
- Software versions (PyMC, NumPy, LightGBM) logged in metadata

---

## Open Questions Deferred to "When We Get There"

These are real decisions but premature to make now:

- **OOP vs functional structure** for the inference layer. Decide once 2–3 modules are written and friction becomes obvious.
- **Whether to use `__init__.py` re-exports** for cleaner imports. Default to empty files; refactor later if imports get ugly.
- **OT/SO model.** Start with a flat 50/50 conditional on tied regulation. Upgrade to a goalie-quality-weighted shootout model later if it materially changes calibration.
- **Score effects, travel.** Real factors, but adding them prematurely complicates the base model. Document as v2 features. (Back-to-backs were added in v2.0d.)
- **Player-level point distributions.** Requires shift-level data and per-player rate modeling. Defer until team-level pipeline is calibrated; planned on top of the v2.2 lineup model.

**Data and model items found in Phases 1–3:**

- **Redirect location drift.** Each season, redirects are recorded closer to the net (share under 10 ft: 24.8% in 2021-22 → 38.2% in 2025-26) while their goal rate falls (9.4% → 7.5%). The model overrates 2025-26 redirects by about 24%, and this accounts for most of the +2.4% total bias. Recency weighting helps only slightly. Retrain every season. The v2.1 season × shot-type effect ($\varepsilon$) is the planned fix.
- **Arena recording bias.** Some rinks log shot locations closer or farther than others. MoneyPuck adjusts for it; this model does not. Planned fix: rink effect ($\delta$) in v2.1.
- **Timestamp artifacts.** The feed logs whole seconds, so an opponent's shot can appear 1–2 s before a shot at the other net. Affects about 0.5% of shots.
- **NST rush definition.** NST appears not to count an opponent's shot as a rush precursor. Unconfirmed at three events; a query-time filter gives NST parity if needed.
- **Rebound and rush windows.** 3 s and 4 s are NST conventions, not fitted. Test 2/3/4 s by retraining and comparing validation log loss.
- **Previous-event location for stoppage-type events** is missing for about 0.5% of shots.

- **Season-level xG offset.** Leave-one-season-out xG runs about 3% high in 2021-22 and 2022-23 and about 3% low in 2023-24 and 2024-25 (xG/goals 1.032, 1.031, 0.974, 0.969, 1.024). Normal luck would move this about 1%, so goal conversion is changing between seasons more than shot quality explains (goaltending trends, recording changes, play the features miss). Within-season team comparisons are unaffected. Goalie ratings now handle it (`season_adjust=True` rescales each season to its own league average). The Phase 6 game model still needs a per-season league-average term or the same rescaling for team xG.
- **Early-season team strength.** v1 rolling windows span seasons (the last 25 games, whatever season), so every team has an estimate on opening night but off-season roster changes are ignored. Refinement: pull early-season estimates toward the league average.
- **When to retrain the xG model.** Keep the live model fixed during a season, so it never scores games it trained on. After the season ends, add it, move `CURRENT_SEASON` / `LAST_SEASON` forward one year, and run `train_xg.py --final` and `score_xg.py`.
- **Rename `CURRENT_SEASON` to `TEST_SEASON`.** It is used as the xG test season, not the season being played, which is easy to confuse.
- **Goalie signal is weak.** Adjusted next-season correlation is 0.125 (±0.12), so goalies should move game predictions only modestly. Repeat the `k` test when a new season is added; the best value sat on a flat stretch (400–1,600).
- **Team strength ignores schedule strength.** The rolling window doesn't adjust for the quality of opponents faced. Track B's joint fit handles this; the v1 rolling estimate does not.
- **Per-game MoneyPuck comparison** is still to do (Phase 2 gap).
- **Goalie uncertainty ignored in Track B v1.** Goalie offsets are fixed at the Phase 3 posterior mean (sd about ±0.03). Fix planned with the v2.1 goalie effect (draws instead of a fixed value).
- **Which goalie counts in Track B.** The starter's rating is used for the whole game; games where the starter is pulled get the starter's value. Splitting by time in net is a refinement if it ever matters.

**Model items found in Phase 5:**

- **Score effects / under-dispersion (resolved in Phase 6).** Regulation goals without empty-netters are less spread than Poisson (0.92) because of score effects; empty-net goals are part of the same behavior. Track B unchanged; score outputs come from the Phase 6 score-state simulator.
- **Track B uses goals, not xG (resolved in Phase 7).** Rolling xG beat v1; Track B v1.1 adds xG and ties it.
- **Fit window (resolved in Phase 7).** One season beats two for static ratings (two seasons were too confident out of sample).
- **Wide team intervals.** Every team's 90% net-rating interval is about ±0.09; mid-table teams overlap heavily. Expected with 164 games each; this uncertainty is what Phase 6 should carry into win probabilities.
- **OT model (resolved in Phase 6).** Sudden death at 2.39× the regulation rates; shootout 50/50 (data agree, p = 0.94).

**Model items found in Phase 6:**

- **Calibration too cautious at the extremes** (in-sample). **Resolved in Phase 7:** did not hold out of sample; the two-season v1 was too confident (0.79), v1.1 one-season is calibrated (0.99).
- **Home win slightly low** in simulation (52.9–53.4% vs 54.2%), inside the 90% range in-sample; recheck in Phase 7.
- **Remaining final-score spread** (score-state 1.011 vs 0.979 actual). Partly rating uncertainty; the rest small unmodeled effects (period-specific score effects, the 11% of empty-net goals outside late pulls). Not tuned in-sample.
- **Empty-net "hidden ties" in the exact model** (23.5% vs 22.8% past regulation). Doesn't affect P(win); score outputs use the score-state simulator. If standings simulations use the exact model, send about 96% of its regulation ties to OT.
- **Default starting goalie** is the team's most frequent starter in its last 10 games; real starters are announced on game day. Phase 7 can measure how much a wrong default costs.
- **2026-27 season.** Not yet loaded; predictions for current games need it plus the rebuild order (fit → late_game → score_state).

**Model items found in Phase 7:**

- **Track B ties but doesn't beat rolling xG,** and averaging the two still helps (about 0.003). The missing piece is recent form handled properly → v2.0 time-varying ratings.
- **Recency half-lives make Track B too cautious** (slopes 1.5–1.6 at 60–90 days): fewer effective games → wider ratings → predictions pulled toward 50%. Not adopted.
- **Rolling xG and Elo are too confident** (slopes 0.83 and 0.64): they treat point estimates as exact.
- **Market comparison not run yet:** needs historical closing moneylines (`data/external/closing_lines.csv`).
- **Total goals** are limited by season-to-season drift in league scoring; a per-season scoring term (or time-varying $\alpha_0$, with v2.0) could help.
- **Divergences per refit** are logged in the backtest CSVs; per-fit R-hat is not yet logged.
---

## Decisions Made

For future reference, the decisions locked in during planning:

- **First shipping target:** single-game win probability.
- **xG approach:** train own model, validate against MoneyPuck.
- **Uncertainty propagation:** honest version (joint hierarchical Bayesian via PyMC NUTS).
- **Build strategy:** Track A (AM MCMC) and Track B (PyMC) in parallel. Track A as integration test and reference; Track B as production.
- **Uncertainty sources to track:** all three (aleatoric, parameter, model misspec); rank with Sobol later, no pre-ranking.
- **Folder structure:** add `analysis/`, `validation/`, `pipelines/` as new top-level under `src/`. `game_model.py` at top level of `models/`.

Decided during Phases 1–2:

- **Raw tables stay raw.** Derived flags live in `shot_context`; filters are applied at query time.
- **xG training set:** unblocked shots, `period <= 4`, empty nets dropped.
- **Season roles:** test = `CURRENT_SEASON`, validation = `LAST_SEASON` (both in `config/settings.py`); the test season is graded once per change, never used to choose settings.
- **Rush in the model:** feed `last_event_type` and `time_since_last_event`; keep `is_rush` for reporting.
- **Strength state:** decoded shooter-relative (skater difference, total skaters, own goalie pulled), not the raw code.
- **Shot types grouped** into `wrist-snap` and `redirect` because the labels changed between seasons.
- **Recency weighting on** (1.0 / 1.25 / 1.5 / 2.0).
- **No recalibration step:** the remaining bias is concentrated in redirects, so a uniform shift would worsen other shot types.
- **Trained `.pkl` files stay out of git;** rebuild with `train_xg.py`.

Decided during Phase 3:

- **Per-shot xG comes from leave-one-season-out models** for historical seasons and from the live model for new ones; stored in `shot_xg`.
- **Team strength:** rolling 25-game window spanning seasons, total xG ÷ total minutes, variance = game-to-game variance ÷ games. Estimates use only earlier games.
- **Goalie rating:** Gamma-Poisson on GA ÷ xGA, `k = 800`, season-adjusted, two-season lookback. Chosen by the next-season prediction test in `phase3_check.ipynb`.
- **Goalie GA excludes empty-net and shootout goals;** team `gf`/`ga` include empty-net goals, `gf_xg`/`ga_xg` exclude them (the goal models use the latter; see Phase 4).
- **Build Phase 3 on complete seasons;** load 2026-27 when live predictions are needed. Keep the live xG model fixed during a season.

Decided 2026-10-17:

- **LightGBM stays as the xG base.** Shooter, goalie, rink and season effects go in a v2 hierarchical layer that uses walk-forward LightGBM xG as a fixed offset, not in a fully Bayesian replacement.
- **Track B gets a global intercept** $\alpha_0 \sim \mathcal{N}(\log 3, 0.5^2)$ and non-centered random effects from the start.
- **Track A reports only identified quantities** ($d_A, d_B$), with $\alpha_0$ fixed to the league mean.
- **Track A vs Track B agreement is tested like-for-like** (matched no-pooling Track B on the same games); the pooled comparison is qualitative only.
- **Lineup (player-level) model is v2,** after Phase 7. Its choices are listed in v2.2 with tentative defaults, not decided.
- **One goalie rating.** The Phase 3 goalie rating is built as a swappable interface; the v2 goalie effect replaces it only if it wins the Phase 7 backtest.

Decided during Phase 4:

- **Goalies enter Track B as a fixed input** ($\gamma = -\log \theta$ from the Phase 3 rating of the starter, as of game date), not as an estimated term. Avoids confounding with team defense and rating goalies twice. Cost: goalie uncertainty is ignored in v1 (about ±0.03, small next to team effects); the v2.1 goalie effect fixes it.

- **Track B v1 fits static team ratings on a recent window** (option A, e.g. the last 1–2 seasons), refit as time moves forward in the Phase 7 backtest. Per-season (B) and random-walk (C) ratings are v2.3 candidates.

- **Goal-model target excludes empty-net goals** (`gf_xg` / `ga_xg`) for Track A, Track B and the game model. Realistic final scores (6v5 endgame) can be a small Phase 6 add-on if ever needed; win probability does not need it.
- **Home ice is in the model.** Track A fixes it at the league value $\eta = 0.0715$; Track B estimates one league-wide $\alpha_{\text{home}}$ (expect about 0.07 ± 0.01, a free check). Team- and season-specific versions are v2.3 candidates.
- **Fixed league terms are computed from data at fit time,** not hard-coded, so they update as seasons are added.
- **Track A samples $(d_A, d_B)$ directly** (two parameters), not the four unidentified team terms.
- **Track A is a verification tool, not a prediction input.** Its saved posteriors are the reference for Track B's matched check.
- **Every MCMC fit runs 4 chains from spread-out starts** and reports R-hat, ESS and acceptance; posterior files carry the observed data and model settings.

Decided during Phase 5:

- **Track B fits regulation goals** (periods 1–3, no empty-netters). OT and the shootout are simulated separately in Phase 6. Track A comparisons use the periods 1–4 goals Track A was fit on.
- **Tie-inflated Poisson likelihood** ($\rho$, prior $\log(1 + \rho) \sim \mathcal{N}(0, 0.5^2)$) instead of independent Poisson, fit jointly with the ratings. `tie_inflation=False` keeps the plain model as a baseline.
- **Fit window `FIT_SEASONS` = 2024-25 + 2025-26,** static ratings, priors $\alpha_0 \sim \mathcal{N}(\log 3, 0.5^2)$, $\alpha_{\text{home}} \sim \mathcal{N}(0, 1)$, $\sigma \sim \text{HalfNormal}(0.3)$.
- **No model change for dispersion.** It comes mostly from excluding empty-net goals (section 11). Phase 6 adds an empty-net step for final-score outputs; score-state simulation stays an optional v2.3 item for in-game predictions.
- **The betting market is a Phase 7 competitor:** closing moneylines, margin removed, scored like the models.
- **Track B checks live in `phase5_check.ipynb`:** game table, leakage, synthetic recovery, matched Track A check, diagnostics, league terms, sanity vs goals and xG, posterior predictive, pooled vs Track A.
- **Saved posteriors are read into memory and closed** (`load_posterior()`), so open notebooks never lock files on Windows.

Decided during Phase 6:

- **P(win) from the exact model, score outputs from the score-state simulator.** Each does better at its job (section 4 vs section 9); the prediction header prints both win shares as a consistency check.
- **Uncertainty = spread of per-draw probabilities.** Each posterior draw gives an exact P(win); mean = prediction, 5th–95th percentile = 90% interval. Game randomness is inside each P(win).
- **OT:** sudden death at $k$ = 2.39 × regulation rates for 5 minutes; who scores follows the rates. **Shootout:** 50/50.
- **Score-state simulator:** 10-second slices, measured score-state factors, pull hazard, 6-on-5 and empty-net rates, one calibrated scale, **no tie inflation**.
- **No in-sample tuning of calibration.** Calibration issues are tested and fixed on held-out games in Phase 7.
- **Default goalie** = most starts in the team's last 10 games, unless given on the command line.
- **Rebuild order:** `fit_hierarchical` → `late_game` → `score_state`.

Decided during Phase 7:

- **Production uses v1.1, 365-day window** (same as the backtest). One record of the active fit (`active_fit.json`), read by every downstream script, instead of each script building the file name itself.

- **Adopted model: Track B v1.1, 1-year window, no recency half-life.** Ratings from xG + goals; statistically tied with rolling xG, best calibrated model tested (slope 0.99), no tuning knob.
- **xG goes inside Track B** as a second observed quantity (Gamma likelihood on regulation xG) sharing the team effects, with small tightly pooled finishing effects for goals beyond xG. Goalie offsets act on goals only.
- **One-year window over two** for static ratings.
- **Recency weighting rejected:** it tied rolling xG in 2025-26 but hurt 2024-25 and calibration.
- **No rolling-xG blend in production.** Kept as a benchmark only (no honest interval, ignores the starter, disagrees with the simulator). Blending should stop helping once v2.0 works.
- **Time-varying team ratings are the first v2 item.**
- **Backtest protocol:** walk-forward, weekly refits on a rolling date window, late-game and score-state parameters measured pre-season, actual starters rated as of game date, baselines tuned only on earlier seasons. Models compared on the same games with paired bootstrap ranges; a difference counts only if its 90% range excludes 0.
- **Market comparison stays in Phase 7's notebook** (section 10) and runs once closing-line data is available.

Decided during Phase 8:

- **Analysis A only** (sensitivity across matchups); analysis B (one matchup's posterior uncertainty, correlated inputs) moved to v2.0c.
- **Truth model = exact game model,** evaluated directly (fast); Saltelli on the exact model checks the PCE indices.
- **Inputs uniform and variance-matched** to their real spread (teams across the league, goalies across starters, league terms over their posterior); full-range inputs as a robustness check.
- **Same PCE method as the soil project** (Legendre, total degree, least squares on LHS, LOO Q², hold-out test), degree 4 by default; no Morris screening step, since 10 inputs are cheap to keep.

Decided during v2.0:

- **Live model: Track B v2.0** (time-varying team ratings and league levels, 730-day window, weekly steps), default in `fit_hierarchical` (`DEFAULT_MODEL = "tv"` in `active_fit.py`); v1.1 stays available with `--model xg`.
- **Weekly steps,** not 2-day: identical in the backtest, 3× cheaper.
- **League levels drift too,** with zero-sum team steps, so league-wide change is not absorbed by the team ratings.
- **Past games use the ratings of their date** in late_game and score_state.
- **Live fits use 2,000 draws** (`--draws 2000`); backtests keep 1,000.

Decided during v2.2:

- **Calibration: leave the model as is** (`scripts/check_calibration.py`, live model, 3,936 games): pooled slope 1.085 [0.96, 1.23], home bias +0.026 [−0.032, +0.079], both consistent with calibrated. Per-season slopes swing (1.16, 1.20, 0.83), and a walk-forward recalibration (fit on earlier seasons, apply to the next) helped 2024-25 (−0.0019) but clearly hurt 2025-26 (+0.0037); pooled +0.0009 [−0.0006, +0.0025]. The 1.07–1.08 seen across variants was noise.

- **Lineup model adopted (live from 2026-11-08):** −0.0009 [−0.0020, +0.0002] over 3,936 games, about 94% likely better. Adopted knowingly despite the range crossing 0; recheck with 2026-27 (checklist in the v2.2 section).
- **Live lineups:** ESPN injuries + the NHL game-day roster, with `--out` / `--in` / `--dtd-out`; see the v2.2 section.
- **Rest on finishing closed:** shrank to −0.0002 with a third season.
- **Shift charts:** `load_all` retries missing charts every run; 57 genuinely missing games are skipped.
- **Team system effects dropped:** they can't be separated from the roster.

Decided during v2.0b / v2.0d:

- **Live model: v2.0 + back-to-back effects.** `--rest` is the default for `--model tv`; `--no-rest` gives plain v2.0.
- **Home ice stays one fixed value:** letting it drift across summers changed nothing once rest was in.
- **ARI → UTA is one franchise** in the game table.
- **Guessed starters are fine:** a wrong guess costs about 0.0001 log loss; `--goalie` stays optional.
- **Rest enters through shot quality first;** rest on finishing/goaltending is tested separately before adding it.
- **v2.0e (rest on finishing) and v2.0f (travel) not adopted:** within luck / no gain vs v2.0 + rest. Both stay in the code (`--rest-goals`, `--travel`); v2.0e is the one to retest with more data.
- **Backtest runs:** one season per window, never the same season twice (two runs on one file duplicate games); check with `scripts/check_backtests.py`.

---

## Glossary

**xG** — Expected Goals. Per-shot probability of becoming a goal, summed over shots. Output of a classifier (usually gradient boosting) trained on shot features.

**RAPM** — Regularized Adjusted Plus-Minus. Ridge regression isolating player impact on shot/goal rates while controlling for teammates and opponents.

**GSAx** — Goals Saved Above Expected. $xGA - GA$ for goalies. Positive = better than average.

**PDO** — On-ice $SH\% + SV\%$. Mean-reverts to 100. Luck/regression indicator.

**Corsi / Fenwick** — Shot attempt differentials. Corsi includes blocks; Fenwick excludes them.

**Game Score** — Dom Luszczyszyn's weighted-stat composite (~1.0 = average game).

**PCE** — Polynomial Chaos Expansion. Spectral surrogate for expensive forward models. Orthogonal polynomial basis under input distribution.

**Sobol indices** — Variance-based sensitivity measures. $S_i$ = fraction of output variance attributable to input $i$ alone; $S_{T_i}$ = total contribution including interactions.

**NUTS** — No-U-Turn Sampler. HMC variant. Default sampler in modern PyMC.

**AM** — Adaptive Metropolis. Random-walk MH with covariance estimated from burn-in.

**ArviZ** — Python library for Bayesian model diagnostics and storage. NetCDF-based `InferenceData` is the standard format.

**Walk-forward backtest** — Validation strategy where, for each held-out game in chronological order, models are refit using only prior data. Avoids look-ahead bias.

**AUC** — Area under the ROC curve. The chance the model gives a random goal a higher xG than a random non-goal. 0.5 is guessing; public xG models land around 0.75–0.80.

**Log loss** — How far predicted probabilities are from what happened. Lower is better; a proper scoring rule.

**Calibration** — Whether predicted probabilities match observed rates: shots given 10% should score about 10% of the time.

**Gradient boosting** — Many small decision trees built in sequence, each correcting the previous ones' errors (LightGBM, XGBoost). Contrast with a random forest, which averages independently built trees.

**Early stopping** — Stop adding trees once the validation score stops improving, to avoid overfitting.

**Recency weighting** — Counting recent seasons more in training so the model follows current recording habits.

**Stint** — A stretch of game time during which the set of players on the ice does not change. The unit of data for RAPM.

**Tweedie distribution** — A compound Poisson–Gamma distribution: a point mass at zero plus a continuous positive part. Fits summed xG per stint.

**Offset** — A term added to a model's linear predictor with its coefficient fixed at 1, not estimated. In the v2 xG layer, logit(LightGBM xG) is the offset, so the Bayesian model only learns adjustments to it.

**Partial pooling** — In a hierarchical model, group effects (teams, shooters, goalies) share a common prior whose width is learned. Groups with little data are pulled toward the average; groups with lots of data keep their own estimate.

**Non-centered parameterization** — Writing a random effect as $\beta_s = \sigma_\beta z_s$ with $z_s \sim \mathcal{N}(0,1)$ instead of sampling $\beta_s \sim \mathcal{N}(0, \sigma_\beta^2)$ directly. Removes the "funnel" geometry that causes NUTS divergences when $\sigma$ is small.

**Leave-one-season-out** — Scoring each season with a model trained on all the other seasons, so no prediction comes from a model that saw that data.

**Conjugate prior** — A prior that, combined with the likelihood, gives a posterior of the same family, so the update is a simple formula. Gamma is conjugate to Poisson: Gamma(k, k) becomes Gamma(k + GA, k + xGA).

**Credible interval** — A range that contains the true value with a stated probability under the posterior (e.g. 90%). The Bayesian counterpart of a confidence interval.

**Shrinkage** — Pulling estimates from small samples toward the average. In the goalie rating, `k` sets how strongly.

**Posterior predictive check** — Simulating new data from the fitted model (using posterior draws) and comparing it with the real data on statistics that matter. A statistic outside the simulated range shows where the model's assumptions break.

**Tie inflation** — Raising the probability of tied scores above what independent scoring gives, with one parameter $\rho$ (tied scores × $(1 + \rho)$, then renormalized).

**Divergence (NUTS)** — A warning that the sampler hit a region of the posterior it can't follow accurately (usually sharp curvature). Even a few mean the results may be biased; the fix is usually a reparameterization.

**BFMI** — Bayesian fraction of missing information. Measures how well NUTS moves between energy levels; below about 0.3 suggests poor exploration.

**ESS** — Effective sample size. How many independent draws the correlated MCMC draws are worth. A few hundred per parameter is enough for means and 90% intervals.

**Score effects** — Teams play differently depending on the score: trailing teams take more risks and generate more shots, leading teams defend. Measured here: trailing teams score at 1.07–1.11× their normal pace, leaders at about 0.93×.

**Closing line** — A sportsbook's final odds before the game starts, after all betting and news (lineups, goalies, injuries). The standard benchmark for a prediction model.

**Vig (bookmaker's margin)** — The sportsbook's built-in edge: the implied probabilities of both sides add up to more than 1 (about 1.04–1.05 for NHL moneylines). Divide each by the sum to get fair probabilities.

**Fair odds** — Odds with no bookmaker margin, from a probability: p ≥ 0.5 → −100·p/(1−p); p < 0.5 → +100·(1−p)/p. What `predict_game.py` prints.

**Puck line** — A bet on the margin: −1.5 means winning by 2 or more (empty-net goals and the OT/shootout +1 count).

**6-on-5 / pulled goalie** — A trailing team replaces its goalie with an extra skater late in the game; it scores more often (about 2.7× here) but can concede into an empty net.

**Pull hazard** — The chance per second that a trailing team pulls its goalie, given the margin and time left; measured from play-by-play.

**Calibration slope** — Fit the outcome against logit(predicted probability) with a logistic regression. Slope 1 = calibrated; below 1 = predictions too extreme (over-confident); above 1 = too close to 50% (too cautious).

**Elo** — A rating system that moves each team's number after every game by $k$ × (result − expected result). Tuned here on earlier seasons: $k$, home edge, and how much rating carries over the summer.

**Paired bootstrap** — Resampling whole games and scoring both models on the same resampled games, so the range reflects the difference between models rather than which games happened to be played.

**Recency half-life** — Weighting each game by $0.5^{\text{age}/\text{half-life}}$, so a game one half-life old counts half as much.

**RPS (ranked probability score)** — Brier score for an ordered outcome (total goals): compares the predicted and actual cumulative distributions. Lower is better.

**Proper scoring rules** — Loss functions for probabilistic predictions that are minimized in expectation by the true distribution. Log-loss, Brier, CRPS are proper. RMSE on probabilities is not.