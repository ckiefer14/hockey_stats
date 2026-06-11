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
│   └── hockey_stats.db
│
├── src/
│   │
│   ├── data/                   # data acquisition & storage
│   │   ├── __init__.py
│   │   ├── nhl_api.py          # NHL API client
│   │   ├── db.py               # SQLite operations
│   │   ├── scraper.py          # Natural Stat Trick / Evolving-Hockey scrapers
│   │   └── cache.py            # caching layer
│   │
│   ├── processing/             # raw data → analysis-ready
│   │   ├── __init__.py
│   │   ├── shots.py            # shot data processing
│   │   ├── corsi.py            # Corsi/Fenwick calculations
│   │   ├── game_score.py       # game score breakdown
│   │   ├── team_strength.py    # rolling xGF/60, xGA/60 aggregates
│   │   ├── rapm.py             # RAPM / WAR (future)
│   │   └── goalie.py           # goalie statistics
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic: Game, Player, Shot, ScheduleGame
│   │   ├── game_model.py       # TOP-LEVEL: maps team inputs → λ_home, λ_away
│   │   │
│   │   ├── deterministic/
│   │   │   ├── __init__.py
│   │   │   ├── xg.py           # gradient-boosted xG model
│   │   │   └── rapm.py         # ridge regression RAPM (future)
│   │   │
│   │   └── stochastic/
│   │       ├── __init__.py
│   │       ├── am_sampler.py   # Adaptive Metropolis (ported from soil)
│   │       ├── goalie.py       # Bayesian goalie rating (simple conjugate)
│   │       ├── toy_mcmc.py     # Track A: hockey-flavored AM MCMC
│   │       ├── hierarchical.py # Track B: PyMC hierarchical model
│   │       ├── montecarlo.py   # Monte Carlo over posterior samples
│   │       └── pce.py          # PCE surrogate for Sobol analysis
│   │
│   ├── analysis/               # NEW: reduce sim output to distributions
│   │   ├── __init__.py
│   │   ├── aggregation.py      # P(win), credible intervals, marginals
│   │   └── sensitivity.py      # Sobol indices from PCE coefficients
│   │
│   ├── validation/             # NEW: scoring, calibration, backtesting
│   │   ├── __init__.py
│   │   ├── scoring.py          # log-loss, Brier, CRPS
│   │   ├── calibration.py      # reliability diagrams
│   │   └── backtest.py         # walk-forward validation harness
│   │
│   ├── pipelines/              # NEW: high-level orchestration
│   │   ├── __init__.py
│   │   ├── train_xg.py         # fit xG model, pickle to disk
│   │   ├── fit_hierarchical.py # fit PyMC model, save ArviZ NetCDF
│   │   ├── fit_toy.py          # fit Track A toy MCMC for comparison
│   │   ├── predict_game.py     # end-to-end single-game prediction
│   │   ├── backtest_season.py  # run validation harness on held-out games
│   │   └── build_pce.py        # fit PCE surrogate for Sobol
│   │
│   ├── viz/                    # visualization (existing)
│   │   ├── __init__.py
│   │   ├── theme.py
│   │   ├── shot_heatmap.py
│   │   ├── game_summary.py     # (future)
│   │   ├── player_card.py      # (future)
│   │   └── components/
│   │
│   └── export/
│       ├── __init__.py
│       └── save.py
│
├── notebooks/                  # exploration
│
├── output/                     # generated images, posterior samples
│   ├── game_summaries/
│   ├── player_cards/
│   ├── posteriors/             # NetCDF posterior samples
│   └── models/                 # pickled trained models (xG)
│
├── tests/
│   ├── test_corsi.py
│   ├── test_game_score.py
│   └── fixtures/
│
└── scripts/                    # CLI entry points
    ├── load_schedule.py
    ├── load_games.py
    └── load_shots.py
```

### Why Certain Things Live Where They Do

**`models/game_model.py` at the top level of `models/`** — neither purely deterministic nor purely stochastic. It's the bridge that takes deterministic components and lifts them into a stochastic combiner. Forcing it into one subfolder would be a category error. It belongs alongside `schemas.py` as a top-level model concern.

**`analysis/` separate from `models/`** — `models/` is about computation that produces parameters or predictions. `analysis/` is about *reducing* simulation output to the distributions you care about. Different responsibility: aggregation, not modeling.

**`validation/` separate from everything** — scoring rules and calibration plots are infrastructure that applies to any model. Keeping them separate means they can be reused if you swap component models.

**`pipelines/` for orchestration** — high-level entry points that chain components together. Each pipeline file should read like a recipe: load this, fit that, save the output. No business logic, just glue.

**Two tracks under `stochastic/`** — `toy_mcmc.py` (Track A, your soil-pipeline port) and `hierarchical.py` (Track B, PyMC) coexist. Track A is the reference implementation; Track B is the production model. They share input contracts so the rest of the pipeline doesn't care which produced the samples.

---
## Data Contract: Filtering Rules

`shot_events` is stored **raw and complete** — no filtering is applied at the
database level. Shootout attempts (period 5) and blocked shots are all present.
This is deliberate: the raw table stays a faithful record, and every consumer
applies the filters it needs at query time.

**Two rules, applied at query time:**

1. `period <= 4` — excludes shootout (period 5) events; keeps OT (period 4).
2. `event_type IN ('shot-on-goal', 'goal', 'missed-shot')` — excludes
   blocked shots (degenerate coordinates) and shootout attempts.

Together these yield **~565,000 clean unblocked non-shootout shots** — the xG
training set (Phase 2).

**Why this matters per consumer:**

- **xG model (Phase 2):** apply both rules. Blocked shots lack a true shot
  location and shootout attempts are mislabeled as goals; either would poison
  training.
- **Game model (Phase 6):** use `period <= 4` goals as the Poisson target so
  shootout-winner credits don't inflate goal counts. The shootout is modeled
  separately (flat 50/50 conditional on a tied regulation — see deferred OT/SO
  decision).

**Background:** the data-sanity notebook found that period-5 shootout attempts
are stored as `event_type` `'goal'` / `'shot-on-goal'` / `'missed-shot'`
(1,065 shootout "goals", ~2.6% of all goal events). All score-reconciliation
residuals after filtering are off-by-exactly-1 (the shootout-winner credit
added to the official score with no corresponding real goal event), zero
off-by-more-than-1 — fully characterized and expected, not a bug.

## Parallel Build: Track A vs Track B

The honest version of the model is the PyMC hierarchical NUTS sampler. But porting your AEM 8290 AM MCMC structure to a tiny hockey problem first buys real value: a reference implementation you trust, and an independent cross-check on the PyMC model.

### Track A — Toy AM MCMC (Soil Pipeline Port)

**Scope:** one team pair at a time. Parameter vector:

$$\theta = (\mu_{\text{off,A}}, \mu_{\text{def,A}}, \mu_{\text{off,B}}, \mu_{\text{def,B}})$$

**Likelihood:** Poisson on observed goal counts in head-to-head games:

$$p(y^{\text{obs}} \mid \theta) = \prod_{g} \text{Poisson}(G_{\text{home},g} \mid \lambda_{\text{home}}(\theta)) \cdot \text{Poisson}(G_{\text{away},g} \mid \lambda_{\text{away}}(\theta))$$

**Priors:** bounded uniform in standardized $\xi \in [-1, 1]^4$ space, same as soil.

**Sampler:** Adaptive Metropolis, three phases:
1. **Burn-in** ($n_b = 5000$): isotropic proposal $\Sigma_p = \sigma_0^2 I$ with $\sigma_0 = 0.1$
2. **Adapt:** estimate posterior covariance from burn-in, set $\Sigma_p = s \cdot \hat\Sigma$ with $s = (2.38)^2 / d$
3. **Sample:** main run with fixed adapted covariance

**Reuses:** your soil sampler module, standardization machinery, diagnostics (trace plots, ACF, acceptance rate, $R$-hat).

**Purpose:** proof of life, reference posteriors, bug-catching for Track B.

### Track B — Hierarchical PyMC Model

**Scope:** full league simultaneously.

**Model:**

$$
\begin{align}
\mu_{\text{off},t} &\sim \mathcal{N}(0, \sigma_{\text{off}}^2) \quad \text{for each team } t \\
\mu_{\text{def},t} &\sim \mathcal{N}(0, \sigma_{\text{def}}^2) \quad \text{for each team } t \\
\gamma_{\text{goalie},g} &\sim \mathcal{N}(0, \sigma_{\text{goalie}}^2) \quad \text{for each goalie } g \\
\alpha_{\text{home}} &\sim \mathcal{N}(0, 1) \\
\log \lambda_{\text{home}} &= \alpha_{\text{home}} + \mu_{\text{off},h} - \mu_{\text{def},a} - \gamma_{\text{goalie},a} \\
\log \lambda_{\text{away}} &= \mu_{\text{off},a} - \mu_{\text{def},h} - \gamma_{\text{goalie},h} \\
G_{\text{home}} &\sim \text{Poisson}(\lambda_{\text{home}}) \\
G_{\text{away}} &\sim \text{Poisson}(\lambda_{\text{away}})
\end{align}
$$

**Hyperpriors** on $\sigma_{\text{off}}, \sigma_{\text{def}}, \sigma_{\text{goalie}}$ — typically half-normal or half-Cauchy.

**Sampler:** NUTS, default PyMC settings to start, reparameterize if divergences appear.

**Output:** ArviZ `InferenceData` with full joint posterior across all teams and goalies.

**Purpose:** the actual production model.

### Cross-Validation Between Tracks

For any team pair $(A, B)$:
1. Fit Track A on $(A, B)$ head-to-head data only.
2. Marginalize Track B's joint posterior down to the four parameters $(\mu_{\text{off},A}, \mu_{\text{def},A}, \mu_{\text{off},B}, \mu_{\text{def},B})$.
3. Compare marginal posteriors. They should be close but not identical — Track B benefits from partial pooling across all teams, Track A doesn't.

When they disagree wildly, one of them is wrong, and you have a much better debugging position than with just one implementation.

---

## Build Sequence

Realistic for a side project with focused sessions a few times a week. Roughly two months to a usable single-game predictor.

### Phase 1 — Data Foundation (Weeks 1–2)

- Expand `scripts/load_shots.py` to iterate over the full league schedule, not just COL. Run overnight.
- Verify shot data integrity: row counts, NULL rates, distribution of shot coordinates, presence of all 32 teams.
- Confirm play-by-play covers at least one full season; ideally backfill 2–3 seasons.

**Blocker for everything downstream.** Don't start Phase 2 until this is solid.

### Phase 2 — xG Model (Week 3)

- Write `src/models/deterministic/xg.py`. Class with `.fit()`, `.predict_proba()`, `.save()`, `.load()`.
- Features: distance, angle, shot type, rebound flag, rush flag, strength state, prior event type, time since prior event.
- Model: LightGBM or XGBoost with log-loss objective.
- Temporal train/test split (no random shuffle — leaks future info into past).
- Validate per-game team xG totals against MoneyPuck on held-out games. Target $r > 0.9$.
- Pickle the trained model with version stamp: `xg_lgbm_v1_2024-2025.pkl`.

### Phase 3 — Team Strength & Goalie Aggregates (Week 4)

- `src/processing/team_strength.py`: rolling 25-game windows of xGF/60 and xGA/60 per team.
- `src/models/stochastic/goalie.py`: simple Bayesian goalie rating. Beta-Binomial on save proportion, or Normal-Normal on GSAx/60. No hierarchy yet — that comes in Track B.
- Both produce: point estimate + variance for each team / goalie at any point in the season.

### Phase 4 — Track A Toy MCMC (Week 5)

- Port your AM sampler from soil to `src/models/stochastic/am_sampler.py`.
- Write `src/models/stochastic/toy_mcmc.py`: 4-parameter Poisson model for one team pair.
- Write `src/pipelines/fit_toy.py`: load team pair data, run AM, save posterior samples (ArviZ NetCDF for consistency with Track B), produce diagnostics (trace plots, ACF, acceptance rate).
- Validate: synthetic recovery on simulated data with known $\theta^*$, like the soil project's posterior recovery check.

### Phase 5 — Track B Hierarchical PyMC (Week 6)

- Write `src/models/stochastic/hierarchical.py` with the full league model.
- Write `src/pipelines/fit_hierarchical.py`: load full-season game data, fit, save InferenceData.
- Diagnostics: $R$-hat, ESS, divergences, energy plots.
- Cross-validate against Track A on 2–3 team pairs.

### Phase 6 — Game Model + Monte Carlo (Week 7)

- Write `src/models/game_model.py`: function taking $(\mu_{\text{off}}, \mu_{\text{def}}, \gamma_{\text{goalie}}, \text{home flag})$ samples and producing $(\lambda_{\text{home}}, \lambda_{\text{away}})$ samples.
- Write `src/models/stochastic/montecarlo.py`: draws posterior samples, runs game model, samples Poisson goal counts, handles OT/SO.
- Write `src/pipelines/predict_game.py`: end-to-end orchestration. Takes two teams + date, returns predictive distribution.
- Write `src/analysis/aggregation.py`: reduce sim output to $P(\text{win})$, credible intervals on score margin, etc.

### Phase 7 — Validation (Week 8)

- `src/validation/scoring.py`: log-loss, Brier score, CRPS for goal totals.
- `src/validation/calibration.py`: reliability diagrams with bootstrap confidence bands.
- `src/validation/backtest.py`: walk-forward harness. For each held-out game in chronological order, fit components only on prior data, predict, score.
- `src/pipelines/backtest_season.py`: run the full validation on a held-out season tail.

### Phase 8 — PCE for Sobol (Week 9+)

- `src/models/stochastic/pce.py`: port PCE machinery from soil. Legendre basis (uniform inputs), total-order truncation, non-intrusive regression.
- Fit on team-level input ranges (e.g., xGF/60 ∈ [2.0, 3.5], goalie GSAx ∈ [-0.3, 0.5]) with $P(\text{home win})$ as the QoI.
- `src/analysis/sensitivity.py`: analytical Sobol indices from PCE coefficients (the "sensitivity for free" trick).
- Cross-check with Saltelli sampling on the surrogate.

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
3. **Model misspecification.** The Poisson form, the independence assumption between home/away goals, the missing factors (rest, travel, injuries). Hardest to quantify; partially addressed by Dixon-Coles corrections and held-out validation.

Sobol analysis at the end ranks which of these matter most for which outcomes.

---

## Persistence & Reproducibility

- **xG model:** pickle with version stamp under `output/models/`. Config entry points to active version.
- **Posterior samples:** ArviZ NetCDF under `output/posteriors/`. Both tracks write to the same format so downstream code is sampler-agnostic.
- **PCE surrogate:** pickle with input-range metadata and coefficient array.
- **Backtest results:** CSV under `output/backtests/` with one row per held-out game (date, teams, prediction, outcome, log-loss contribution).

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
- **Score effects, rest, travel, back-to-backs.** Real factors, but adding them prematurely complicates the base model. Document as v2 features.
- **Player-level point distributions.** Requires shift-level data and per-player rate modeling. Defer until team-level pipeline is calibrated.

---

## Decisions Made

For future reference, the decisions locked in during planning:

- **First shipping target:** single-game win probability.
- **xG approach:** train own model, validate against MoneyPuck.
- **Uncertainty propagation:** honest version (joint hierarchical Bayesian via PyMC NUTS).
- **Build strategy:** Track A (AM MCMC) and Track B (PyMC) in parallel. Track A as integration test and reference; Track B as production.
- **Uncertainty sources to track:** all three (aleatoric, parameter, model misspec); rank with Sobol later, no pre-ranking.
- **Folder structure:** add `analysis/`, `validation/`, `pipelines/` as new top-level under `src/`. `game_model.py` at top level of `models/`.

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

**Proper scoring rules** — Loss functions for probabilistic predictions that are minimized in expectation by the true distribution. Log-loss, Brier, CRPS are proper. RMSE on probabilities is not.