# Timbers2 readiness passport — pre-registration

Written and committed before any outcome data was fetched for this analysis (commit timestamp is the record; the commit is local until J approves a push, so there is no external timestamp yet).

## Question
Which MLS NEXT Pro (T2) players are ready for first-team minutes, and how sure are we?

## Data
- American Soccer Analysis public API, MLS NEXT Pro and MLS, seasons 2022–2026, at player-match level (`split_by_games=true` is available for xG and goals added in both leagues; checked 18 Nov 2026). MLS minutes before 2022 come from ASA MLS player-season data, 2013–2021, only to establish prior MLS minutes.
- Birth dates: ASA's `birth_date`. This is a declared deviation from the brief ("MLS official rosters and club announcements only"): those pages do not expose birth dates in a collectable form, and the API behind them is undocumented. Approved by J on 18 Nov 2026. The 14 current Timbers2 regulars' birth dates are spot-checked against official Timbers/MLS NEXT Pro pages where shown.
- Contract outcomes: not collected (club announcements cannot be gathered reliably at this scale). Neither outcome below needs them.
- Baseline model: the published pathway model (mls/results/pathway_model.json).

## Cohorts
- A cohort row is a player-season S in MLS NEXT Pro with: outfield position; ≥ 900 MLS NEXT Pro minutes in S; < 500 cumulative MLS minutes (as counted by ASA, all seasons up to and including S).
- Training cohorts: S = 2022, 2023. Test cohort: S = 2024 (outcomes 2025–2026). A player can appear in more than one training cohort; resampling is by player.
- Position groups as in the profiles explorer: Strikers, Wingers, Midfield, Fullbacks, Centre-backs (from ASA general_position).

## Outcomes
- (a) Reaching 500 MLS minutes in total over seasons S+1 and S+2. This partly measures coach selection and club context (depth chart, affiliation, injuries), not only ability; the page says so.
- (b) MLS g+ per 96 (above average, ASA), pooled over S+1 and S+2, for players who reach 500 MLS minutes in that window.

## KPIs (per 96 minutes, MLS NEXT Pro, season S)
g+ total; g+ by action: dribbling, passing, receiving, shooting, interrupting; xG; xA.

## Reliability gate
- Split-half: odd vs even matches (by date) within a season, players with ≥ 450 minutes in each half, training seasons 2022–2023 pooled. Pearson r between halves, Spearman–Brown corrected to a full season.
- A KPI stays in the passport if corrected r ≥ 0.40. A KPI below 0.40 is shown on the page as "not reliable enough to rank players", with its number, and is not used in any model.

## Methods
- Shrinkage: empirical-Bayes normal–normal model per KPI × position group × season. Each player's sampling variance comes from his own match-level variation; the prior mean and variance are estimated across players by method of moments. Output: posterior mean and 90% credible interval.
- Age-adjusted percentile: percentile, within position group and season, of the residual from a quadratic regression of the shrunk KPI on age.
- Readiness model (new): discrete-time hazard, two periods (S+1, S+2), logistic link: period intercepts + age + shrunk g+ total (z-scored within group and season) + log NEXT Pro minutes in S + log(1 + MLS minutes in S) + position group. P(a) = 1 − (1 − h1)(1 − h2). Intervals: 90% from 500 player-level bootstrap refits, each with the g+ input drawn from its posterior.
- Baseline 1 (raw percentiles): same hazard structure with only the raw (unshrunk) g+ total percentile within group and season.
- Baseline 2 (minutes-only): same hazard structure with only log NEXT Pro minutes and log(1 + MLS minutes) in S.
- Calibration curves (five equal-count bins) for every probability reported: the new model and both baselines on the test cohort.

## Pass rules
- (a) The new model is used on the page only if, on the 2024 test cohort, it has lower Brier score AND lower log loss than both baselines. Otherwise the page says the new model was not supported and shows the probability from the baseline with the lower test log loss, labelled as such.
- (b) Model: linear regression of MLS g+/96 on shrunk NEXT Pro g+/96 and age, fitted on training cohorts. Baseline: the published pathway projection of MLS g+/96 from NEXT Pro production, applied as published. Supported only if test RMSE is lower than the baseline's. If fewer than 10 test players reach 500 MLS minutes, (b) is reported as insufficient data. Leakage note: the pathway model was fitted on move-ups that likely include the 2024 cohort, which favours the baseline.
- Results that do not meet these rules are reported as not supported. No threshold, cohort rule, covariate or KPI changes after outcome data is fetched, except to fix a coding error, which is logged in the methods note with what had been seen.

## 2027 predictions
- Final model: whichever model the pass rule selects, refitted on every person-period with an observed outcome (cohorts 2022–2025).
- Prediction: for every 2026 cohort player (same eligibility), P(≥ 500 MLS minutes in 2027), one season only, with 90% interval. Written to a timestamped file and scored after the 2027 season (Brier, log loss, calibration).

## Disclosures (seen before writing this)
- On 18 Nov 2026, while building the first passport, I saw the 2024–2026 MLS minutes of the 14 Timbers2 regulars of 2026 (e.g. Eric Izoita, 475 MLS minutes in 2026). MLS season files for 2024–2026 were downloaded then; only those 14 players' rows were viewed. The pipeline refetches everything.
- The pathway model's published results (translation factors, track record of past move-ups) were known.
- No league-wide outcome for any 2022–2024 cohort had been looked at.
