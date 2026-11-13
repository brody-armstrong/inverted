# Paper 2 — pre-registered predictions

Written 2026-11-12, before any Paper 2 data was downloaded or any model was fitted. These are fixed. Results that contradict them are reported as such, not reworded.

## Definitions

- **Player rating**: a player's contribution to his team's expected goal difference per 90 minutes, from the plus-minus model (MLS: xG on the field; all leagues: closing-odds expected goal difference).
- **Team quality**: a generalized mean of the starting lineup's ratings, Q = (Σ wᵢ qᵢ^ρ)^(1/ρ), with q a positive transform of the rating and wᵢ minutes weights. ρ = 1 means team quality is the plain average. ρ < 1 means weaker players count more (weak-link). ρ > 1 means stronger players count more (strong-link).
- **Roster rank**: a player's rank on his team by rating, among players with ≥ 900 regular-season minutes.

## Predictions

| # | Prediction | Direction | Falsified if |
|---|---|---|---|
| P1 | Soccer is weak-link: ρ < 1 in MLS | ρ_MLS < 1 | 95% interval for ρ_MLS lies at or above 1 |
| P2 | Same property outside MLS: ρ < 1 in each peer league | ρ_peer < 1 | interval at or above 1 in most peer leagues |
| P3 | The method separates sports: the NBA is strong-link | ρ_NBA > 1 | interval at or below 1 → the **method** fails, and P1–P2 are not trusted |
| P4 | Absence of an outside-cause player (call-up, suspension) costs more per rating point when he is a lower-ranked starter (ranks 6–11) than when he is the best player (rank 1) | larger per-point effect for ranks 6–11 | rank-1 effect per rating point ≥ ranks 6–11 effect |
| P5 | In MLS, roster ranks 4–11 account for a smaller share of between-team differences in team quality than in peer leagues; ranks 1–3 account for a larger share | MLS depth share < peer depth share | MLS depth share ≥ peer median |
| P6 | Expected points per $1M is higher for ranks 4–11 than for ranks 1–3 in MLS | depth > stars | interval for the difference includes 0 or favours stars |
| P7 | Persistence: players whose rating rose are more likely to leave their club the next offseason than players whose rating fell | positive | association ≤ 0 |
| P8 | The weak-link design (cap on depth, free spending on top three) explains at least a third of the MLS-vs-peer gap in team-quality spread | ≥ 33% | point estimate < 33% |

## What would make each prediction meaningless

- The method fails parameter recovery on simulated leagues with a known ρ.
- Player ratings are unstable across half-seasons (split-half correlation below 0.3).
- Fake absences on random dates show effects as large as real ones.

If any of these fails, the dependent predictions are reported as untestable, not as confirmed or rejected.

## Addendum 1 — 2026-11-12, after data collection, before any model was fitted

Recorded after the data were downloaded and checked (counts only: 1,932 call-up absences of regular MLS starters) and before any rating, ρ or absence effect was estimated. P1–P8 are unchanged.

- **P9 (comparative absence slope).** If MLS's roster design compresses depth, losing a lower-ranked starter should cost MLS teams no less, per rating point, than it costs peer-league teams. Prediction: the ratio (per-point cost at ranks 6–11) ÷ (per-point cost at rank 1) is at least as large in MLS as in the peer leagues. Falsified if the MLS ratio's 95% interval lies entirely below the peer-league ratio.
- **Not added:** "MLS has a lower ρ than peer leagues." It conflicts with P1–P2, which treat weak-link as a property of the sport, with MLS's design acting on it rather than creating it. A clearly lower ρ in MLS would be reported as a finding that contradicts that framing.
- **Estimation order (fixed now).** Stage 1: player ratings alone. Stage 2: freeze ratings, estimate ρ. Stage 3: the fully joint model, reported as a robustness check. If stages 2 and 3 disagree on whether ρ < 1, both are reported and the disagreement is a headline caveat.
- **Face-validity check (descriptive, not a test).** Before the ρ results are read, the model's ranking of known teams is shown: 2019 LAFC, 2022 Philadelphia and LAFC, 2024 Inter Miami, plus the weakest teams of the same seasons. Their ratings must place them among the elite or the weakest teams of their seasons for interpretable reasons (who drives the rating). A failure here is reported, not fixed silently.

## Addendum 2 — 2026-11-12, after the recovery test, before any real-data ρ was estimated

Results recorded so far (no real-data ρ has been computed):

- **Stability gate: passed.** Split-half correlation of final ratings 0.48 (threshold 0.30); the on-field plus-minus part alone is 0.15, so most stability comes from the box-score prior.
- **Face validity: mixed, reported.** 2022 Philadelphia #1 and 2019 LAFC #2 by rating; 2022 LAFC #4; 2023 Columbus #1. **2024 Inter Miami ranks 11th of 29** despite the best record: ASA's xG agrees (xG difference 20th), and Miami scored 0.85 goals per game above xG. Cause: xG-based ratings cannot see finishing. Response: a second rating version adds shrunk finishing skill and goalkeeper shot-stopping. Finishing barely persists year to year in MLS (r ≤ 0.06), so the most generous estimate the data allow is used (skill SD 0.009 per shot), deliberately favouring stars. ρ is reported for both rating versions.
- **Parameter recovery: the naive frozen-rating estimator FAILED.** It recovers the direction but is biased toward weak-link: with a true ρ of 1 it returned 0.75 (spread ×1) and 0.5, interval 0.0–1.0 (spread ×2); true −2 → 0.0; true 3 → 1.25–1.5. Cause: noise in estimated ratings, which a lower ρ partly absorbs. **The naive estimate will not be used to test P1–P3.**

Fixed now, before any real-data ρ:

1. **Primary ρ estimator = simulation-calibrated (Neyman) confidence set.** Simulate the league from known ratings on a grid of true ρ (−3 to 4), at a true rating spread chosen so that the simulated split-half correlation of the plain plus-minus ratings matches the real one (0.15). For each true ρ, run the same estimator on 10+ simulated datasets. The 95% confidence set for ρ is every true value whose simulated estimates' central 95% range contains the real-data estimate. P1 is supported only if the whole set lies below 1; falsified if it lies at or above 1; otherwise **inconclusive**.
2. **The calibrated estimator must itself pass recovery** on fresh simulations, including a spread different from the one used to calibrate (×0.5 and ×2). If it does not, ρ from stints is reported as not identifiable from these data.
3. **The absence study (P4, P9) becomes the main independent test of weak vs strong link**, because it uses outside-caused lineup changes rather than noisy rating extremes.
4. The same calibration is applied to the NBA (P3) and the peer leagues (P2).
