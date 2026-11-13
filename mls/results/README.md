# Results

A snapshot of the small derived outputs behind the papers' figures, tables and checks. Scripts write to `out/` (not committed); these files are copies of `out/` as of 2026-11-12, when Paper 2 was frozen.

## Paper 1 — *Is MLS Built to Be Random?* (`src/paper1/`)

| File | Contents |
|---|---|
| `checks.csv` | Every validation check, pass/fail |
| `dc_diagnostics.csv` | Dixon-Coles NUTS diagnostics per season (R-hat, ESS) |
| `balance_*.csv` | Competitive balance across 16 leagues: luck share, Noll-Scully, rank correlation, titles |
| `sim_shapley.csv`, `sim_shapley_calibrated.csv` | Decomposition of MLS's extra randomness |
| `sim_formats.csv`, `sim_rankcorr.json`, `sim_relegation.json`, `sim_checks.json` | Format replays, year-to-year rank correlation, relegation replay, posterior predictive checks |
| `tests_*.csv/json` | Rule-change difference-in-differences with placebos, home advantage and travel, ghost games |
| `holdout_summary.json` | 2026 holdout: model vs closing odds (log loss, Brier) |

## Paper 2 — *Why MLS Teams Stay So Close Together* (`src/paper2/`)

| File | Contents | Prediction |
|---|---|---|
| `xg_validation.json` | Text-based xG model validation | — |
| `espn_asa_link.csv`, `espn_asa_players.csv` | ESPN ↔ ASA match and player crosswalks | — |
| `ratings_meta.json`, `ratings_split_half.json`, `finishing_meta.json` | Rating model parameters, stability gate, finishing estimate | gates |
| `rho_recovery_rating.json`, `rho_calib_rating*.json`, `nba_rho_rel*.json` | ρ recovery test, calibrated sets for MLS and the NBA method check | P1–P3 |
| `absence_rating_{xg,market}.json`, `absence_rating_goals_{goals,xg,market}.json` | Absence study and robustness versions | P4 |
| `peer_<league>.1.json` | Identical market-based pipeline for MLS and five peer leagues | P5, P9 |
| `roster_rank.json` | Roster-rank shares and points per $1M | P5, P6 |
| `p8_tiers.json` | Tier Shapley: stretching MLS tiers to peer spread | P8 |
| `persistence.json` | Who leaves after a rise; year-to-year quality; returning minutes | P7 |
| `proj_model.json`, `proj_backtest.json`, `futures_backtest.json` | Roster projection, backtest, preseason futures check | — |

`p8_tiers.json`, `persistence.json` and `futures_backtest.json` come from code not yet in the repository (see `src/paper2/README.md`, "Known reproducibility gaps").

## Earlier components

| File | Component |
|---|---|
| `comps_2026.json`, `roles_2026.json`, `teams.json` | Player profiles and team analysis (`src/profiles/`) |
| `spending.json`, `mls_payroll.csv` | MLS wage bills 2013–2026, team-season level (`src/profiles/spending.py`) |
| `parity_*.csv/json`, `mls_continuity.csv` | Parity and roster continuity vs other leagues (`src/profiles/parity.py`) |
| `pathway*.json` | USL → MLS translation (`src/pathway/`) |

## Deliberately not included

Player-level tables with salaries (the 2026 player table, DP lists, DP spells and team-season DP pay, which use the private 2026 salary list), per-match bookmaker probabilities (`holdout_preds.csv`), the built explorer and simulator pages, and the large `*.pkl` intermediates.
