# Paper 2 — Why MLS Teams Stay So Close Together

Code for the frozen v1.0 article. Predictions and verdicts: [`papers/paper2/`](../../papers/paper2/). Run from the `mls/` folder with Python 3.13 (`requirements.txt`).

## Run order

| Step | Command | Produces |
|---|---|---|
| 0 | Paper 1 steps 1–2 (`src/paper1/random_fetch.py`, `random_matches.py`) | `out/mls_matches.pkl` (ASA match table with closing odds), used by steps 3 and 10 |
| 1 | `python3 src/paper2/p2_fetch_espn.py <league> <first_year> <last_year>` for `usa.1`, `bel.1`, `ned.1`, `por.1`, `sco.1`, `tur.1` (2012–2026); `espn_fetch.py` is the single-season variant | `data/espn/<league>/<year>.jsonl.gz` |
| 2 | `python3 src/paper2/espn_shots.py usa.1` | shot and match tables from commentary text |
| 3 | `python3 src/paper2/shot_xg.py` | text-based xG, validation (`xg_validation.json`), ESPN↔ASA match link |
| 4 | `python3 src/paper2/stints.py` | 42,463 stints, player crosswalk (`espn_asa_players.csv`) |
| 5 | `python3 src/paper2/ratings.py` (and `--split-half`), then `ratings_finishing.py` | xG ratings (λ = 8,000) and finishing-adjusted ratings (`rating_goals`) |
| 6 | `python3 src/paper2/rho.py <rating_col> [--recover]`; `rho_calib.py spread \| grid \| real \| check <rating_col>` | naive ρ, recovery test, calibrated Neyman sets |
| 6b | `python3 src/paper2/nba_fetch.py 2014 … 2025`; `nba_rho.py [REL]` | NBA method check (P3) |
| 7 | `python3 src/paper2/absence.py <rating|rating_goals> <xg|market|goals>` | absence study (P4, placebo) |
| 8 | `python3 src/paper2/peer.py <espn_league> <football-data code or MLS>` | identical market-based pipeline for MLS and five peer leagues (P5, P9) |
| 9 | `python3 src/paper2/roster_rank.py` | roster-rank shares and points per $1M (P5, P6) |
| 10 | `python3 src/paper2/project.py` | roster projection, backtest, 2027 outlook (feeds the simulator) |

`rho.py` and `rho_calib.py` load the model pieces from `src/paper2/ratings.py` by path, so run them from the `mls/` folder.

## Known reproducibility gaps

`out/market_gd_mls.pkl` is no longer a gap: `src/paper2/market_gd_mls.py` (added 2026-11-13) recreates it from the documented method and reproduces the v1.0 market ratings exactly (see `src/simulator/VALIDATION.md`). `peer.py --ratings-only` runs without the national-team squads.

The following inputs or results were produced in the original working sessions by code that is not in this repository yet. The numbers in `results/` and in the article are the ones those runs produced; the steps below cannot regenerate them until the code is added.

| File | Used by / reported in | What it is |
|---|---|---|
| `out/national_team_squads.pkl` | `absence.py`, `peer.py` | national-team match-day squads from ESPN (≈10,500 international matches), used to identify call-up absences |
| `results/futures_backtest.json` | `src/simulator/tool_data.py`; article (futures check) | model vs preseason MLS Cup futures, 2025 and 2026 |
| `results/p8_tiers.json` | article (P8, tier Shapley) | stretching each MLS roster tier to the peer-median spread |
| `results/persistence.json` | article (P7, persistence accounting) | who leaves after a rise in rating; year-to-year team quality; returning minutes |

## Large intermediates

Stints, ratings and match tables (`out/*.pkl`) are rebuilt by the steps above and are not committed. The small JSON/CSV results behind every figure and table are in [`results/`](../../results/).
