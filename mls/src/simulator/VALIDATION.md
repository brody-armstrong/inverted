# Simulator v1.1 — validation against v1.0

**Date:** 2026-11-13 · **Reference:** Simulator v1.0, the development build live since 2026-11-12 · **Protocol:** [`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md), with thresholds fixed before any rebuilt number was seen.

**Verdict:** all pipeline checks pass, and none of the differences is material. Differences between v1.0 and v1.1 come from three sources, measured separately. (1) The rebuilt pipeline: none beyond rounding and simulation noise. (2) Player-ID matching: v1.0 missed MLS history for 45 players, and v1.1 corrects it. (3) The roster source: the intended change from a privately compiled, club-updated list to the public MLSPA release of April 16, 2026.

## What changed in the build

| Change | Why |
|---|---|
| Rosters from the latest MLSPA salary release (via the ASA API) instead of a privately compiled club-updated list | The private list was copied from commercial salary sites whose terms do not allow reuse; v1.1 uses public data only |
| No contract-end years; the "Remove contracts ending 2026" button is removed | Contract years came only from the private list |
| New `src/paper2/market_gd_mls.py` | The original step that made `out/market_gd_mls.pkl` was never saved; recreated from the documented method |
| New `src/simulator/fetch_inputs.py` | `mls_stadia.json`, match xG and MLS odds had no standalone fetch script |
| `src/paper2/peer.py --ratings-only` | The simulator needs only the ratings; the absence section needs `national_team_squads.pkl`, which still has no script |
| ESPN ↔ ASA crosswalks (`espn_asa_link.csv`, `espn_asa_players.csv`) taken from the committed v1.0 results | Regenerating them needs the full commentary-xG pipeline; they are ID mappings, not estimates |

## A. Pipeline checks

| # | Check | v1.0 | v1.1 | Threshold | Verdict |
|---|---|---|---|---|---|
| A1 | MLS matches with a market line | 5,987 | 5,987 | ±1% | Pass (exact) |
| A2 | Market ratings: matches · CV R² · team-quality SD · P5 shares | 5,731 · 0.779 · 0.2773 · 21/49/30 | 5,731 · 0.777 · 0.2773 · 21/49/30 | ±1% · ±0.01 · ±0.01 | Pass (SD and shares identical to 13 decimals) |
| A4 | Projection model: 8 coefficients, minutes curve, h, c, residual SD | committed `proj_model.json` | identical | 10% or 0.002; h, c ±2% | Pass (largest coefficient difference 2 × 10⁻¹⁴) |
| A5 | Backtest mean Spearman, model · naive | 0.285 · 0.222 | 0.280 · 0.222 | ±0.01 | Pass; see note |
| A6 | Player rating component independent of roster source (682 players matched with the same history flag) | — | corr 0.999995; median abs. diff 0.00002; max 0.00044 | corr ≥ 0.99; median ≤ 0.002 | Pass (differences are v1.0's 4-decimal rounding) |
| A7 | v1.0 rosters held fixed, ratings from v1.1 → projected points (page engine, 20,000 seasons, same seed) | — | max 0.22 pts; mean 0.08 pts | max ≤ 1.0; mean ≤ 0.5 | Pass |

**Note on A5.** The model is identical, so the backtest can differ only through simulation noise. `project.py` draws all backtest seasons from one random stream, so a change in the number or order of matches shifts every later draw. With seeds 2027, 2028 and 2029 the backtest gives 0.282, 0.288 and 0.286; v1.0's 0.285 lies inside that range. The team-strength SD in the page data (0.1325 vs 0.1323) moves for the same reason.

**Other checks.** The market line correlates 0.36 with ASA match xG difference. The results log reports 0.33 against Paper 2's text-based xG, so this is consistent, not a direct match. Model parameters on the page (curve, h, c, signing curve, comparables) match v1.0 to floating-point precision. The schedule is the same 510 matches in a different order.

## B. Roster-source change (documented, intended)

| | v1.0 | v1.1 |
|---|---|---|
| Roster source | private club-updated list (Mar–Sep 2026) | MLSPA release, April 16, 2026 |
| Players | 793 | 911 |
| In one build only | 66 | 184 |
| Rank correlation of projected points | — | 0.94 with v1.0 |
| Mean / largest change in projected points | — | 1.1 / 6.7 points |
| Largest change in playoff / MLS Cup probability | — | 19 / 3.4 percentage points |

**Decomposition of each club's change in projected points** (page engine, same seed):

| Source | Mean | Largest |
|---|---|---|
| Pipeline + player-ID matching (v1.0 rosters, v1.1 ratings and history flags) | 0.15 | 0.51 (Chicago) |
| Roster source | 1.15 | 6.8 (Columbus) |

**Causes of the largest changes:**

- **Gaps in v1.0's coverage.** v1.0 had only 17 Columbus players, 21 for LAFC and 21 for Orlando. The MLSPA release has 31, 30 and 27. The page fills missing roster spots at replacement level, so these clubs were understated in v1.0. Columbus rises from 14th to 4th (+6.7 points) once Steven Moreira, Andrés Herrera, André Gomes and others are included; LAFC gains 2.2.
- **Roster date.** Summer signings are not in the April release: Chicago without Robert Lewandowski (−3.0 points, 4th → 7th) and Orlando without Antoine Griezmann and Luis Muriel (−2.4, 19th → 26th). This is the cost of public data until the fall MLSPA release, and users can add these players back as scenarios.
- **Player-ID matching (a v1.0 defect, corrected).** v1.0 matched names from the private list to ESPN player ids and treated 45 players with MLS minutes as new arrivals (e.g. Djé D'Avilla, Sam Junqua, Andrew Thomas). v1.1 uses the ASA ↔ ESPN crosswalk from Paper 2. The effect is at most 0.5 points per club.

Name variants (e.g. "Nouhou Tolo" / "Nouhou", "Jeff Gal" / "Jeffrey Gal") count as one-build-only players in the B table. They are artefacts of comparing two name lists and do not affect either build.

## Reproducing this report

```bash
python3 src/simulator/fetch_inputs.py && python3 src/profiles/fetch.py && python3 src/profiles/fetch_parity.py
python3 src/paper2/espn_fetch.py usa.1 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
python3 src/paper1/random_matches.py && python3 src/paper2/market_gd_mls.py
python3 src/paper2/espn_shots.py usa.1 && python3 src/paper2/peer.py usa.1 MLS --ratings-only
python3 src/paper2/project.py
python3 src/simulator/tool_data.py && python3 src/simulator/roster_2027.py && python3 src/simulator/assemble_tool.py
node src/simulator/validation/engine.js <v1 data.json> > v1_engine.json && node src/simulator/validation/engine.js out/tool/data.json > v11_engine.json
python3 src/simulator/validation/compare_v1.py <v1 data.json> v1_engine.json v11_engine.json
```

The v1.0 reference data contains the private salary list and is not redistributed, so the comparison step can only be rerun by someone who holds it. The per-club and per-player comparison tables are generated in `out/validation/` and are not committed, for the same reason.
