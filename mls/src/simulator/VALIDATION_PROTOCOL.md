# Simulator rebuild — validation protocol (written 2026-11-13, before any rebuilt number was seen)

Reference: Simulator v1.0 (live development build, `out/tool/data.json` from the 2026-11-12 handoff bundle) and the committed v1.0 results (`results/peer_usa.1.json`, `results/proj_model.json`, `results/proj_backtest.json`).

The rebuild differs from v1.0 in two separable ways, and they are measured separately:

- **A. Pipeline differences**: same method, re-fetched data, recreated steps (`market_gd_mls.pkl`, `mls_stadia.json`). These should be close to zero.
- **B. Roster-source change** (intended): rosters from the MLSPA Spring 2026 release (via ASA) instead of the private club-updated list; no contract-end years. These differences are expected and are documented, not treated as failures.

## Stage checks (A)

| # | Quantity | Reference | Material if |
|---|---|---|---|
| A1 | MLS matches with a market line (`market_gd_mls`) | 5,987 (results log) | differs by more than 1% |
| A2 | `peer.py usa.1`: matches, CV R², team-quality SD | 5,731; 0.78; 0.28 | matches ±1%; R² ±0.01; SD ±0.01 |
| A3 | Player ratings (market), player-season level | v1.0 ratings not saved; checked via A2 and A4 | — |
| A4 | Projection model coefficients (`proj_model.json`) | committed v1.0 | any coefficient moves by more than 0.1 × its own magnitude or 0.002 absolute, whichever is larger; h, c ±2% |
| A5 | Backtest (`proj_backtest.json`) mean Spearman, model and naive | 0.285 / 0.222 | ±0.01 |
| A6 | Per-player rating component that does not depend on roster source (projection minus salary and age terms), players in both builds | v1.0 `data.json` | correlation < 0.99, or median absolute difference > 0.002 |
| A7 | Team strength (Q) and projected points with **v1.0 rosters held fixed** | v1.0 rosters run through rebuilt model vs v1.0 model | any team's projected points differ by > 1.0, or mean absolute difference > 0.5 |

If any A-check is material, the cause is investigated and explained before release. If the cause cannot be found or fixed, the release is held and the gap is reported.

## Roster-source change (B), documented

- Players per team; players in v1.0 not in the rebuild and vice versa.
- Per team: change in team strength Q and projected points (rebuild vs v1.0), with the largest contributors named.
- League level: Spearman correlation of team projected points between v1.0 and the rebuild.

## Output

`VALIDATION.md` with every number above, pass/material verdict for each A-check, the B tables, and the decision.
