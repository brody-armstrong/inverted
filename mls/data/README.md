# Data

Raw and third-party data are **not redistributed**. The fetch scripts download them into the folders below; every folder except the small committed inputs is listed in `.gitignore`. Each source keeps its own terms, so check them before you reuse anything.

## Sources

| Folder / file | Contents | Source | How to get it | In repo |
|---|---|---|---|---|
| `data/raw/` | MLS, USL Championship, USL League One and MLS NEXT Pro player and team data: xG, xA, goals added (g+), minutes, positions, MLSPA salary releases, match xG | [American Soccer Analysis](https://www.americansocceranalysis.com/) public API (`app.americansocceranalysis.com/api/v1`) | `src/profiles/fetch.py`, `fetch_multi.py`, `fetch_teams.py`, `fetch_parity.py`; `src/paper1/random_fetch.py` | No. Please cite ASA |
| `data/espn/<league>/<year>.jsonl.gz` | Match summaries: lineups, substitution minutes, cards, shot commentary text, for MLS and five peer leagues (BEL, NED, POR, SCO, TUR), 2012–2026 | ESPN site API (unofficial, undocumented) | `src/paper2/p2_fetch_espn.py`, `espn_fetch.py` | No |
| `data/ext/odds_*.csv` | Closing 1X2 odds, MLS and European leagues | [football-data.co.uk](https://www.football-data.co.uk/) | `src/paper1/random_fetch.py` | No. Link to source |
| `data/ext/` (other) | Weather (Open-Meteo archive), NFL games (nflverse), MLB/NHL/NBA standings (league and ESPN APIs) for the cross-sport comparison in Paper 1 | Open-Meteo, nflverse, MLB Stats API, NHL API, ESPN | `src/paper1/random_fetch.py` | No |
| `data/raw/mls_stadia.json` | Stadium locations, capacity, surface | American Soccer Analysis API | `src/simulator/fetch_inputs.py` | No |
| `data/ext/stadium_elevation.json` | Stadium elevations | Open-Meteo elevation API | `src/paper1/random_fetch.py` | **Yes** (small; CC BY 4.0, Open-Meteo) |
| `data/nba/` | NBA player advanced stats (Box Plus-Minus) and each team's top 20 five-man lineups, 2014–2025 | [Basketball-Reference](https://www.basketball-reference.com/) | `src/paper2/nba_fetch.py` (one request every 3.5 s) | No. Check the site's terms before fetching |
| `data/tm/` | Transfermarkt tables (`games`, `appearances`, valuations, transfers) | Recorded in project notes as the dcaribou *transfermarkt-datasets* release; source and licence still to be confirmed | No fetch script included; download manually | No |
| `data/dp_wikipedia_*.psv` | Club-by-club Designated Player histories, transcribed 2025-12-19 and parsed 2026-05-12 | Wikipedia, "Designated Player Rule" (CC BY-SA 4.0) | `src/dp/dp_parse_wikipedia.py` on a saved copy of the page | **Yes**, under CC BY-SA 4.0 with attribution to Wikipedia contributors |
| `data/dp_overrides.csv` | Manual corrections to the DP list, each with a note | Hand-checked for this project | — | **Yes** |
| `data/dp_current_2026-05-12.csv` | Current DPs with 2026 compensation | Parsed from the Wikipedia page | `src/dp/dp_parse_wikipedia.py` | No (salary column pending a publication decision) |
| `data/ext/preseason_futures*.csv` | Preseason MLS Cup futures, all clubs, 2025 and 2026 (Feb 20), margin removed | SportsBettingDime odds tracker, entered by hand | — | No (publication decision pending) |

## Rebuilding from scratch

Rebuilding Paper 2 means re-fetching about 400 MB (ESPN ≈ 57 MB, ASA ≈ 122 MB, odds/standings/weather ≈ 126 MB, Transfermarkt ≈ 67 MB, plus NBA). Fetch scripts are resumable and rate-limited; please keep them polite.

## Sample data

None yet. A small sample (one MLS season of stints and ratings, for example) can be added so reviewers can run the Paper 2 pipeline end to end without fetching everything.

## Salary data

All salary figures come from the MLS Players Association's public salary releases, via the American Soccer Analysis API.

Historical prototype data removed from public repository due to source restrictions. An early development build of the simulator (v1.0) used a club-updated salary list compiled from commercial salary websites whose terms do not allow redistribution or machine-learning use. That list was never committed, is not used by any current output, and has been removed from the working project. `src/dp/dp_build.py` now uses the MLSPA releases only.
