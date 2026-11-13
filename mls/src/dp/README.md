# Designated Players, 2013–2026

Builds a season-by-season list of MLS Designated Players and measures what clubs get from them. Run from the `mls/` folder:

    python3 src/dp/dp_parse_wikipedia.py <saved_page.html> <out_history.psv> <out_current.csv>
    python3 src/dp/parse_salaries_2026.py <salary_workbook.xlsx> <out.csv>     # private input, see below
    python3 src/dp/dp_build.py            # expand to player-seasons, match to ASA players, join salaries -> out/dp_draft_2013_2026.csv
    python3 src/dp/dp_late_arrivals.py    # players with 2026 minutes but no row in the April 2026 salary release
    python3 src/dp/dp_analysis.py         # points per $1M, DP spells, dead money -> out/dp_analysis.json

Inputs committed in `data/`: the parsed Wikipedia "Designated Player Rule" club histories (`dp_wikipedia_*.psv`, CC BY-SA, attribution in `data/README.md`) and `dp_overrides.csv` (manual corrections, each with a note).

Not committed: `data/salaries_2026-08.csv` (a third-party 2026 salary list, private). `dp_build.py` uses it for 2026 salaries, so a public rebuild has to drop that step or substitute the MLSPA release. The DP list and analysis outputs carry player-level salaries and are not committed either.

Article: [What MLS Clubs Get From Designated Players, 2013–2026](https://claude.ai/artifact/NZZtsuHn4o933qZYLpwYP1).
