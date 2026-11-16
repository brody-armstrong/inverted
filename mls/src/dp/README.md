# Designated Players, 2013–2026

Builds a season-by-season list of MLS Designated Players and measures what clubs get from them. Run from the `mls/` folder:

    python3 src/dp/dp_parse_wikipedia.py <saved_page.html> <out_history.psv> <out_current.csv>
    python3 src/profiles/fetch_parity.py && python3 src/profiles/fetch_multi.py   # ASA players, salaries, player xG (2013-2026)
    python3 src/dp/dp_build.py            # expand to player-seasons, match to ASA players, join salaries -> out/dp_draft_2013_2026.csv
    python3 src/dp/dp_late_arrivals.py    # players with 2026 minutes but no row in the April 2026 salary release
    python3 src/dp/dp_analysis.py         # points per $1M, DP spells, dead money -> out/dp_analysis.json

Inputs committed in `data/`: the parsed Wikipedia "Designated Player Rule" club histories (`dp_wikipedia_*.psv`, CC BY-SA, attribution in `data/README.md`) and `dp_overrides.csv` (manual corrections, each with a note).

Salaries come from the public MLSPA releases (via ASA) only. Players signed after the latest release (April 16, 2026) have no 2026 salary yet and are flagged as late arrivals until the next release. `parse_salaries_2026.py` is kept for parsing future salary workbooks; no third-party salary list is used.
