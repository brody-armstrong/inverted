# Player profiles (phase 1), team analysis, spending and parity

Run from the `mls/` folder.

    python3 src/profiles/fetch.py 2026 2025      # pull ASA data -> data/raw
    python3 src/profiles/fetch_multi.py          # ASA data for USL Championship, USL League One, MLS NEXT Pro (used by src/pathway)
    python3 src/profiles/fetch_teams.py          # ASA team-level data
    python3 src/profiles/build.py 2026            # fingerprints, roles, comparables, value -> out/players_2026.csv, comps_, roles_
    python3 src/profiles/export.py               # compact JSON for the explorer -> out/data.json
    python3 src/profiles/team_analysis.py        # team drivers and 2026 squad gaps -> out/teams.json (needs out/pathway.json)
    python3 src/profiles/assemble.py             # out/data.json + out/pathway.json + out/teams.json -> out/index.html

    python3 src/profiles/spending.py             # MLS wage bills 2013-2026 -> out/spending.json, out/mls_payroll.csv
    python3 src/profiles/fetch_parity.py && python3 src/profiles/parity.py   # luck share, stability, continuity vs other leagues

`parity.py` reads Transfermarkt tables from `data/tm` (see [`data/README.md`](../../data/README.md); no fetch script is included).

## Methods

- **Roles:** Gaussian mixture on z-scored style metrics within position group (k = 3, midfield k = 4). Role names are matched to cluster signatures in `build.py` (`ROLE_NAMES`); check them when running a new season.
- **Quality:** ASA goals added above average per 96, shrunk toward the group mean with a 900-minute prior.
- **Value:** season g+ above what a Huber regression of g+/96 on log10(guaranteed compensation) predicts within the group.
- **Comparables:** cosine similarity on standardized style and quality metrics within group.

The built explorer page is not committed: it embeds player-level ASA values. Build it locally with the steps above.
