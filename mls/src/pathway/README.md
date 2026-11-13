# Pathway: USL → MLS translation

How production in the USL Championship, USL League One and MLS NEXT Pro carries over to MLS. Needs the ASA data fetched by `src/profiles/fetch.py` and `src/profiles/fetch_multi.py`. Run from the `mls/` folder:

    python3 src/pathway/pathway_data.py      # one long table of player-seasons across MLS, USLC, USL1, MLS NEXT Pro
    python3 src/pathway/pathway_model.py     # translation factors per stat and league, with intervals -> out/pathway_model.json
    python3 src/pathway/pathway_project.py   # chain USL1 -> USLC -> MLS, test the chain on direct moves, project 2026 players
    python3 src/pathway/export_pathway.py    # explorer data -> out/pathway.json

Pairs are a lower-league season followed by an MLS season in the same or next calendar year, each above a minutes threshold (see `pathway_model.py`). Results: [`results/pathway_model.json`](../../results/pathway_model.json), [`results/pathway_chain.json`](../../results/pathway_chain.json), [`results/pathway.json`](../../results/pathway.json).
