# MLS Roster Investment Simulator

A roster evaluation tool for front offices. Pick a club, remove players, add MLS players or new signings, and set an extra budget. The tool shows projected points, playoff and MLS Cup probability and table rank (each with an 80% range), the best way to split the extra budget, and the club's weakest position group against the league average. It is not a 2027 forecast; offseason moves are user scenarios.

Limits stated in the tool: every minute is valued alike, so star upgrades may be undervalued (the absence study suggests elite players matter somewhat more); budget means guaranteed compensation, and salary cap, DP, TAM and GAM mechanics are not modelled.

**Live tool (v1.0, "post-2026 roster snapshot"):** https://claude.ai/artifact/ES6Evuy4YEL2YBWCpFZH9W

The simulator is separate, living software. It is versioned through the offseason ([`ROADMAP.md`](ROADMAP.md)); its updates never change Paper 2.

## Build

Run from the `mls/` folder, after Paper 2 step 10 (`src/paper2/project.py`):

    python3 src/simulator/tool_data.py      # projections, calibrated uncertainty, schedule -> out/tool/data.json
    python3 src/simulator/roster_2027.py    # rosters + comparables for signings -> out/tool/data.json
    python3 src/simulator/assemble_tool.py  # src/simulator/tool_template.html + data.json -> out/tool/index.html

`assemble_tool.py` fills every placeholder in the template from the `meta` block of `data.json`, so the page cannot disagree with its data. Rebuilding from the v1.0 `data.json` reproduces the live page byte for byte.

## What is not committed

`out/tool/data.json` and the built page contain player-level 2026 salaries from a private third-party list (`data/salaries_2026-08.csv`, read by `roster_2027.py`). Neither is in this repository; the live tool above is the published build.

## How the model is framed

The projection model is calibrated against preseason market expectations and explains where the two disagree: against the 2025 and 2026 preseason futures, model and market correlate 0.81 and 0.80, and on the 12 largest divergences the model was closer in 5 and the market in 7. Signings are valued from comparable past arrivals by salary band and age group.
