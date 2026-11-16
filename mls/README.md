# Inverted · MLS

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![Code: MIT](https://img.shields.io/badge/code-MIT-green) ![Text: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey) ![Paper 2: frozen v1.0](https://img.shields.io/badge/Paper%202-frozen%20v1.0-orange)

A reproducible MLS analytics project consisting of two research papers, a roster simulator, and the underlying data pipelines.

**Author:** Brody Armstrong

## Description

**Why I built this.** I have loved football (soccer) for as long as I can remember. I played until I started college, I still follow it passionately, and it has created so many great connections in my life. I studied computer science, and my favourite courses were statistics, machine learning and AI. Fusing what I learned in computer science with this passion has been fun and motivating: it lets curiosity, and a little healthy delusion, push me to be creative and build real applications that people can use.

**The problem.** The top clubs in Europe employ research engineers who build their own player ratings, test ideas against data and tell decision makers how sure they can be. Most MLS clubs have small analytics departments and cannot do that work at the same depth. MLS is also an unusual league to analyse: a salary cap, Designated Players and allocation money keep clubs closer together than almost anywhere else, so methods built for Europe do not transfer as is.

**What this project does.** It applies research-engineer methods to MLS's central puzzle, parity, and turns the findings into a tool a front office can use:

- **Paper 1** measures how random MLS really is and what causes it.
- **Paper 2** asks why MLS clubs stay so close together in quality.
- **The Roster Investment Simulator** lets a front office test roster moves and budget changes against the Paper 2 models.

**How it works.** The pipeline builds a shot-quality (xG) model from match commentary text, splits 14 seasons of matches into 42,463 stretches with a fixed set of players on the field, rates every player from those stretches, and runs the same market-based pipeline on MLS and five comparable European leagues. Every major method is checked against outside evidence before its results are read.

**What makes it different.** Predictions are written down before the data is analysed, and the checks are reported whether they pass or fail. Paper 1's validation table has 8 passes, 7 failures and 1 partial result. Paper 2's starting hypothesis, that MLS compresses team quality by limiting spending on depth, was **not supported**, and the paper says so. Every finding is descriptive: the papers evaluate how MLS's roster design shapes the league and do not argue for or against it.

**What I learned.**
- **Not answering a big question is not failure.** I could not settle a big-picture question like "is soccer a weak-link sport?" with the data and methods available. That taught me what research really is: good research does not always lead to a direct answer, but it can narrow the question.
- **Narratives are seductive.** I found myself falling in love with stories the evidence did not fully support. It took several review passes to replace strong words with precise ones that the results actually justified.
- **New methods or adapted ones.** I am still learning the balance between building a new method and adapting an existing one.
- **Curiosity is the engine.** My curiosity leads me down rabbit holes that excite me, and I truly believe we can quantify things our eyes see but struggle to explain.
- **Communicating uncertainty.** I learned a lot about stating how sure a result is, and the structure of the papers taught me how to make technical work accessible to decision makers.

**Where I want to grow.**
- **Causal inference.** Much of Paper 2 brushes up against causal questions. Frameworks such as difference-in-differences, synthetic controls, regression discontinuity and instrumental variables would help me separate descriptive patterns from causal effects, which is exactly what Paper 3 needs.
- **Hierarchical (multilevel) models.** Player, team and league effects fit these models naturally, and they could replace some of the ad hoc shrinkage choices.
- **Bayesian statistics.** Given my interest in uncertainty and simulation, deeper Bayesian modelling is a natural extension.
- **Experimental design and sensitivity analysis.** Formal methods here would strengthen future work.

## Table of contents

- [Papers](#papers)
- [Results in brief](#results-in-brief)
- [Roster Investment Simulator](#roster-investment-simulator)
- [Other work](#other-work)
- [Installation](#installation)
- [Usage](#usage)
- [Validation](#validation)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Roadmap](#roadmap)
- [Credits](#credits)
- [License](#license)

## Papers

| | Question | Status |
|---|---|---|
| **Paper 1** | [Is MLS Built to Be Random?](papers/paper1/is-mls-built-to-be-random.pdf) (PDF) | Complete |
| **Paper 2** | [Why MLS Teams Stay So Close Together](papers/paper2/why-mls-teams-stay-so-close-together.pdf) (PDF) · [executive summary](papers/paper2/executive-summary.pdf) (PDF) | Complete, frozen v1.0 (2026-11-12) |
| **Paper 3** | Which roster rules cause the compression | Not started; will be preregistered |

## Results in brief

**Paper 1 — Is MLS Built to Be Random?** Chance decides 56% of the spread in an MLS table (95% interval 47–66%), against 23% across eleven European top flights. Replaying every MLS season since 2013 with one design feature at a time set to its European value shows that the closeness of team quality accounts for about 92% of that gap. The schedule, travel and playoffs matter much less.

**Paper 2 — Why MLS Teams Stay So Close Together.** MLS parity is better explained by league-wide compression of roster quality, weaker clustering of talent and rapid erosion of competitive advantages than by a depth-first ("weak-link") mechanism.

- **Compression.** MLS clubs are much closer together in quality than clubs in comparable leagues: team-quality SD 0.28 market goals per match, against 0.42–0.71 in Belgium, the Netherlands, Portugal, Scotland and Turkey. The compression is largest at the top of the roster (MLS's top-three spread is 40% of the peer median; depth tiers about 50%).
- **The weak-link mechanism was not supported.** The generalized-mean (ρ) approach failed its NBA method check, so the ρ predictions are untestable with these data. In the absence study, losing a club's best player costs the most (−0.40 xG difference per match, against +0.01 for ranks 6–11): relative marginal impact is concentrated toward elite players, in MLS and in all five peer leagues.
- **Talent clusters less.** The correlation between a club's top-three quality and its depth is 0.78 in MLS, against 0.87–0.96 in peer leagues.
- **Advantages erode fast.** Year-to-year team quality correlates 0.68 in MLS, against 0.87–0.96 in peer leagues, even though MLS clubs keep more of their minutes (67% returning, against 43–53%).
- **Salary is an imperfect predictor** of player quality, with sharply diminishing returns (MLS only; no peer-league salary data).

All nine preregistered predictions and their verdicts (supported / not supported / inconclusive / untestable) are in [`papers/paper2/`](papers/paper2/).

## Roster Investment Simulator

The front-office tool built on Paper 2. Pick a club, remove players, add MLS players or new signings, and set an extra budget. The simulator shows projected points, playoff and MLS Cup probability and table rank, each with an 80% range, along with the best way to split the budget and the club's weakest position group.

It reproduces preseason market expectations (correlation with preseason futures 0.81 in 2025 and 0.80 in 2026) and shows where the model and the market disagree. It is a roster evaluation tool, not a 2027 forecast: offseason moves are scenarios the user sets.

**[Open the simulator (v1.1)](https://claude.ai/artifact/ES6Evuy4YEL2YBWCpFZH9W)** · [code](src/simulator/) · [validation against v1.0](src/simulator/VALIDATION.md) · [roadmap](src/simulator/ROADMAP.md)

Rosters come from the MLS Players Association's public salary release of April 16, 2026; every input is public and the build is reproducible.

## Other work

- [Player profiles](src/profiles/): data-driven roles, comparable players and value for every MLS player; team analysis, spending and parity.
- [USL → MLS pathway](src/pathway/): how production in the USL Championship, USL League One and MLS NEXT Pro translates to MLS.
- [Designated Players, 2013–2026](src/dp/): what clubs get from their DPs.

## Installation

Requires Python 3.13.

```bash
git clone https://github.com/brody-armstrong/inverted.git
cd inverted/mls
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run every script from the `mls/` folder; scripts use paths relative to it (`data/...`, `out/...`). Each component has its own README with the full run order:

| Component | Start here | Typical run |
|---|---|---|
| Paper 1 | [`src/paper1/`](src/paper1/) | `python3 src/paper1/random_fetch.py` → … → `random_validate.py` (about 25 min on 2 CPUs, fixed seeds) |
| Paper 2 | [`src/paper2/`](src/paper2/) | ten steps from `p2_fetch_espn.py` to `project.py` |
| Simulator | [`src/simulator/`](src/simulator/) | `tool_data.py` → `roster_2027.py` → `assemble_tool.py` |
| Player profiles | [`src/profiles/`](src/profiles/) | `fetch.py` → `build.py` → `export.py` → `assemble.py` |
| Pathway | [`src/pathway/`](src/pathway/) | `pathway_data.py` → `pathway_model.py` → `pathway_project.py` |
| Designated Players | [`src/dp/`](src/dp/) | `dp_parse_wikipedia.py` → `dp_build.py` → `dp_analysis.py` |

The fetch scripts download about 400 MB of raw data. The small results behind every figure are already in [`results/`](results/), so the findings can be inspected without rebuilding anything.

**Not everything can be rebuilt yet.** A few Paper 2 intermediates were produced by code that is still to be added; they are listed under "Known reproducibility gaps" in [`src/paper2/README.md`](src/paper2/README.md).

## Validation

This project uses validation checks rather than a unit-test suite: each one tests a model against outside evidence and is reported as pass or fail.

- **Paper 1** ([`results/checks.csv`](results/checks.csv)): sampler convergence, an equal-teams league, luck-share benchmarks against other sports, a 2026 holdout against closing odds, and placebo tests for rule changes. 8 pass, 7 fail, 1 partial.
- **Paper 2** ([`papers/paper2/results-log.md`](papers/paper2/results-log.md)): rating stability (pass), placebo absences (pass), parameter recovery on simulated leagues (naive estimator fail; calibrated estimator weak pass), market check of the absence study (partial), NBA method check (fail), and face validity on known elite teams (mixed, reported).
- **Simulator:** the 2016–2026 rolling projection backtest beats carrying last season forward (rank correlation 0.285 against 0.222), and the 2025 and 2026 preseason projections are compared with the futures market.
- **Simulator rebuild:** v1.1 was rebuilt from public data and checked against v1.0 with thresholds set in advance ([`src/simulator/VALIDATION.md`](src/simulator/VALIDATION.md)); every pipeline check passes.

## Reproducibility

Every published result can be traced to public data and to code in this repository.

**Data sources**
- [American Soccer Analysis](https://www.americansocceranalysis.com/): player, team and match data, and the MLSPA salary releases
- ESPN: match lineups, events and commentary
- Public odds data: closing odds from [football-data.co.uk](https://www.football-data.co.uk/)
- MLS Players Association public salary releases

**Unavailable features**

The public data does not include these, so the simulator does not model them:
- contract end dates
- option years
- private roster metadata

**Validation**

The simulator was rebuilt from public data only and validated against the original development build ([`src/simulator/VALIDATION.md`](src/simulator/VALIDATION.md)):
- player ratings reproduce to 13 decimal places
- projection-model coefficients reproduce exactly
- roster sensitivity is documented club by club
- simulation noise is quantified

Historical prototype data was removed from the public repository due to source restrictions.

## Repository layout

```
papers/            Paper 1 and Paper 2 PDFs; Paper 2 preregistration and results log
src/profiles/      player profiles, team analysis, spending, parity
src/pathway/       USL -> MLS translation
src/dp/            Designated Player list and analysis
src/paper1/        Paper 1: Bayesian Dixon-Coles, simulations, validation
src/paper2/        Paper 2: text xG, stints, ratings, rho, absence study, peer leagues, projection
src/simulator/     Roster Investment Simulator: data build, template, assembler, roadmap
data/              data dictionary and the few small inputs that may be redistributed
results/           small derived JSON/CSV behind the figures and checks
```

## Data

No raw or private data is committed. [`data/README.md`](data/README.md) lists every source, how the scripts fetch it, and what is not redistributed. Player-level salary data is kept out of the repository.

## Roadmap

- **Paper 3:** which roster rules cause the compression. Natural experiments on allocation money, Designated Player rule changes, the U22 Initiative and expansion; preregistered before any analysis.
- **Simulator:** winter roster updates as offseason moves are announced, a 2027 preseason edition, and new scenarios for club options, trades, injuries and the expansion draft ([roadmap](src/simulator/ROADMAP.md)).
- **Reproducibility:** add the remaining Paper 2 code and a small sample dataset so the pipeline can run end to end without fetching everything.

## Credits

This work depends on public data from:

- [American Soccer Analysis](https://www.americansocceranalysis.com/): xG, goals added and MLSPA salary releases
- ESPN: match lineups, events and commentary
- [football-data.co.uk](https://www.football-data.co.uk/): closing odds
- [Basketball-Reference](https://www.basketball-reference.com/): NBA ratings and lineups for the method check
- Transfermarkt: market values and transfers
- Wikipedia, "Designated Player Rule": DP histories
- [Open-Meteo](https://open-meteo.com/), nflverse and the MLB, NHL and NBA public APIs: weather, elevation and other-sport standings

## License

Code: [MIT](../LICENSE). Papers, documentation and results: [CC BY 4.0](../LICENSE-TEXT.md). Third-party data keeps its owners' terms; see [`data/README.md`](data/README.md).
