# Inverted · MLS

Inverted applies the methods that research engineers use at top European clubs to Major League Soccer, for clubs whose analytics departments have fewer resources. It is organised as a research project: questions are stated in advance, predictions are preregistered, every method is checked against outside evidence, and results that go against the hypothesis are reported as they are.

The findings are descriptive. They evaluate how MLS's roster design shapes the league; they do not argue for or against it.

**Author:** Brody Armstrong

## Papers

| | Question | Status |
|---|---|---|
| **Paper 1** | [Is MLS Built to Be Random?](https://claude.ai/artifact/4ZB9UUSKRT5SiMvQjafJb3) | Complete |
| **Paper 2** | [Why MLS Teams Stay So Close Together](https://claude.ai/artifact/YD67sqzPxspeE2gWBid1sd) · [executive summary](https://claude.ai/artifact/LFfY6yRu5SnJztbhpKCHXU) | Complete, frozen v1.0 (2026-11-12) |
| **Paper 3** | Which roster rules cause the compression: natural experiments on allocation money, Designated Player rule changes, the U22 Initiative and expansion | Not started; will be preregistered |

## Results in brief

**Paper 1.** Most of MLS's extra unpredictability comes from how close the clubs are in quality (about 92% of the excess randomness), not from playoffs or the schedule.

**Paper 2.** MLS parity is better explained by league-wide compression of roster quality, weaker clustering of talent and rapid erosion of competitive advantages than by a depth-first ("weak-link") mechanism.

- **Compression.** MLS clubs are much closer together in quality than clubs in comparable leagues: team-quality SD 0.28 market goals per match, against 0.42–0.71 in Belgium, the Netherlands, Portugal, Scotland and Turkey. The compression is largest at the top of the roster (MLS's top-three spread is 40% of the peer median; depth tiers about 50%).
- **Weak-link mechanism not supported.** The generalized-mean (ρ) approach failed its NBA method check, so the ρ predictions are untestable with these data. In the absence study, losing a club's best player costs the most (−0.40 xG difference per match, against +0.01 for ranks 6–11): relative marginal impact is concentrated toward elite players, in MLS and in all five peer leagues.
- **Talent clusters less.** The correlation between a club's top-three quality and its depth is 0.78 in MLS, against 0.87–0.96 in peer leagues.
- **Advantages erode fast.** Year-to-year team quality correlates 0.68 in MLS, against 0.87–0.96 in peer leagues, even though MLS clubs keep more of their minutes (67% returning, against 43–53%).
- **Salary is an imperfect predictor** of player quality, with sharply diminishing returns (MLS only; no peer-league salary data).

Preregistered predictions and verdicts (supported / not supported / inconclusive / untestable) are in [`papers/paper2/`](papers/paper2/).

## Roster Investment Simulator

The front-office tool built on Paper 2: pick a club, add or remove players, set an extra budget, and see projected points, playoff and MLS Cup probability with 80% ranges. It reproduces preseason market expectations (rank correlation 0.81 in 2025, 0.80 in 2026) and shows where the model and the market disagree.

**[Open the simulator (v1.0)](https://claude.ai/artifact/ES6Evuy4YEL2YBWCpFZH9W)** · [code and roadmap](src/simulator/)

## Other work in this repository

- [Player profiles](src/profiles/): data-driven roles, comparables and value for every MLS player ([explorer](https://claude.ai/artifact/2JhkoRfPcLL8d78G2Buej6)); team analysis; spending; parity.
- [USL → MLS pathway](src/pathway/): how lower-division production translates to MLS.
- [Designated Players, 2013–2026](src/dp/): what clubs get from DPs ([article](https://claude.ai/artifact/NZZtsuHn4o933qZYLpwYP1)).
- [San Diego FC: 2026 Squad Review and Recruitment Targets](https://claude.ai/artifact/UtDDJaWrXP8HURR8sH4M1x) (club report).

## Repository layout

```
papers/            paper links; Paper 2 preregistration and results log
src/profiles/      player profiles, team analysis, spending, parity
src/pathway/       USL -> MLS translation
src/dp/            Designated Player list and analysis
src/paper1/        Paper 1: Bayesian Dixon-Coles, simulations, validation
src/paper2/        Paper 2: text xG, stints, ratings, rho, absence study, peer leagues, projection
src/simulator/     Roster Investment Simulator: data build, template, assembler, roadmap
data/              data dictionary and the few small inputs that may be redistributed
results/           small derived JSON/CSV behind the figures and checks
```

Each `src/` folder has a README with its run order.

## Reproducing

Python 3.13. From the `mls/` folder (`cd mls`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the run order in each component's README. Scripts use paths relative to the `mls/` folder (`data/...`, `out/...`), so always run them from there. Paper 1 takes about 25 minutes on two CPUs with fixed seeds. Raw data is re-fetched by the scripts (about 400 MB in total).

Not everything can be rebuilt from this repository yet: a few Paper 2 intermediates were produced by code that is still to be added. They are listed under "Known reproducibility gaps" in [`src/paper2/README.md`](src/paper2/README.md).

## Data

No raw or private data is committed. [`data/README.md`](data/README.md) lists every source (American Soccer Analysis, ESPN, football-data.co.uk, Basketball-Reference, Transfermarkt, Wikipedia, Open-Meteo and others), how the scripts fetch it, and what is not redistributed. Player-level salary data is kept out of the repository.

## Licence

Code: [MIT](../LICENSE). Papers, documentation and results: [CC BY 4.0](../LICENSE-TEXT.md). Third-party data keeps its owners' terms.
