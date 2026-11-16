# Inverted

An open football (soccer) analytics project combining reproducible research, player valuation, and roster simulation for clubs with smaller analytics departments.

The top clubs in Europe have research engineers who build their own player ratings, test ideas against data before acting on them, and tell decision makers how sure they can be. Inverted brings that way of working to clubs that do not have those departments, starting with Major League Soccer. Every project works the same way: questions are stated in advance, predictions are preregistered, methods are checked against outside evidence, and results that go against the hypothesis are reported as they are.

**Author:** Brody Armstrong

## Projects

| Folder | What it covers | Status |
|---|---|---|
| [`mls/`](mls/) | Major League Soccer: why the league is so unpredictable (Paper 1), why its clubs stay so close in quality (Paper 2), the Roster Investment Simulator, player profiles, the USL → MLS pathway and Designated Players | Papers 1–2 complete; simulator v1.0 live; Paper 3 planned |

**Start with [`mls/README.md`](mls/README.md)** for the motivation, findings, installation and usage.

Each project folder is self-contained, with its own README, code, data dictionary, results and `requirements.txt`. Run its scripts from inside that folder.

## Data

No raw, third-party or private data is committed. Each project's `data/README.md` lists its sources, how the scripts fetch them and what is not redistributed.

## License

Code: [MIT](LICENSE). Papers, documentation and results: [CC BY 4.0](LICENSE-TEXT.md). Third-party data keeps its owners' terms.
