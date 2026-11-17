# 2026 MLS Cup Playoffs forecast

Every tie of the 2026 MLS Cup Playoffs, forecast before the first wild-card match (19 Nov 2026). The page is at
`/inverted/playoffs/` on the site.

## Method

- **Ratings.** Closing odds of every 2026 regular-season match (Pinnacle, market average where Pinnacle is missing, margin
  removed) are inverted to home and away Poisson scoring rates, the same conversion Paper 2 uses. A weighted least-squares fit gives each club an
  attack and a defence rating plus a league-wide home advantage (90-day half-life, chosen before the backtest). 200 match
  bootstraps carry the uncertainty in the ratings. This is the scoring model from Paper 1.
- **Rules.** Wild card: one game, then penalties. Round One: best of three (higher, lower, higher seed), each draw goes to penalties.
  Conference semifinals and finals: one game at the higher seed, with extra time and then penalties. MLS Cup: hosted by the
  finalist with the better record. Extra time is scored at a third of the 90-minute rate, and shootouts are 50/50.
- **Exact.** The bracket probabilities are computed exactly for each rating draw, then averaged.
  `engine.js`, which runs the page's "play it out" button, is checked against them on 400,000 brackets.
- **Seeds.** The seeds are the official ones; `simulate.py` asserts that they agree with the points table built from the ASA data.

## Results (16 Nov 2026)

| | |
|---|---|
| Most likely champion | Vancouver 26%, Nashville 20%, Inter Miami 16% |
| Rank correlation with Polymarket title odds (via MLSsoccer.com, 16 Nov) | 0.92 |
| Backtest, 2023–2025 playoffs, 87 matches: log loss | closing odds of the match 0.954 · this model 0.962 · Paper 1 Dixon-Coles on goals 0.993 · base rates 1.034 |
| Backtest, 51 ties | favourite advanced in 33 (65%); log loss 0.603 vs 0.693 for a coin flip |
| JS engine vs exact | max difference 0.2 percentage points |

Limits: the ratings cannot see injuries, suspensions or call-ups. Shootouts are coin flips, every club gets the same home
advantage, and because the ratings are built on betting lines the model is calibrated against the market rather than independent of it.

## Build (from `mls/`)

```bash
python3 src/simulator/fetch_inputs.py          # football-data.co.uk MLS odds (and ASA inputs)
for s in 2023 2024 2025 2026; do curl -s "https://app.americansocceranalysis.com/api/v1/mls/games?season_name=$s" -o data/raw/mls_games_$s.json; done
curl -s https://app.americansocceranalysis.com/api/v1/mls/teams -o data/raw/teams.json
python3 src/playoffs/data.py                   # match table with market scoring rates -> out/playoffs/matches.pkl
python3 src/playoffs/backtest.py               # 2023-2025 playoffs -> results/playoffs_backtest.json
python3 src/playoffs/simulate.py               # 2026 bracket -> results/playoffs_2026.csv, out/playoffs/bracket.json
python3 src/playoffs/market_check.py           # vs Polymarket -> results/playoffs_market_check.json
node src/playoffs/validate_js.js               # page engine vs exact probabilities
python3 src/playoffs/assemble_page.py          # -> out/playoffs/index.html
```

The Dixon-Coles check needs `numpyro` and `jax`. Match data comes from American Soccer Analysis, and closing odds from football-data.co.uk.
