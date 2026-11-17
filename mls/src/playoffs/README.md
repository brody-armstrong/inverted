# 2026 MLS Cup Playoffs forecast

Every tie of the 2026 MLS Cup Playoffs, forecast two ways before the first wild-card match (19 Nov 2026): from betting lines
(the market forecast) and from team and player profiles alone (the profile forecast). The page is at `/inverted/playoffs/` on the site.

## Method: market forecast

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

## Method: profile forecast (no betting data)

- **Model.** A Poisson regression of each side's match xG (ASA, 3,604 regular-season matches, 2017–2025, without 2020) on its attacking profile and the opponent's defensive profile, with features z-scored within each season.
  - Attacking profile: g+ for per game through receiving, passing, dribbling and shooting, plus possession.
  - Defensive profile: g+ against through the same four actions, plus pressing.
  - Position units: shrunk player g+/96, weighted by minutes, for wide attack, strikers and 10s, fullbacks and centre-backs. The 2026 forecast uses minutes from the last 10 weeks.
- **Matchup test.** Seven style-matchup hypotheses were written down before analysis (project doc `claude/playoffs-profile-lens-plan.md`), with the rule "use them only if they lower held-out deviance in at least 7 of 8 seasons".
  - They improved it in 2 of 8, with a pooled loss. **The matchup effects were not supported**; every 95% interval includes zero.
  - The forecast therefore uses main effects only.
- **Backtest.** For each of the 2023–25 playoffs, the coefficients are refitted without that season.

## Results (16 Nov 2026)

| | |
|---|---|
| Most likely champion | Vancouver 26%, Nashville 20%, Inter Miami 16% |
| Rank correlation with Polymarket title odds (via MLSsoccer.com, 16 Nov) | 0.92 |
| Backtest, 2023–2025 playoffs, 87 matches: log loss | closing odds of the match 0.954 · this model 0.962 · Paper 1 Dixon-Coles on goals 0.993 · base rates 1.034 |
| Backtest, 51 ties | favourite advanced in 33 (65%); log loss 0.603 vs 0.693 for a coin flip |
| JS engine vs exact | max difference 0.2 percentage points (market), 0.1 (profile) |
| **Profile forecast** champion | Vancouver 27%, Nashville 13%, Chicago 12%, Inter Miami 11% |
| Profile backtest, 87 matches / 51 ties | match log loss 0.962 · tie log loss 0.596 · favourite advanced in 36 of 51 |
| Picks that differ between the forecasts | East semi (Chicago over Nashville), East champion (Chicago), West 4 v 5 (San Jose over Houston), West semi (FC Dallas over LAFC) |

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
python3 src/playoffs/fetch_profiles.py         # ASA team, player and match-xG data for the profile forecast
python3 src/playoffs/profile_model.py          # matchup hypothesis test -> results/playoffs_profile_test.json
python3 src/playoffs/profile_bracket.py        # profile backtest + 2026 bracket -> results/playoffs_profile_*.{csv,json}
python3 src/playoffs/market_check.py           # vs Polymarket -> results/playoffs_market_check.json
node src/playoffs/validate_js.js && node src/playoffs/validate_js.js profile   # page engine vs exact probabilities
python3 src/playoffs/assemble_page.py          # -> out/playoffs/index.html
```

The Dixon-Coles check needs `numpyro` and `jax`; the profile model needs `statsmodels`. Match data comes from American Soccer Analysis, and closing odds from football-data.co.uk.
