# Paper 2 — results log

Results in the order they were produced. Predictions and the rules for judging them are in `preregistration.md` (Addenda 1–2 were written before the results they govern).

**Wording rule (from J, 2026-11-12):** write "the weak-link mechanism was not supported", never "the weak-link hypothesis failed": only specific operationalizations were tested. The absence study shows that *relative marginal impact is concentrated toward elite players*; it does not show that lower-ranked players never matter.

## Scorecard (2026-11-12)

| # | Prediction | Verdict |
|---|---|---|
| P1 | ρ < 1 in MLS | Untestable (ρ approach not validated; calibrated set −0.5 to 4) |
| P2 | ρ < 1 in peer leagues | Untestable (same reason; not run) |
| P3 | NBA strong-link (method check) | **Fail** → ρ approach not validated |
| P4 | Absence costs more per point at ranks 6–11 than rank 1 | **Not supported** (reverse in realized xG; market: same direction, not significant; finishing-adjusted ratings: per-point gap flat, per-absence gap holds) |
| P5 | MLS depth share of between-team differences < peers | **Not supported** (MLS 49% vs peer median 44%) |
| P6 | Points per $1M higher at ranks 4–11 than 1–3 | Supported as stated (0.41 vs 0.33), but driven by diminishing returns to salary, not weak-link |
| P7 | Rising players more likely to leave | Supported with market ratings; inconclusive with xG ratings |
| P8 | Depth tiers explain ≥ 33% of MLS-vs-peer spread gap | Criterion met (71%), but the top tier is compressed even more than depth, so the cap-on-depth mechanism it stood for is not supported |
| P9 | MLS absence slope ratio ≥ peers | Consistent (MLS 0.15, interval −0.16 to 0.53, overlaps peer median ~0.25) |

Gates: stability PASS; placebo PASS; parameter recovery FAIL (naive) / weak pass (calibrated, xG version); market check PARTIAL; NBA method check FAIL.

## 2026-11-12

### Player ratings (Stage 1)
- Plus-minus on stint xG difference (42,463 stints, 2013–2026), pulled toward a box-score prior (g+ by action, position, age; prior mapping fitted with the target season left out). Shrinkage λ = 8,000 by 5-fold CV grouped by match (out-of-sample error 4.073 vs 4.163 covariates-only).
- Controls: home +0.34 xG difference per 90; extra player +1.04; leading team −0.40; trailing team +0.45.
- **Stability gate: PASS.** Split-half correlation 0.48 (threshold 0.30). On-field part alone 0.15: most stability comes from the box-score prior.
- **Face validity: mixed.** 2022 Philadelphia #1, 2019 LAFC #2, 2022 LAFC #4, 2023 Columbus #1. 2024 Inter Miami #11 of 29 despite the best record: ASA's xG agrees (xG difference 20th); Miami scored 0.85 goals per game above xG. xG-based ratings cannot see finishing.
- **Finishing in MLS is mostly luck:** goals minus xG per shot correlates ≤ 0.06 from one season to the next (keepers 0.17). A second rating version adds the most generous finishing estimate the data allow (skill SD 0.009 per shot) plus goalkeeper shot-stopping. Messi has the top finishing estimate (+0.05 per 90).

### ρ from stints (Stage 2)
- **Naive frozen-rating estimator: FAILED parameter recovery.** It is biased toward weak-link: with true ρ = 1 it returned 0.5–0.75. Not used for P1–P3 (Addendum 2).
- **Calibrated estimator (Neyman set; true spread calibrated to the real split-half, 1.28× current ratings):**
  - xG ratings: real naive estimate 0.75; **95% set −0.5 to 4 → P1 inconclusive.** Rules out strong weak-link (ρ ≤ −1). Recovery check: covered the truth in 7 of 9 fresh simulations, no false directional verdicts (misses were "inconclusive"); low power.
  - Finishing-adjusted ratings judged against goals: real naive 0.5, set 0.5 to 3. **Recovery check FAILED** (4 of 9, false "strong-link" verdicts when the truth was neutral). Reported as not identifiable.

### NBA method check (P3)
- Ratings = Box Plus-Minus; outcome = five-man lineup net points per 100 possessions (Basketball-Reference top 20 lineups per team-season, 2014–2025; 6,105 of 7,200 lineups matched to players). Same calibrated Neyman set; BPM reliability assumed 0.7 (sensitivity 0.5 and 0.9).
- Naive estimate 0.75; calibrated 95% set **0 to 0.5** (reliability 0.5: −1 to 0; 0.9: 0.5). **P3: FAIL** at every reliability — the method does not find the NBA strong-link.
- The NBA version also fails its own recovery check (coverage 5–7 of 9, one or two false directional verdicts).
- Likely contributor: Basketball-Reference lists only each team's top 20 lineups, and lineups that perform well earn more minutes (selection on the outcome).
- **Consequence (preregistered):** the generalized-mean (ρ) approach is not validated. All ρ results — MLS (P1), peer leagues (P2) — are reported as untestable with these data. Conclusions about weak vs strong link rest on the absence study, which does not use ρ.

### Absence study (P4, placebo) — xG ratings
- Outside-cause absences of regulars (≥ 900 min): 1,778 call-ups and 627 red-card suspensions across 11,466 team-matches.
- xG difference lost per rating point (team-season and opponent-season fixed effects; bootstrap by team-season; 500-permutation placebo): rank 1: 6.3 (4.8–8.0), placebo p < 0.002; ranks 2–3: 2.8 (0.6–4.9), p 0.006; ranks 4–5: 3.8 (1.3–5.8), p 0.006; ranks 6–11: 1.0 (−0.9 to 3.1), p 0.40.
- **Placebo: PASS.**
- **P4: not supported (falsification criterion met).** Difference 5.2 (3.1–7.8).
- Count version (no rating scale): best player −0.40 xG difference per match (se 0.05); ranks 2–3 −0.14; ranks 4–5 −0.11; ranks 6–11 +0.01 (se 0.02).
- Caveats: shrinkage compresses top ratings most (inflates rank-1 per-point cost); absences measure value relative to the actual replacement, not absolute contribution.

### Market check of the absence study
- Outcome = closing market's expected goal difference (Pinnacle 1X2, margin removed, inverted through a Poisson model; 5,987 matches; correlates 0.33 with match xG difference).
- Cost per absence: best player −0.078 goals per match (se 0.021); ranks 2–3 −0.046; ranks 4–5 −0.067; ranks 6–11 −0.023 (se 0.008).
- **Verdict: PARTIAL.** Direction agrees; the market's gradient is much flatter (about a fifth of the realized-xG cost for the best player; a small detectable cost for ranks 6–11). Per-point P4 contrast not significant (0.82, −0.18 to 1.98).

### Robustness: absence study on finishing-adjusted ratings (`rating_goals`)
- Script: `src/paper2/absence.py rating_goals <goals|xg|market>` (new `goals` outcome = actual goal difference). Outputs `out/absence_rating_goals_*.json` (committed copies in `results/`).
- Depth-minus-star contrast per rating point: goals 0.6 (−4.4 to 5.6); xG 0.4 (−2.2 to 3.3); market −0.1 (−1.1 to 1.0). **The per-point gradient flattens to about zero.**
- Per absence (count version): goals — best player −0.34 (se 0.12), ranks 2–3 −0.19, ranks 4–5 −0.03, ranks 6–11 −0.06 (se 0.04); xG — −0.26 (0.06) vs 0.00 (0.02); market — −0.064 (0.020) vs −0.033 (0.008). **Best player still costs most.**
- Placebo p for rank 1: goals 0.028; xG and market < 0.002.
- **Reading:** the per-point gap (6.3 vs 1.0) depends on the rating version, consistent with the shrinkage caveat; the per-absence result is robust. P4 remains not supported under every version (no version shows depth costing more per point). Article now says so; the per-point gap is no longer presented as robust.

### Roster ranks and money, MLS (P6) — xG ratings, plain sum (ρ = 1)
- Quality to points: +1.09 points per game per unit of team quality (348 team-seasons).
- **Pay–quality curve:** rating rises with log salary at about the same rate below and above $1.2M (0.006 vs 0.007 per log-dollar): each extra dollar buys less. Mean rating: under $100k −0.002; $400–700k +0.006; $1.2–2M +0.008; $2–4M +0.016; $4M+ +0.020.
- **Money does not reliably buy a team's best player:** only 20% of teams' top-rated players earn ≥ $1.2M; only 35% of players paid ≥ $4M rank in their team's top three.
- **P6:** points per $1M per season — ranks 1–3 0.33 (0.26–0.40); ranks 4–11 0.41 (0.33–0.50); difference +0.08 (0.06–0.10). Follows from diminishing returns to salary, not a weak-link effect.

### Same pipeline for MLS and five peer leagues (market-based ratings)
- Ratings: match-level plus-minus on the closing market's expected goal difference; out-of-sample R² MLS 0.78, Belgium 0.91, Turkey 0.88, Netherlands/Portugal/Scotland 0.95. Matches linked: MLS 5,731; BEL 2,922; NED 4,206; POR 4,140; SCO 2,958; TUR 3,425.
- **Team-quality spread (SD within season, market goals per match):** MLS 0.28; Turkey 0.42; Belgium 0.48; Scotland 0.66; Portugal 0.68; Netherlands 0.71. MLS is the most compressed league (independently confirms Paper 1).
- **P5 shares (ranks 1–3 / 4–11 / rest):** MLS 21/49/30 (xG version 21/48/31); BEL 22/44/33; TUR 19/42/39; NED 23/50/27; POR 25/48/27; SCO 24/40/36. **P5 not supported** (MLS depth share above the peer median).
- **Absence cost per player lost (market):** best player MLS −0.15, BEL −0.16, TUR −0.15, NED −0.18, POR −0.11, SCO −0.11; ranks 6–11 between +0.01 and −0.05 everywhere. **Relative marginal impact concentrated toward elite players appears to be a soccer-wide pattern, not an MLS one.**
- **P9:** per-point ratio (ranks 6–11 ÷ rank 1): MLS 0.15 (−0.16 to 0.53); BEL 0.25, NED −0.05, POR 0.39, SCO 0.04, TUR 0.57. Consistent with peers.
- **Spread by tier, MLS as % of peer median:** top 3: 40%; ranks 4–11: 52%; rest: 50%; best single player: 49%. MLS is compressed in every tier, most at the top.
- **P8 (tier Shapley):** stretching each MLS tier to the peer-median spread raises MLS team-quality SD from 0.28 to 0.58 (peers 0.66). Shares of that gap: top 3 29%, ranks 4–11 42%, rest 29%; depth tiers together 71% (70–93%).
- **Clustering of talent (new):** within-season correlation between a club's top-3 quality and its ranks 4–11 quality — MLS 0.78 (0.74–0.82); peers 0.87–0.96 (BEL 0.90, TUR 0.87, NED 0.95, POR 0.96, SCO 0.92). In peer leagues the clubs with the best stars also have the best depth; in MLS markedly less so. This accounts for the remaining gap (0.58 → 0.66).

### Persistence (P7 and accounting)
- **P7:** players whose rating rose are more likely to leave their club the next offseason — market ratings: log-odds +0.19 per 0.1 rise (0.03–0.36), n 3,003, leave rate 32%; xG ratings: +0.025 per 0.01 (−0.012 to 0.062), not significant.
- **Team quality year to year (market ratings):** MLS 0.68; peers 0.87–0.96.
- **Share of minutes from players returning to the same club:** MLS 0.67; peers 0.43–0.53. MLS keeps more of its players yet its team quality persists far less: the persistence gap is not about roster churn — peer clubs reload to the same level, MLS clubs do not.

### Roster projection, futures check and simulator
- Rolling projection backtest 2016–2026: rank correlation 0.285 vs 0.222 for carrying last season forward.
- Preseason futures: model vs market correlation 0.81 (2025), 0.80 (2026); vs actual results model 0.45 / market 0.42 (2025), 0.32 / 0.31 (2026); model closer on 5 of the 12 largest divergences. Framing: calibrated against the market and explains where disagreement occurs (never "beats the market").
- Roster Investment Simulator published (80% ranges; signings valued from comparable arrivals by salary band × age group; rosters from salary list updated Mar–Sep 2026, 642 of 793 matched).

### Where it stands
- **The weak-link mechanism was not supported** in the operationalizations tested.
- **What the evidence suggests instead:** MLS team quality is compressed at every roster tier (most at the top); talent does not cluster at the same clubs the way it does in peer leagues; money buys rating with sharply diminishing returns and does not reliably buy a team's best player; and clubs that lose quality do not reload to the same level the next year, even though MLS rosters turn over less. All of these are descriptive; causal attribution to specific roster rules is Paper 3.
- Executive summary and full article draft written (2026-11-12). Headline wording: "MLS clubs are much closer together in quality than comparable leagues, with the largest compression at the top of the roster" (does not attribute the mechanism). One-sentence contribution: "MLS parity is better explained by league-wide compression of roster quality, weaker clustering of talent and rapid erosion of competitive advantages than by a depth-first (weak-link) mechanism."
- Structure frozen per J (2026-11-12): no more structural edits. Remaining: simulator roster refresh after the 2026–27 offseason; publication-quality figures and tables; hostile-reviewer pass for overreaching claims.
