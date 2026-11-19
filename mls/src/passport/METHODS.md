# Timbers2 readiness passport — methods note (18 Nov 2026)

Question: which Timbers2 players are ready for first-team minutes, and how sure are we? Rules: PREREGISTRATION.md, written and committed locally at 16:23 PT on 18 Nov 2026, before outcome data was fetched; first public at push (time added after the push). A local commit's date is set by whoever makes the commit, and the unpushed commits were later rewritten (changing their hashes), so the push is the only timestamp anyone else can check, and it came after the outcome data was fetched. PREREGISTRATION.md itself says the commit timestamp is the record; that overstates it, and the file is left as written. Data: American Soccer Analysis public API; thanks to ASA. Rebuild everything with `bash run.sh <work_dir> <site_repo> <private_dir>` (fetch → DuckDB → models → page → 14 pytest checks). Players are joined by ASA player ID throughout, including the club roster-status file; names are used only for display.

## What did not work or was not supported
1. **Official birth dates could not be collected.** MLS and MLS NEXT Pro roster pages don't publish them in usable form, and the API behind them is undocumented (not used). ASA birth dates were used instead, a deviation approved by J before the pre-registration. Three regulars' dates appear in club signing announcements (Cervantes, Barjolo, Griffith); all three match ASA.
2. **Contract outcomes were not collected league-wide.** Only the Timbers2 regulars' end-of-2026 roster status was recorded, from the club's announcement. It is context on the page; no model uses it.
3. **Outcome (b) is not supported.** For the 48 test players who reached 500 MLS minutes, the new projection of MLS g+ per 96 was less accurate than the published pathway model (RMSE 0.066 vs 0.061). The pathway model was fitted on move-ups that likely include these players, which favours it. The page keeps the pathway projection.
4. **g+ interrupting did not meet the reliability threshold** (split-half reliability 0.18; threshold 0.40). The page shows it greyed out with its number; no model uses it.
5. **Among players with no MLS minutes yet, the edge is smaller (exploratory, not pre-registered).** The model uses MLS minutes already given, so part of its full-cohort advantage could come from players already in MLS staying there. On the 278 test players with zero career MLS minutes (21 reached 500), it still scored best on both measures (log loss 0.232 vs 0.257 raw percentile and 0.262 minutes only; Brier 0.065 vs 0.068 and 0.069). Against minutes only, the 90% interval for the log-loss difference excludes zero (−0.050 to −0.010); against the raw percentile it does not (−0.047 to +0.002). For these players the model is ahead of the minutes baseline but not clearly ahead of the raw percentile, so the strongest claim, that it beats both for players not yet in MLS, is not established.
6. **ASA caps responses at 10,000 rows**, so a league-wide match-level request is silently truncated. Data is fetched one team-season at a time, and a test fails if any response reaches the cap.
7. **Corrections made after outcome data was fetched** (none changes a rule, threshold, cohort or covariate):
   - The first interval bootstrap also drew noisy g+ values for the training players. That weakened the fitted g+ effect and left point estimates at the top of their own intervals. Now only the predicted player's own g+ is drawn from its posterior; afterwards, 0 of 356 estimates fall outside their interval. The point estimates, backtest and selection were unchanged.
   - Two tests assumed no match runs past 130 minutes; playoff games with extra time do (up to 139). The tests now treat regular-season and playoff games separately. Playoff games were already excluded from the models.
   - Roster status was first matched to the wrong "Daniel Nunez" (two players share the name). The roster-status file now carries each player's ASA player ID, and the build checks every ID is a 2026 Timbers2 player.
   - The exploratory check above first drew from the same random stream as the final interval bootstrap, which moved some passport intervals by a point. It now has its own stream; a test checks the passports' 2027 figures equal the frozen list exactly.
8. **Implementation choices not spelled out in the pre-registration** (set in code before the first model run): regular-season matches only; ages at 1 July of the season; prior variance floored at 5% of the observed variance; player-seasons without an ASA birth date dropped (55 across 2022–26).

## What worked
**Reliability** (odd vs even matches, 545 player-seasons, 2022–23, corrected to a full season): xG 0.89, xA 0.77, g+ receiving 0.65, g+ shooting 0.65, g+ passing 0.61, g+ total 0.60, g+ dribbling 0.45, g+ interrupting 0.18 (did not meet the threshold; not used).

**Readiness backtest, outcome (a)**: 500+ MLS minutes within two seasons. Train: 2022–23 cohorts (233 + 329 player-seasons). Test: 2024 cohort (370; 13.0% reached 500 minutes).

| Model | Brier | Log loss | Mean predicted |
| --- | --- | --- | --- |
| Readiness model | 0.084 | 0.283 | 12.3% |
| Raw NEXT Pro percentile | 0.104 | 0.347 | 12.9% |
| Minutes only | 0.100 | 0.336 | 12.0% |

The readiness model beat both baselines on both scores, so under the pass rule the page uses it. Bootstrap 90% intervals for the log-loss difference exclude zero (vs raw percentile −0.098 to −0.032; vs minutes only −0.076 to −0.031). Calibration by fifths of predicted chance: 1.7% predicted vs 1.4% observed; 3.3% vs 1.4%; 5.8% vs 6.8%; 11.5% vs 17.6%; 39.5% vs 37.8%. The fourth fifth is under-predicted.

Associations in the training fit (descriptive, not causal): younger players, higher shrunk g+, more NEXT Pro minutes and MLS minutes already given all go with a higher chance in the following season; the second season's chance is lower.

**Timbers2 (14 regulars), chance of 500+ MLS minutes over 2027–28 (90% interval):**
- The group to watch: Charles Ondo 13% (4–35); Adolfo Enriquez 9% (4–16); Colin Griffith 8% (2–24). Noah Santos is at 10% (4–21), but the club says he is out of contract and won't return.
- Everyone else is 2–7%, close to or below the 13% base rate.
- At the top the model agrees with a decision the club had already made: Eric Izoita 98% (91–100), resting mostly on the 475 MLS minutes he had in 2026; the club moved him to the first team on an MLS contract in September.

**Shrinkage:** even strong T2 numbers come with wide intervals. Izoita's g+ total is 0.037 per 96, with a 90% interval of −0.008 to 0.081, despite being at the 91st percentile raw and 96th age-adjusted.

**2027 predictions:** 356 MLS NEXT Pro regulars of 2026 with under 500 career MLS minutes. P(500+ MLS minutes in 2027), with 90% intervals. Written 2026-11-19T00:46:54Z. Many of these players are 16 or 17, so the list itself is kept private; only its SHA-256 is committed (`results/passport/predictions_2027_frozen.sha256`: 7f1ac92ed9230e046c60a79b5a2bf0bc2110bb384521f488ce6e2f37c81776ce). The full list will be published when it is scored (Brier, log loss, calibration) after the 2027 season, and anyone can check it against the hash. run.sh never overwrites the private copy and stops if it no longer matches the hash. The public page shows 2027 figures only for the 14 Timbers2 regulars.

## Limits
- Reaching 500 MLS minutes partly measures coach selection, depth charts, injuries and club needs, not only ability.
- One test cohort.
- Public event data only: no tracking, physical or medical data.
- The model describes patterns in who was promoted in the past; it doesn't show what causes readiness.
- The pre-registration's first checkable timestamp is the push, after outcome data was fetched; its earlier date rests on the local commit. The 2027 hash has no such issue: its public timestamp comes well before the 2027 season.
