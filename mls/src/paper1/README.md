# Is MLS built to be random? — reproducible code

Run from the `mls/` folder with `PYTHONPATH=src/paper1` (Python 3.13; numpy, pandas, scipy, statsmodels, jax, numpyro).

1. `src/paper1/random_fetch.py`     downloads raw data (ASA, football-data.co.uk, Open-Meteo, nflverse, MLB/NHL/ESPN APIs)
2. `src/paper1/random_matches.py`   builds out/mls_matches.pkl (match table with travel, venue, weather, odds)
3. `src/paper1/random_strength.py`  Bayesian Dixon-Coles per season (NUTS) -> out/dc_posteriors.pkl, out/dc_diagnostics.csv
4. `src/paper1/balance.py`          competitive balance across 16 leagues -> out/balance_*.csv
5. `src/paper1/random_sim.py`       Shapley decomposition + format replays -> out/sim_shapley.csv, out/sim_formats.csv
6. `src/paper1/random_extra.py`     calibrated decomposition, year-to-year split, relegation -> out/sim_*.json/csv
7. `src/paper1/random_tests.py`     rule-change DiD + placebos, home-advantage regressions, ghost games -> out/tests_*
8. `src/paper1/random_holdout.py`   2026 weekly-refit holdout vs closing odds -> out/holdout_*
9. `src/paper1/random_validate.py`  every check, pass/fail -> out/checks.csv
Shared: `src/paper1/dc_model.py`, `src/paper1/sim_core.py`. About 25 minutes on 2 CPUs. Seeds are fixed.
