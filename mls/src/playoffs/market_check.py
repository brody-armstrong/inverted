"""Compare the model's MLS Cup odds with prediction-market title odds -> results/playoffs_market_check.json.

Market: Polymarket MLS Cup 2026 winner odds as listed in MLSsoccer.com's "MLS Cup 2026 odds: Who's the favorite to win it all?"
(https://www.mlssoccer.com/playoffs/2026/news/mls-cup-2026-odds-who-s-the-favorite-to-win-it-all), retrieved 2026-11-16.
The article gives whole percentages; Real Salt Lake is listed as "<1%" and is entered as 0.5 for the rank correlation.
Usage: python3 src/playoffs/market_check.py
"""
import json
import pandas as pd
from scipy.stats import spearmanr

PM = {"NSH": 24, "MIA": 18, "VAN": 16, "PHI": 7, "STL": 6, "FCD": 5, "LAFC": 5, "CHI": 4, "NER": 4, "HOU": 3, "SJE": 3,
      "CLT": 2, "CLB": 1, "POR": 1, "CIN": 1, "ORL": 1, "SEA": 1, "RSL": 0.5}
D = pd.read_csv("results/playoffs_2026.csv")
D["pm"] = D.abbr.map(PM)
out = {"source": "Polymarket, as listed on MLSsoccer.com (MLS Cup 2026 odds article), retrieved 2026-11-16", "pct": PM,
       "spearman_model": round(float(spearmanr(D.p_champ, D.pm).statistic), 2),
       "spearman_results_dc": round(float(spearmanr(D.p_champ_dc, D.pm).statistic), 2)}
json.dump(out, open("results/playoffs_market_check.json", "w"), indent=1)
print(out["spearman_model"], out["spearman_results_dc"])
