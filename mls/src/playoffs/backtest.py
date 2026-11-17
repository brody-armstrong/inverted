"""Backtest: how well do regular-season ratings predict the 2023, 2024 and 2025 MLS Cup Playoffs?

For each season, ratings are fitted on that season's regular-season matches only, then used to predict every playoff
match (90-minute result) and every playoff tie (who advanced).
  Match level: log loss of the 90-minute result, against the closing odds of the same playoff match and against base rates.
  Tie level:   log loss of P(the team that actually advanced), and how often the favourite advanced.
Variants: market ratings with no time decay and with 120-, 90- and 60-day half-lives; Paper 1 Dixon-Coles on goals.
Writes out/playoffs/backtest.json and results/playoffs_backtest.json.
Usage: python3 src/playoffs/backtest.py
"""
import json, sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "src/playoffs"); sys.path.insert(0, "src/paper1")
from engine import fit_market, probs90, p_game, p_bo3, dc_draws

M = pd.read_pickle("out/playoffs/matches.pkl")
OUT = pathlib.Path("out/playoffs")


def ties(K):
    """Group a season's knockout matches into ties; label format and winner (team appearing later, or final score)."""
    K = K.sort_values("kickoff_utc")
    K["pair"] = [tuple(sorted(p)) for p in zip(K.home, K.away)]
    T = []
    for pair, g in K.groupby("pair", sort=False):
        T.append({"pair": pair, "games": g, "first": g.kickoff_utc.min(), "n": len(g)})
    T.sort(key=lambda t: t["first"])
    bo3_start = min(t["first"] for t in T if t["n"] >= 2)
    for t in T:
        g = t["games"]
        if t["n"] >= 2:
            t["fmt"] = "bo3"; t["host"] = g.iloc[0].home
        elif t["first"] < bo3_start:
            t["fmt"] = "wildcard"; t["host"] = g.iloc[0].home
        else:
            t["fmt"] = "single"; t["host"] = g.iloc[0].home
        later = K[K.kickoff_utc > g.kickoff_utc.max()]
        alive = [x for x in t["pair"] if ((later.home == x) | (later.away == x)).any()]
        if len(alive) == 1:
            t["winner"] = alive[0]
        else:                                         # MLS Cup final: decided in 90 or 120 minutes in 2023-2025
            r = g.iloc[-1]; assert r.hg != r.ag
            t["winner"] = r.home if r.hg > r.ag else r.away
    return T


def p_tie(R, t, idx):
    h = idx[t["host"]]; o = idx[[x for x in t["pair"] if x != t["host"]][0]]
    p = p_bo3(R, h, o) if t["fmt"] == "bo3" else p_game(R, h, o, t["fmt"] == "single")
    return p if t["winner"] == t["host"] else 1 - p


variants = {"market_all": None, "market_hl120": 120, "market_hl90": 90, "market_hl60": 60}
res = {v: {"match_ll": [], "tie_ll": [], "tie_fav": []} for v in list(variants) + ["results_dc"]}
base = {"match_ll_market": [], "match_ll_base": []}
for s in (2023, 2024, 2025):
    S = M[M.season == s]; R = S[~S.knockout]; K = S[S.knockout]
    teams = sorted(set(R.home) | set(R.away)); idx = {t: i for i, t in enumerate(teams)}
    end = R.kickoff_utc.max()
    br = R.assign(res=np.sign(R.hg - R.ag))
    bH, bD, bA = (br.res == 1).mean(), (br.res == 0).mean(), (br.res == -1).mean()
    for _, r in K.iterrows():
        o = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
        base["match_ll_market"].append(-np.log([r.mkt_H, r.mkt_D, r.mkt_A][o]))
        base["match_ll_base"].append(-np.log([bH, bD, bA][o]))
    T = ties(K)
    fits = {v: [fit_market(R, teams, hl, end)] for v, hl in variants.items()}
    fits["results_dc"] = dc_draws(R, teams)
    for v, draws in fits.items():
        for _, r in K.iterrows():
            o = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            p = np.mean([probs90(d, idx[r.home], idx[r.away]) for d in draws], axis=0)
            res[v]["match_ll"].append(-np.log(p[o]))
        for t in T:
            p = float(np.mean([p_tie(d, t, idx) for d in draws]))
            res[v]["tie_ll"].append(-np.log(p)); res[v]["tie_fav"].append(p > 0.5)
    print(s, "playoff matches", len(K), "ties", len(T), {f: sum(t["fmt"] == f for t in T) for f in ("wildcard", "bo3", "single")})

summary = {"seasons": [2023, 2024, 2025], "matches": len(base["match_ll_market"]), "ties": len(res["market_all"]["tie_ll"]),
           "match_logloss": {"closing_odds": float(np.mean(base["match_ll_market"])), "base_rates": float(np.mean(base["match_ll_base"]))},
           "tie_logloss_coinflip": float(np.log(2))}
for v, d in res.items():
    summary["match_logloss"][v] = float(np.mean(d["match_ll"]))
    summary.setdefault("tie_logloss", {})[v] = float(np.mean(d["tie_ll"]))
    summary.setdefault("tie_favourite_advanced", {})[v] = float(np.mean(d["tie_fav"]))
print(json.dumps(summary, indent=1))
json.dump(summary, open(OUT / "backtest.json", "w"), indent=1)
json.dump(summary, open("results/playoffs_backtest.json", "w"), indent=1)
