"""Profile bracket: the 2026 playoffs from team and player profiles only (no betting lines), plus its 2023-2025 backtest.

Model: main effects of src/playoffs/profile_model.py. The matchup interactions were tested and not supported.
  Team action profiles: full regular season. Position units: players weighted by minutes in the last 10 weeks.
  Uncertainty: 200 draws of the coefficients from their quasi-Poisson covariance.
Backtest: for each of 2023-2025, the coefficients are fitted without that season, and its playoffs are predicted
from its own regular-season profiles.
Writes results/playoffs_profile_2026.csv, results/playoffs_profile_backtest.json, out/playoffs/profile_bracket.json.
Usage: python3 src/playoffs/profile_bracket.py
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src/playoffs")
import profile_model as pm
import simulate as sim
from engine import probs90
from backtest import ties, p_tie


def draws_for(D, Z, idx, n=200, seed=0):
    m = pm.fit_main(D)
    rng = np.random.default_rng(seed)
    out = []
    for p in rng.multivariate_normal(m.params.values, m.cov_params().values, size=n):
        r = pm.ratings_from_fit(pd.Series(p, index=m.params.index), Z)
        out.append({"c": r["c"], "home": r["home"], "att": np.array([r["att"][t] for t in idx]),
                    "def": np.array([r["def"][t] for t in idx])})
    return out, m


# ---------- backtest on the 2023-2025 playoffs ----------
M = pd.read_pickle("out/playoffs/matches.pkl")
ll_m, ll_t, fav = [], [], []
for s in (2023, 2024, 2025):
    D = pm.training([x for x in pm.SEASONS if x != s])
    Z, _ = pm.team_features(s, window=True)
    S = M[M.season == s]; K = S[S.knockout]
    teams = sorted(Z.index); idx = {t: i for i, t in enumerate(teams)}
    dr, _ = draws_for(D, Z.loc[teams], teams, n=100, seed=s)
    for _, r in K.iterrows():
        o = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
        p = np.mean([probs90(d, idx[r.home], idx[r.away]) for d in dr], axis=0)
        ll_m.append(-np.log(p[o]))
    for t in ties(K):
        p = float(np.mean([p_tie(d, t, idx) for d in dr]))
        ll_t.append(-np.log(p)); fav.append(p > 0.5)
bt = {"matches": len(ll_m), "ties": len(ll_t), "match_logloss": float(np.mean(ll_m)), "tie_logloss": float(np.mean(ll_t)),
      "tie_favourite_advanced": float(np.mean(fav)), "tie_favourite_advanced_n": int(np.sum(fav))}
print("backtest", bt)
json.dump(bt, open("results/playoffs_profile_backtest.json", "w"), indent=1)

# ---------- 2026 ----------
D = pm.training()
Z, F = pm.team_features(2026, window=True)
teams = sim.teams
dr, m = draws_for(D, Z.loc[teams], teams, n=200, seed=2026)
P, acc = sim.run(dr)
for conf, names in sim.SEEDS.items():
    assert abs(sum(P[sim.by_name[n]]["final"] for n in names) - 1) < 1e-9

# what drives each team's profile rating: contribution of each feature (point estimate), for the matchup breakdown
par = m.params
contrib = {}
for t in teams:
    a = {c: float(par["A_" + c] * Z.at[t, c]) for c in pm.ATT}
    d = {c: float(-par["B_" + c] * Z.at[t, c]) for c in pm.DEF}     # positive = harder to create against
    contrib[t] = {"att": a, "def": d}

mk = pd.read_csv("results/playoffs_2026.csv").set_index("abbr")
rows = []
for conf, names in sim.SEEDS.items():
    for k, n in enumerate(names):
        t = sim.by_name[n]; ab = sim.TN[t]["team_abbreviation"]
        rows.append({"conf": conf, "seed": k + 1, "team": n, "abbr": ab,
                     **{f"p_{s}": P[t][s] for s in sim.STAGES}, "p_champ_market": mk.at[ab, "p_champ"],
                     "att": sum(contrib[t]["att"].values()), "def": sum(contrib[t]["def"].values())})
R = pd.DataFrame(rows)
R.to_csv("results/playoffs_profile_2026.csv", index=False, float_format="%.4f")
print(R[["conf", "seed", "abbr", "att", "def", "p_sf", "p_cf", "p_final", "p_champ", "p_champ_market"]].round(3).to_string(index=False))

ab = lambda t: sim.TN[t]["team_abbreviation"]
out = {"probs": {r["abbr"]: {s: round(r[f"p_{s}"], 4) for s in sim.STAGES} for r in rows},
       "draws": [{"c": round(d["c"], 4), "h": round(d["home"], 4),
                  "a": {ab(t): round(float(d["att"][i]), 4) for i, t in enumerate(teams) if ab(t) in {r["abbr"] for r in rows}},
                  "d": {ab(t): round(float(d["def"][i]), 4) for i, t in enumerate(teams) if ab(t) in {r["abbr"] for r in rows}}}
                 for d in dr],
       "contrib": {ab(t): {k: {c: round(v, 4) for c, v in x.items()} for k, x in contrib[t].items()}
                   for t in teams if ab(t) in {r["abbr"] for r in rows}},
       "raw": {ab(t): {c: round(float(F.at[t, c]), 4) for c in F.columns} for t in teams if ab(t) in {r["abbr"] for r in rows}},
       "home_goals_mult": round(float(np.exp(par["home"])), 3),
       "point": {"c": round(float(par["const"]), 4), "h": round(float(par["home"]), 4)}}
json.dump(out, open("out/playoffs/profile_bracket.json", "w"), separators=(",", ":"))
