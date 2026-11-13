"""Export pathway results (translation table, survival, 2026 prospects with MLS comparables, case studies)."""
import json
import numpy as np, pandas as pd

M = json.load(open("out/pathway_model.json"))
CH = json.load(open("out/pathway_chain.json"))
L = M["labels"]
STATS = list(L)
STYLE = ["xg_96", "shots_96", "xa_96", "kp_96", "passes_96", "recv_96", "drib_96", "int_96", "share_team_touches"]
cur = pd.read_pickle("out/pathway_2026.pkl")
allp = pd.read_pickle("out/all_seasons.pkl")
mls26 = pd.read_csv("out/players_2026.csv")
mls_idx = {pid: i for i, pid in enumerate(mls26.player_id)}

# MLS comparables for each prospect: cosine on projected style, z-scored against MLS 2026 group
comps = []
for _, r in cur.iterrows():
    pool = mls26[mls26.group == r.group]
    if pool.empty:
        comps.append([]); continue
    mu, sd = pool[STYLE].mean(), pool[STYLE].std().replace(0, 1)
    Z = ((pool[STYLE] - mu) / sd).values
    z = ((np.array([r[f"proj_{s}"] for s in STYLE]) - mu.values) / sd.values)
    sim = Z @ z / (np.linalg.norm(Z, axis=1) * np.linalg.norm(z) + 1e-9)
    top = np.argsort(-sim)[:3]
    comps.append([[mls_idx[pool.player_id.iloc[j]], round(float(sim[j]), 2)] for j in top])
cur["comps"] = comps

nn = lambda v, k=3: None if pd.isna(v) else round(float(v), k)
prospects = [{
    "n": r.player_name, "lg": r.league, "t": r.team or "", "p": r.general_position, "g": r.group,
    "a": nn(r.age, 1), "m": int(r.minutes_played), "gp": nn(r.gplus_96), "pgp": nn(r.proj_gplus_96),
    "pct": None if pd.isna(r.mlspct_gplus_96) else int(r.mlspct_gplus_96),
    "pxg": nn(r.proj_xg_96, 2), "pxa": nn(r.proj_xa_96, 2), "xg": nn(r.xg_96, 2), "xa": nn(r.xa_96, 2),
    "sp": [None if pd.isna(r[f"mlspct_{s}"]) else int(r[f"mlspct_{s}"]) for s in STYLE],
    "mm": int(r.mls_min_2026), "c": r.comps} for _, r in cur.iterrows()]

table = {}
for lg, res in M["results"].items():
    rows = []
    for s in STATS:
        v = res["stats"][s]
        rows.append({"stat": L[s], "factor": v.get("factor"), "ci": v.get("ci"),
                     "skill": None if lg == "usl1" else round(v["skill"], 3), "corr": round(v["corr"], 2)})
    table[lg] = {"n": res["n"], "players": res["players"], "rows": rows,
                 "survival": res["survival_rate"], "surv_q": res["survival_by_gplus_quintile"]}

# Case studies: move-ups, predicted vs actual MLS g+/96
P = pd.read_pickle("out/pathway_pairs.pkl")
P = P[P.league_lo != "usl1"].dropna(subset=["gplus_96_lo", "gplus_96_mls"])
def pred(row):
    m = M["models"][row.league_lo]; c = m["coef"]["gplus_96"]; g = m["groups"]
    return c[0] + c[1] * row.gplus_96_lo + sum(c[2 + i] * (row.group == gg) for i, gg in enumerate(g[1:]))
P["pred"] = P.apply(pred, axis=1)
P["q"] = pd.qcut(P.pred.rank(method="first"), 4, labels=False)
tiers = P.groupby("q").agg(n=("pred", "size"), pred=("pred", "mean"), actual=("gplus_96_mls", "mean")).round(3).reset_index().to_dict("records")
cases = [{"n": r.player_name_lo, "lg": r.league_lo, "s": int(r.season), "g": r.group, "lo": nn(r.gplus_96_lo),
          "pr": nn(r.pred), "ac": nn(r.gplus_96_mls), "mm": int(r.minutes_played_mls)}
         for _, r in P.sort_values("gplus_96_mls", ascending=False).drop_duplicates("player_id").head(12).iterrows()]

out = {"table": table, "chain": {"n": CH["n_direct"], "pairs": CH["usl1_uslc_pairs"],
       "test": {L[s]: v for s, v in CH["chain_test"].items()}},
       "tiers": tiers, "cases": cases, "prospects": prospects,
       "style_labels": [L[s] for s in STYLE], "league_names": M["league_names"], "thresholds": M["thresholds"]}
json.dump(out, open("out/pathway.json", "w"), separators=(",", ":"))
print(len(prospects), "prospects |", round(len(json.dumps(out)) / 1024), "KB")
