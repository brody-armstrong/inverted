"""Compare the rebuilt simulator with Simulator v1.0 (see VALIDATION_PROTOCOL.md).

A6  per-player rating component that does not depend on the roster source (projection minus salary and age terms)
A7  v1.0 rosters held fixed, player ratings from the rebuilt pipeline -> team strength and projected points (page engine)
B   rebuilt (MLSPA) rosters vs v1.0 rosters -> team strength and projected points, roster differences

Usage: python3 src/simulator/validation/compare_v1.py <v1 data.json> <v1 engine json> <new engine json>
Writes out/validation/*.csv and out/validation/summary.json; needs node for A7 (src/simulator/validation/engine.js).
"""
import json, re, subprocess, sys, unicodedata, pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr

V1, E1, E2 = sys.argv[1:4]
OUT = pathlib.Path("out/validation"); OUT.mkdir(parents=True, exist_ok=True)
coef = np.array(json.load(open("out/proj_model.json"))["coef"])
v1 = json.loads(re.sub(r"\bNaN\b", "null", open(V1).read()))
P = pd.read_pickle("out/proj_players_2027.pkl")


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]", " ", n).split())


def sal_age_terms(sal, age, new):
    ls = np.log(max(sal, 5e4)); ag = ((age if age is not None else 26) - 27) / 5
    return coef[4] * ls + coef[5] * ls * new + coef[6] * ag + coef[7] * ag ** 2


# ---------- A6 ----------
rows = []
for ti, t in enumerate(v1["teams"]):
    for p in t["players"]:
        rows.append({"ti": ti, "team": t["name"], "n": p["n"], "nm": norm(p["n"]), "sal": p["sal"], "age": p["age"], "new": p["new"],
                     "r_v1": p["r"], "base_v1": p["r"] - sal_age_terms(p["sal"], p["age"], p["new"])})
A = pd.DataFrame(rows)
P["nm"] = P.name.map(norm)
P["base_new"] = coef[0] + coef[1] * P.p1 + coef[2] * P.p2 + coef[3] * P.new
Pu = P.drop_duplicates("nm", keep=False)                       # names unique in the rebuild
J = A.merge(Pu[["nm", "base_new", "new", "club", "sal", "age"]].rename(columns={"new": "new_r", "sal": "sal_r", "age": "age_r", "club": "club_r"}),
            on="nm", how="left")
m = J.base_new.notna(); same = m & (J.new == J.new_r)
d = (J.base_new - J.base_v1)[same]
a6 = {"v1_players": int(len(J)), "matched_by_name": int(m.sum()), "matched_same_history_flag": int(same.sum()),
      "corr": float(np.corrcoef(J.base_v1[same], J.base_new[same])[0, 1]), "median_abs_diff": float(d.abs().median()),
      "p95_abs_diff": float(d.abs().quantile(.95)), "max_abs_diff": float(d.abs().max()),
      "matched_but_history_flag_differs": int((m & ~same).sum())}
a6["material"] = bool(a6["corr"] < 0.99 or a6["median_abs_diff"] > 0.002)
J.to_csv(OUT / "a6_players.csv", index=False)

# ---------- A7: v1 rosters, rebuilt ratings for players matched with the same history flag ----------
over = {}
for ti, t in enumerate(v1["teams"]):
    g = J[J.ti == ti]; pl = []
    for p, (_, r) in zip(t["players"], g.iterrows()):
        q = dict(p)
        if r.base_new == r.base_new and r.new == r.new_r:
            q["r"] = float(r.base_new + sal_age_terms(p["sal"], p["age"], p["new"]))
        pl.append(q)
    over[ti] = pl
json.dump(over, open(OUT / "a7_rosters.json", "w"))
a7_eng = json.loads(subprocess.run(["node", "src/simulator/validation/engine.js", V1, str(OUT / "a7_rosters.json")], capture_output=True, text=True, check=True).stdout)
e1 = pd.DataFrame(json.load(open(E1))); e7 = pd.DataFrame(a7_eng); e2 = pd.DataFrame(json.load(open(E2)))
T7 = e1[["name", "Q", "pts"]].merge(e7[["name", "Q", "pts"]], on="name", suffixes=("_v1", "_a7"))
T7["dQ"] = T7.Q_a7 - T7.Q_v1; T7["dpts"] = T7.pts_a7 - T7.pts_v1
a7 = {"max_abs_dpts": float(T7.dpts.abs().max()), "mean_abs_dpts": float(T7.dpts.abs().mean()), "max_abs_dQ": float(T7.dQ.abs().max())}
a7["material"] = bool(a7["max_abs_dpts"] > 1.0 or a7["mean_abs_dpts"] > 0.5)
T7.to_csv(OUT / "a7_teams.csv", index=False)

# ---------- B: roster source ----------
TB = e1[["name", "n", "Q", "pts", "lo", "hi", "po", "cup"]].merge(e2[["name", "n", "Q", "pts", "lo", "hi", "po", "cup"]], on="name", suffixes=("_v1", "_new"))
TB["dQ"] = TB.Q_new - TB.Q_v1; TB["dpts"] = TB.pts_new - TB.pts_v1
TB["rank_v1"] = TB.pts_v1.rank(ascending=False).astype(int); TB["rank_new"] = TB.pts_new.rank(ascending=False).astype(int)
newd = json.load(open("out/tool/data.json"))
only_v1, only_new = [], []
for t1 in v1["teams"]:
    t2 = next(t for t in newd["teams"] if t["name"] == t1["name"])
    s1 = {norm(p["n"]): p for p in t1["players"]}; s2 = {norm(p["n"]): p for p in t2["players"]}
    only_v1 += [{"team": t1["name"], "player": p["n"], "r": p["r"], "sal": p["sal"]} for k, p in s1.items() if k not in s2]
    only_new += [{"team": t1["name"], "player": p["n"], "r": p["r"], "sal": p["sal"]} for k, p in s2.items() if k not in s1]
OV, ON = pd.DataFrame(only_v1), pd.DataFrame(only_new)
OV.to_csv(OUT / "b_only_v1.csv", index=False); ON.to_csv(OUT / "b_only_new.csv", index=False)
TB.to_csv(OUT / "b_teams.csv", index=False)
b = {"spearman_pts": float(spearmanr(TB.pts_v1, TB.pts_new).statistic), "mean_abs_dpts": float(TB.dpts.abs().mean()),
     "max_abs_dpts": float(TB.dpts.abs().max()), "players_v1": int(sum(len(t["players"]) for t in v1["teams"])),
     "players_new": int(sum(len(t["players"]) for t in newd["teams"])), "only_in_v1": int(len(OV)), "only_in_new": int(len(ON))}
summary = {"A6": a6, "A7": a7, "B": b}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ---------- B decomposition: correct v1.0's history flags (player-ID matching) with v1.0 rosters held fixed ----------
over2 = {}
for ti, t in enumerate(v1["teams"]):
    g = J[J.ti == ti]; pl = []
    for p, (_, r) in zip(t["players"], g.iterrows()):
        q = dict(p)
        if r.base_new == r.base_new:                     # matched: rebuilt rating component and the rebuild's history flag
            q["r"] = float(r.base_new + sal_age_terms(p["sal"], p["age"], int(r.new_r))); q["new"] = int(r.new_r)
        pl.append(q)
    over2[ti] = pl
json.dump(over2, open(OUT / "b1_rosters.json", "w"))
e3 = pd.DataFrame(json.loads(subprocess.run(["node", "src/simulator/validation/engine.js", V1, str(OUT / "b1_rosters.json")], capture_output=True, text=True, check=True).stdout))
D3 = e1[["name", "pts"]].merge(e3[["name", "pts"]], on="name", suffixes=("_v1", "_idfix")).merge(e2[["name", "pts"]], on="name")
D3 = D3.rename(columns={"pts": "pts_new"})
D3["pipeline_and_id_matching"] = D3.pts_idfix - D3.pts_v1; D3["roster_source"] = D3.pts_new - D3.pts_idfix; D3["total"] = D3.pts_new - D3.pts_v1
D3.to_csv(OUT / "b_decomposition.csv", index=False)
summary["B_decomposition"] = {"mean_abs_id_matching": float(D3.pipeline_and_id_matching.abs().mean()),
                              "max_abs_id_matching": float(D3.pipeline_and_id_matching.abs().max()),
                              "mean_abs_roster_source": float(D3.roster_source.abs().mean()),
                              "max_abs_roster_source": float(D3.roster_source.abs().max())}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(summary["B_decomposition"], indent=1))
print(D3.sort_values("total").round(2).to_string(index=False))
