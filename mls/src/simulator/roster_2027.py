"""2027 rosters for the simulator from public data, plus comparables for signings.

Rosters: the latest MLSPA salary release (via the American Soccer Analysis API; currently Spring 2026, players under
contract as of April 16, 2026), projected to 2027 by src/paper2/project.py with the production model (2026 and 2025
market ratings shrunk by minutes, age + 1, salary). tool_data.py has already written these rosters to out/tool/data.json;
this step adds the roster date, the residual SD and the signing comparables.

Signings: first-season market rating of past new arrivals by salary band and age group -> mean, 10th/90th percentile, n,
share who did not play.

v1.0 (development build, 2026-11) used a privately compiled club-updated salary list instead; that list is not used
or redistributed. See VALIDATION.md for the comparison with v1.0.
Usage: python3 src/simulator/roster_2027.py   (after src/simulator/tool_data.py)
"""
import json
import numpy as np, pandas as pd

M = json.load(open("out/proj_model.json"))
D = pd.read_pickle("out/proj_D.pkl")
S = pd.concat([pd.DataFrame(json.load(open(f"data/raw/mls_sal_{s}.json"))) for s in (2026,)])
release = pd.to_datetime(S.mlspa_release).max()
label = f"{release:%b} {release.day}, {release.year} (MLSPA)"

# comparables for signings
N = D[(D.new == 1)].copy(); N["played"] = N["min"].fillna(0) > 0
N["sb"] = pd.cut(N.sal, [0, 5e5, 1e6, 2e6, 4e6, 1e10], labels=False)
N["ab"] = (N.age >= 28).astype(int)
comps = {}
for (sb, ab), g in N.groupby(["sb", "ab"]):
    pl = g[g.played]
    comps[f"{int(sb)}_{int(ab)}"] = {"n": int(len(pl)), "dnp": float(1 - g.played.mean()), "mean": float(pl.rating.mean()) if len(pl) else None,
                                     "p10": float(pl.rating.quantile(.1)) if len(pl) else None, "p90": float(pl.rating.quantile(.9)) if len(pl) else None}

data = json.load(open("out/tool/data.json"))
for t in data["teams"]:
    t["updated"] = label
data["player_sd"] = M["player_resid_sd"]; data["comps"] = comps; data["sal_bands"] = [5e5, 1e6, 2e6, 4e6]
data["meta"]["roster_source"] = f"MLSPA salary release of {release:%Y-%m-%d}"
data["meta"]["roster_date"] = f"{release:%B} {release.day}, {release.year}"
json.dump(data, open("out/tool/data.json", "w"), separators=(",", ":"))
print("teams", len(data["teams"]), "| players", sum(len(t["players"]) for t in data["teams"]), "| roster date", label)
print("comparables:", {k: (v["n"], None if v["mean"] is None else round(v["mean"], 3), round(v["dnp"], 2)) for k, v in comps.items()})
