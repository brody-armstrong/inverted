"""2027 rosters for the simulator from J's August 2026 salary list (summer signings included), plus comparables for signings.

Roster = the uploaded list (per-club update dates Mar–Sep 2026); San Diego, missing from it, uses the April MLSPA release.
Each player is matched by name to his 2026 MLS record (ESPN, within his club first, then league-wide, then ASA names),
then projected to 2027 with the production model (2026 and 2025 market ratings shrunk by minutes, age + 1, salary).
Signings: first-season market rating of past new arrivals by salary band and age group -> mean, 10th/90th percentile, n,
and share who did not play at all. Writes out/tool/data.json (replaces tool_data.py's rosters).
"""
import json, re, unicodedata, pathlib
import numpy as np, pandas as pd

M = json.load(open("out/proj_model.json")); coef = np.array(M["coef"])
D = pd.read_pickle("out/proj_D.pkl")
PM = pd.read_pickle("out/peer_pm_usa.1.pkl")
SPN = pd.read_pickle("out/stint_players_usa.1.pkl").drop_duplicates("pid").set_index("pid").name
T27 = pd.read_pickle("out/proj_teams_2027.pkl"); P27 = pd.read_pickle("out/proj_players_2027.pkl")
U = pd.read_csv("data/salaries_2026-08.csv")


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]", " ", n).split())


CLUB = {"Inter Miami": "Inter Miami CF", "Chicago Fire": "Chicago Fire FC", "Los Angeles FC": "LAFC", "Orlando City": "Orlando City SC",
        "LA Galaxy": "LA Galaxy", "Vancouver Whitecaps": "Vancouver Whitecaps", "FC Cincinnati": "FC Cincinnati", "Nashville SC": "Nashville SC",
        "Atlanta United": "Atlanta United FC", "Houston Dynamo": "Houston Dynamo FC", "Austin FC": "Austin FC", "NY Red Bulls": "Red Bull New York",
        "Toronto FC": "Toronto FC", "Charlotte FC": "Charlotte FC", "New York City FC": "New York City FC", "Seattle Sounders": "Seattle Sounders FC",
        "Portland Timbers": "Portland Timbers", "New England Revolution": "New England Revolution", "Columbus Crew": "Columbus Crew",
        "St. Louis City": "St. Louis CITY SC", "Minnesota United": "Minnesota United FC", "Colorado Rapids": "Colorado Rapids",
        "San Jose Earthquakes": "San Jose Earthquakes", "DC United": "D.C. United", "Real Salt Lake": "Real Salt Lake",
        "Sporting KC": "Sporting Kansas City", "FC Dallas": "FC Dallas", "Philadelphia Union": "Philadelphia Union", "CF Montreal": "CF Montréal"}
name2team = dict(zip(T27.name, T27.team))
U["team_e"] = U.club.map(CLUB).map(name2team)
assert U.team_e.notna().all(), U[U.team_e.isna()].club.unique()

# 2026 and 2025 player records (ESPN pid), by club and league-wide
s26 = PM[PM.season == 2026].groupby(["pid", "team"])["min"].sum().reset_index()
s26["nm"] = s26.pid.map(SPN).map(norm)
rat = PM.groupby(["season", "pid"]).agg(r=("rating", "first"), m=("min", "sum")).reset_index()


def match(row):
    nm = norm(row.player); toks = set(nm.split())
    for pool in (s26[s26.team == row.team_e], s26):
        exact = pool[pool.nm == nm]
        if len(exact):
            return exact.sort_values("min").pid.iloc[-1]
        cand = pool[pool.nm.apply(lambda x: len(toks & set(x.split())) >= max(1, min(len(toks), 2)) and x.split()[-1:] == nm.split()[-1:])]
        if len(cand) == 1:
            return cand.pid.iloc[0]
    return None


U["pid"] = [match(r) for r in U.itertuples()]
# San Diego from the April release (already in P27)
sd_id = name2team.get("San Diego FC")
SD = P27[P27.team_e == sd_id][["name", "sal", "age", "pid"]].rename(columns={"name": "player", "sal": "annual"})
SD["age"] = SD.age - 1; SD["contract_end"] = np.nan; SD["club"] = "San Diego FC"; SD["team_e"] = sd_id; SD["updated"] = "Apr 16, 2026"
R = pd.concat([U[["club", "updated", "player", "annual", "age", "contract_end", "team_e", "pid"]], SD], ignore_index=True)


def rec(pid, season):
    x = rat[(rat.pid == pid) & (rat.season == season)]
    return (float(x.r.iloc[0]), float(x.m.iloc[0])) if len(x) else (0.0, 0.0)


rows = []
for r in R.itertuples():
    r1, m1 = rec(r.pid, 2026) if r.pid else (0.0, 0.0); r2, m2 = rec(r.pid, 2025) if r.pid else (0.0, 0.0)
    age27 = (r.age + 1) if pd.notna(r.age) else 26
    p1, p2 = r1 * m1 / (m1 + 900), r2 * m2 / (m2 + 1800); new = float(m1 == 0 and m2 == 0)
    ls = np.log(max(r.annual if pd.notna(r.annual) else 1e5, 5e4)); ag = (age27 - 27) / 5
    proj = float(np.array([1, p1, p2, new, ls, ls * new, ag, ag ** 2]) @ coef)
    rows.append({"club": r.club, "team_e": r.team_e, "player": r.player, "pid": r.pid, "sal": float(r.annual) if pd.notna(r.annual) else 1e5,
                 "age": int(age27), "contract_end": None if pd.isna(r.contract_end) else int(r.contract_end), "proj": proj, "new": int(new),
                 "min26": m1, "updated": r.updated})
X = pd.DataFrame(rows)
# positions: ESPN position in 2026 (most common), else unknown
pos = {}
import gzip, glob
for f in ("data/espn/usa.1/2026.jsonl.gz", "data/espn/usa.1/2025.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        for p in json.loads(l)["rosters"]:
            if p["pos"] and p["pos"] not in ("SUB",):
                pos.setdefault(p["pid"], []).append(p["pos"])
def pgroup(codes):
    if not codes:
        return "M"
    c = pd.Series(codes).value_counts().index[0]
    return "GK" if c in ("G", "GK") else ("D" if c in ("D", "CD", "CD-L", "CD-R", "SW", "LB", "RB") else
                                          ("F" if c in ("F", "CF-L", "CF-R", "LF", "RF", "RCF") else "M"))
# position: ASA's general position (latest season) where known, else ESPN slots (formation-based, less reliable)
MAPP = pd.read_csv("out/espn_asa_players.csv", dtype=str); e2a = dict(zip(MAPP.pid, MAPP.asa_id))
gp = {}
for f in sorted(glob.glob("data/raw/mls_ga_20*.json")):
    for r in json.load(open(f)):
        gp[r["player_id"]] = r["general_position"]
ASAPOS = {"GK": "GK", "CB": "D", "FB": "D", "DM": "M", "CM": "M", "AM": "M", "W": "F", "ST": "F"}
X["pos"] = [ASAPOS.get(gp.get(e2a.get(p, ""), ""), None) or pgroup(pos.get(p, [])) for p in X.pid]
print("position source: ASA", int(sum(gp.get(e2a.get(p, ""), "") in ASAPOS for p in X.pid)), "of", len(X))

# comparables for signings
N = D[(D.new == 1)].copy(); N["played"] = N["min"].fillna(0) > 0
N["sb"] = pd.cut(N.sal, [0, 5e5, 1e6, 2e6, 4e6, 1e10], labels=False)
N["ab"] = (N.age >= 28).astype(int)
comps = {}
for (sb, ab), g in N.groupby(["sb", "ab"]):
    pl = g[g.played]
    comps[f"{int(sb)}_{int(ab)}"] = {"n": int(len(pl)), "dnp": float(1 - g.played.mean()), "mean": float(pl.rating.mean()) if len(pl) else None,
                                     "p10": float(pl.rating.quantile(.1)) if len(pl) else None, "p90": float(pl.rating.quantile(.9)) if len(pl) else None}

# assemble tool data (keeps schedule, model, metadata from tool_data.py)
data = json.load(open("out/tool/data.json"))
order = [t["id"] for t in data["teams"]]
for t in data["teams"]:
    g = X[X.team_e == t["id"]].sort_values("proj", ascending=False)
    t["players"] = [{"n": r.player, "pos": r.pos, "age": r.age, "sal": int(r.sal), "r": round(r.proj, 4), "new": r.new,
                     "ce": r.contract_end} for r in g.itertuples()]
    t["updated"] = g.updated.iloc[0] if len(g) else None
data["player_sd"] = M["player_resid_sd"]; data["comps"] = comps; data["sal_bands"] = [5e5, 1e6, 2e6, 4e6]
json.dump(data, open("out/tool/data.json", "w"), separators=(",", ":"))
print("players", len(X), "| matched to 2026 MLS record", int(X.pid.notna().sum()), "| treated as new to MLS", int(X.new.sum()),
      "| contracts ending 2026", int((X.contract_end == 2026).sum()))
for nm in ("Antoine Griezmann", "Robert Lewandowski", "Breel Embolo", "Lionel Messi", "Allan Saint-Maximin"):
    x = X[X.player.str.contains(nm.split()[-1])]
    print("  ", nm, "->", x[["club", "pid", "min26", "proj", "sal"]].round(3).to_dict("records")[:1])
print("comparables:", {k: (v["n"], None if v["mean"] is None else round(v["mean"], 3), round(v["dnp"], 2)) for k, v in comps.items()})
