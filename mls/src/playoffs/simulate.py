"""2026 MLS Cup Playoffs: exact round-by-round probabilities for the official bracket.

Ratings: market model (90-day half-life, chosen before the backtest) fitted on the 2026 regular season,
200 match-bootstrap draws for rating uncertainty. Paper 1's Dixon-Coles on goals is run alongside as a check.
Every probability is computed exactly for each rating draw (no Monte Carlo noise), then averaged over draws.
Seeds are the official ones (MLS announcement, Nov 2026); src/playoffs/data.py reproduces them from points.

Writes out/playoffs/bracket.json (page data) and results/playoffs_2026.csv.
Usage: python3 src/playoffs/simulate.py
"""
import json, sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "src/playoffs")
from engine import fit_market, p_game, p_bo3, dc_draws

OUT = pathlib.Path("out/playoffs")
M = pd.read_pickle("out/playoffs/matches.pkl")
TN = {t["team_id"]: t for t in json.loads(pathlib.Path("data/raw/teams.json").read_text())}
R = M[(M.season == 2026) & ~M.knockout]
teams = sorted(set(R.home) | set(R.away)); idx = {t: i for i, t in enumerate(teams)}
by_name = {TN[t]["team_name"]: t for t in teams}
SEEDS = {"East": ["Nashville SC", "Inter Miami CF", "New England Revolution", "Chicago Fire FC", "Charlotte FC",
                  "Philadelphia Union", "Columbus Crew", "Orlando City SC", "FC Cincinnati"],
         "West": ["Vancouver Whitecaps FC", "FC Dallas", "St. Louis City SC", "San Jose Earthquakes", "Houston Dynamo FC",
                  "Los Angeles FC", "Portland Timbers FC", "Seattle Sounders FC", "Real Salt Lake"]}

# regular-season record (MLS Cup host = better record: points, then wins, then goal difference)
rec = {}
for t in teams:
    h, a = R[R.home == t], R[R.away == t]
    w = int((h.hg > h.ag).sum() + (a.ag > a.hg).sum()); d = int((h.hg == h.ag).sum() + (a.ag == a.hg).sum())
    gf = int(h.hg.sum() + a.ag.sum()); ga = int(h.ag.sum() + a.hg.sum())
    rec[t] = {"w": w, "d": d, "l": len(h) + len(a) - w - d, "pts": 3 * w + d, "gf": gf, "ga": ga}
for conf, names in SEEDS.items():           # the official seeds must agree with the points table
    pts = [rec[by_name[n]]["pts"] for n in names]
    assert pts == sorted(pts, reverse=True), (conf, pts)
rank_key = lambda t: (rec[t]["pts"], rec[t]["w"], rec[t]["gf"] - rec[t]["ga"])

STAGES = ["r1", "sf", "cf", "final", "champ"]   # reached Round One, won R1, won conf semi, won conf final, won MLS Cup


def bracket(Rt):
    """Exact P(stage) for every playoff team under one set of ratings."""
    P = {}
    winner_prob = {}

    def g(host, guest, et):          # P(host wins one game)
        return p_game(Rt, idx[host], idx[guest], et)

    def merge(A, B, host_of):       # A, B: dict team -> P(reaches this slot); single game with extra time
        out = {}
        for a, pa in A.items():
            for b, pb in B.items():
                h = host_of(a, b); o = b if h == a else a
                ph = g(h, o, True)
                out[h] = out.get(h, 0) + pa * pb * ph
                out[o] = out.get(o, 0) + pa * pb * (1 - ph)
        return out

    champs = {}
    for conf, names in SEEDS.items():
        s = [by_name[n] for n in names]
        seed = {t: k + 1 for k, t in enumerate(s)}
        p8 = g(s[7], s[8], False)
        wc = {s[7]: p8, s[8]: 1 - p8}
        r1 = {}
        # 1 v 8/9
        A = {s[0]: 0.0}
        for t, pt in wc.items():
            q = p_bo3(Rt, idx[s[0]], idx[t])
            A[s[0]] += pt * q; A[t] = pt * (1 - q)
        ties = [A]
        for hi, lo in ((1, 6), (2, 5), (3, 4)):
            q = p_bo3(Rt, idx[s[hi]], idx[s[lo]])
            ties.append({s[hi]: q, s[lo]: 1 - q})
        by_seed = lambda a, b: a if seed[a] < seed[b] else b
        semi1 = merge(ties[0], ties[3], by_seed)      # 1/8/9 v 4/5
        semi2 = merge(ties[1], ties[2], by_seed)      # 2/7 v 3/6
        final = merge(semi1, semi2, by_seed)
        for t in s:
            P[t] = {"wc": 1.0 if seed[t] <= 7 else 1.0,
                    "r1": 1.0 if seed[t] <= 7 else wc[t],
                    "sf": sum(d.get(t, 0) for d in ties), "cf": semi1.get(t, 0) + semi2.get(t, 0), "final": final.get(t, 0)}
        champs[conf] = final
    host = lambda a, b: a if rank_key(a) > rank_key(b) else b
    cup = merge(champs["East"], champs["West"], host)
    for t in P:
        P[t]["champ"] = cup.get(t, 0)
    return P


def run(draws):
    acc = None
    for Rt in draws:
        P = bracket(Rt)
        if acc is None:
            acc = {t: {k: [] for k in P[t]} for t in P}
        for t in P:
            for k, v in P[t].items():
                acc[t][k].append(v)
    return {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in acc.items()}, acc


end = R.kickoff_utc.max()
point = fit_market(R, teams, 90, end)
boot = fit_market(R, teams, 90, end, boot=200, rng=np.random.default_rng(2026))
P_mkt, acc = run(boot)
P_point, _ = run([point])
for t, d in P_mkt.items():           # every team's stage probabilities must be non-increasing and conferences sum to 1
    v = [d[k] for k in STAGES]; assert all(v[i] >= v[i + 1] - 1e-12 for i in range(len(v) - 1))
for conf, names in SEEDS.items():
    assert abs(sum(P_mkt[by_name[n]]["final"] for n in names) - 1) < 1e-9
assert abs(sum(d["champ"] for d in P_mkt.values()) - 1) < 1e-9

# Paper 1 Dixon-Coles on 2026 goals, as a check
P_dc, _ = run(dc_draws(R, teams, n=200))

rows = []
for conf, names in SEEDS.items():
    for k, n in enumerate(names):
        t = by_name[n]
        rows.append({"conf": conf, "seed": k + 1, "team": n, "abbr": TN[t]["team_abbreviation"], **rec[t],
                     "rating_gd": float(point["att"][idx[t]] + point["def"][idx[t]]),
                     **{f"p_{s}": P_mkt[t][s] for s in STAGES},
                     "p_champ_lo": float(np.percentile(acc[t]["champ"], 10)), "p_champ_hi": float(np.percentile(acc[t]["champ"], 90)),
                     "p_champ_point": P_point[t]["champ"], "p_champ_dc": P_dc[t]["champ"]})
D = pd.DataFrame(rows)
D.to_csv("results/playoffs_2026.csv", index=False, float_format="%.4f")
print(D[["conf", "seed", "abbr", "pts", "rating_gd", "p_r1", "p_sf", "p_cf", "p_final", "p_champ", "p_champ_lo", "p_champ_hi", "p_champ_dc"]]
      .round(3).to_string(index=False))

# page data: point ratings + 200 bootstrap draws (for in-browser "play it out" runs)
data = {"teams": {TN[t]["team_abbreviation"]: {"name": TN[t]["team_name"], "short": TN[t].get("team_short_name") or TN[t]["team_name"],
                                                 **rec[t]} for t in teams if TN[t]["team_name"] in sum(SEEDS.values(), [])},
        "seeds": {c: [TN[by_name[n]]["team_abbreviation"] for n in ns] for c, ns in SEEDS.items()},
        "probs": {r["abbr"]: {s: round(r[f"p_{s}"], 4) for s in STAGES} | {"lo": round(r["p_champ_lo"], 4), "hi": round(r["p_champ_hi"], 4),
                                                                              "dc": round(r["p_champ_dc"], 4)} for r in rows},
        "draws": [{"c": round(float(d["c"]), 4), "h": round(float(d["home"]), 4),
                   "a": {TN[t]["team_abbreviation"]: round(float(d["att"][idx[t]]), 4) for t in teams},
                   "d": {TN[t]["team_abbreviation"]: round(float(d["def"][idx[t]]), 4) for t in teams}} for d in boot],
        "meta": {"matches": int(len(R)), "half_life_days": 90, "draws": len(boot), "last_match": str(end.date()),
                 "home_goals_mult": round(float(np.exp(point["home"])), 3)}}
data["draws"] = [{"c": d["c"], "h": d["h"], "a": {k: v for k, v in d["a"].items() if k in data["teams"]},
                  "d": {k: v for k, v in d["d"].items() if k in data["teams"]}} for d in data["draws"]]
json.dump(data, open(OUT / "bracket.json", "w"), separators=(",", ":"))
print("home advantage: goals x", data["meta"]["home_goals_mult"], "| bracket.json KB", (OUT / "bracket.json").stat().st_size // 1024)
