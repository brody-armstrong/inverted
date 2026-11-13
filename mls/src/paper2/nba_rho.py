"""P3: the method check on the NBA. Ratings = player Box Plus-Minus (team-season rows); outcome = five-man lineup net
points per 100 possessions (top 20 lineups per team-season, Basketball-Reference), weighted by minutes.
Same generalized-mean team quality and the same calibrated (Neyman) confidence set as for MLS:
  truth t = BPM x scale; observed rating = truth/scale + noise with reliability REL (BPM's correlation with true impact);
  lineup net = a + Q_rho(truth) + noise (sd = k / sqrt(minutes), k from the real fit).
Scale chosen so the simulated spread of lineup nets matches the real one. Pass (P3) if the whole set lies above 1.
Usage: python src/paper2/nba_rho.py [REL]
"""
import sys, glob, json, re, unicodedata
import numpy as np, pandas as pd

REL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
rng = np.random.default_rng(3)
RHOS = np.round(np.arange(-6, 6.01, 0.25), 2)


def abbr(name):
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    parts = n.replace(".", "").split()
    return (parts[0][0] + ". " + " ".join(parts[1:])).lower() if len(parts) > 1 else n.lower()


rows = []
for f in sorted(glob.glob("data/nba/lineups_*.csv")):
    y = int(f[-8:-4])
    L = pd.read_csv(f); A = pd.read_csv(f"data/nba/advanced_{y}.csv")
    tcol = "Team" if "Team" in A.columns else "Tm"
    A = A[~A[tcol].isin(["TOT", "2TM", "3TM", "4TM", "5TM"])]
    A["k"] = A.Player.map(abbr)
    bpm = {(t, k): b for t, k, b in zip(A[tcol], A.k, pd.to_numeric(A.BPM, errors="coerce"))}
    for r in L.itertuples():
        names = [abbr(x.strip()) for x in str(r.Lineup).split("|")]
        vals = [bpm.get((r.team, n)) for n in names]
        mp = str(r.MP).split(":"); mins = int(mp[0]) + int(mp[1]) / 60 if len(mp) == 2 else np.nan
        net = getattr(r, "_4")
        if len(vals) == 5 and all(v is not None and not np.isnan(v) for v in vals) and mins > 0:
            rows.append({"season": y, "team": r.team, "min": mins, "net": float(net), "r": vals})
D = pd.DataFrame(rows)
R = np.array(D.r.tolist()); w = D["min"].values; ynet = D.net.values
print(f"lineups matched {len(D)} of {sum(len(pd.read_csv(f)) for f in glob.glob('data/nba/lineups_*.csv'))}; minutes covered {w.sum():.0f}")


def Q(Rm, rho, floor, s):
    q = np.clip(1 + (Rm - floor) / s, 0.05, None)
    m = np.exp(np.mean(np.log(q), axis=1)) if rho == 0 else np.mean(q ** rho, axis=1) ** (1 / rho)
    return 5 * (floor + s * (m - 1))


def naive(Rm, yv):
    floor, s = np.percentile(Rm, 1), Rm.std()
    sw = np.sqrt(w); sse = []
    for r in RHOS:
        Xd = np.c_[np.ones(len(yv)), Q(Rm, r, floor, s)] * sw[:, None]
        c, *_ = np.linalg.lstsq(Xd, yv * sw, rcond=None); e = yv * sw - Xd @ c; sse.append(e @ e)
    return float(RHOS[int(np.argmin(sse))])


obs_rho = naive(R, ynet)
floor, s = np.percentile(R, 1), R.std()
Xd = np.c_[np.ones(len(ynet)), Q(R, 1.0, floor, s)]
c1 = np.linalg.lstsq(Xd * np.sqrt(w)[:, None], ynet * np.sqrt(w), rcond=None)[0]
k = np.sqrt(np.average((ynet - Xd @ c1) ** 2 * w, weights=np.ones(len(w))) / 1.0)  # resid * sqrt(min)
real_var = np.average((ynet - np.average(ynet, weights=w)) ** 2, weights=w)
sd_e = R.std() * np.sqrt((1 - REL) / REL)


def simulate(true_rho, scale):
    t = R * scale
    o = t / scale + rng.normal(0, sd_e, R.shape)
    ft, st = np.percentile(t, 1), t.std()
    ys = c1[0] + Q(t, true_rho, ft, st) + rng.normal(0, 1, len(w)) * k / np.sqrt(w)
    return o, ys


# spread: match the weighted variance of lineup nets
best = None
for sc in np.arange(0.2, 3.01, 0.1):
    v = np.mean([np.average((lambda y: (y - np.average(y, weights=w)) ** 2)(simulate(1.0, sc)[1]), weights=w) for _ in range(5)])
    if best is None or abs(v - real_var) < best[1]:
        best = (float(sc), abs(v - real_var))
SC = best[0]
GRID = [-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
G = {g: [naive(*simulate(g, SC)) for _ in range(30)] for g in GRID}
cset = [g for g, v in G.items() if np.mean(v) - 1.96 * np.std(v, ddof=1) <= obs_rho <= np.mean(v) + 1.96 * np.std(v, ddof=1)]
# recovery check of the calibrated set, spreads x0.5/x1/x2 and reliability +/- 0.15
chk = []
for mult in (0.5, 1.0, 2.0):
    for tr in (-2.0, 1.0, 3.0):
        o, ys = simulate(tr, SC * mult); est = naive(o, ys)
        cs = [g for g, v in G.items() if np.mean(v) - 1.96 * np.std(v, ddof=1) <= est <= np.mean(v) + 1.96 * np.std(v, ddof=1)]
        verdict = "inconclusive" if not cs else ("weak-link" if max(cs) < 1 else ("strong-link" if min(cs) > 1 else "inconclusive"))
        chk.append({"true": tr, "mult": mult, "est": est, "covers": tr in cs, "verdict": verdict})
res = {"reliability_assumed": REL, "lineups": int(len(D)), "scale": SC, "naive_rho": obs_rho,
       "grid_means": {str(g): round(float(np.mean(v)), 2) for g, v in G.items()},
       "confidence_set": cset, "P3": "pass" if cset and min(cset) > 1 else ("fail" if cset and max(cset) <= 1 else "inconclusive"),
       "recovery_check": chk, "coverage": f"{sum(c['covers'] for c in chk)} of {len(chk)}",
       "false_directional": sum((c['verdict'] == 'weak-link' and c['true'] >= 1) or (c['verdict'] == 'strong-link' and c['true'] <= 1) for c in chk)}
json.dump(res, open(f"out/nba_rho_rel{REL}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "recovery_check"}, indent=1))
for c in chk:
    print(c)
