"""Task 16 (continued).
A. Sensitivity: the raw model overstates MLS luck share (0.65 vs 0.56 observed). Rescale every season's team strengths
   by one common factor so the model reproduces the observed average, then redo the Shapley split.
B. Year-to-year rank correlation = true-talent persistence x reliability of the two tables (classical disattenuation),
   reliability = 1 - luck share. Two-factor Shapley split of the MLS-Europe gap, bootstrap over season pairs.
C. Relegation replay (mechanical only): a 30-team league whose true strengths drift year to year with MLS persistence;
   the bottom three of the observed table are replaced by promoted clubs placed where European promoted clubs land.
Writes out/sim_shapley_calibrated.csv, out/sim_rankcorr.json, out/sim_relegation.json.
"""
import itertools, math, json, glob, re
import numpy as np, pandas as pd
from sim_core import *

rng = np.random.default_rng(5)
SEASONS = [s for s in range(2013, 2027) if s != 2020]
IDX = np.arange(0, 400, 2)

# ---------- A. calibrated baseline ----------
FIX = {}
for s in SEASONS:
    o = P[s]; ids = {t: x for x, t in enumerate(o["teams"])}; x = R[R.season == s]
    FIX[s] = (x.home.map(ids).values, x.away.map(ids).values, x.neutral.astype(float).values)


def mean_luck(c):
    v = []
    for s in SEASONS:
        o = P[s]; n = len(o["teams"])
        ls, _, _ = luck_share(o["draws"], IDX, n, FIX[s], frozenset({"talent"}), np.full(len(IDX), c), np.ones(len(IDX)))
        v.append(ls.mean())
    return np.mean(v)


target = float(mls_bs.luck_share.mean())
lo, hi = 1.0, 2.0
for _ in range(14):
    mid = (lo + hi) / 2
    if mean_luck(mid) > target:
        lo = mid
    else:
        hi = mid
C0 = (lo + hi) / 2
print("common strength multiplier", round(C0, 3), "-> model luck", round(mean_luck(C0), 3), "observed", round(target, 3))

V = {}
for s in SEASONS:
    o = P[s]; dr = o["draws"]; n = len(o["teams"])
    c0 = np.full(len(IDX), C0)
    k_eu = calibrate(dr, IDX, n, T_EU, "k")
    r_base = np.ones(len(IDX)); r_eu = calibrate(dr, IDX, n, H_EU, "r", k_fixed=c0)
    for cmb in [frozenset(c) for r_ in range(5) for c in itertools.combinations(FEAT, r_)]:
        k = k_eu if "talent" in cmb else c0
        # luck_share applies k only when "talent" is in the combo, so always include it and choose k here
        ls, _, _ = luck_share(dr, IDX, n, FIX[s], cmb | {"talent"}, k, r_eu)
        V.setdefault(cmb, []).append(ls)
Vm = {c: np.mean(np.stack(v), 0) for c, v in V.items()}
shap = {}
for f in FEAT:
    tot = np.zeros(len(IDX)); others = [g for g in FEAT if g != f]
    for r_ in range(4):
        for S_ in itertools.combinations(others, r_):
            S_ = frozenset(S_); w = math.factorial(len(S_)) * math.factorial(3 - len(S_)) / math.factorial(4)
            tot += w * (Vm[S_] - Vm[S_ | {f}])
    shap[f] = tot
q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
base = Vm[frozenset()]; allf = Vm[frozenset(FEAT)]
rows = [{"component": "MLS as played (calibrated model)", **dict(zip(["mean", "lo", "hi"], q(base)))}]
rows += [{"component": f, **dict(zip(["mean", "lo", "hi"], q(shap[f])))} for f in FEAT]
rows += [{"component": "MLS with all four switched", **dict(zip(["mean", "lo", "hi"], q(allf)))},
         {"component": "Europe observed", "mean": LUCK_EU, "lo": np.nan, "hi": np.nan},
         {"component": "residual (not these features)", **dict(zip(["mean", "lo", "hi"], q(allf - LUCK_EU)))}]
SC = pd.DataFrame(rows); gap = float(base.mean() - LUCK_EU)
SC["share_of_gap"] = np.where(SC.component.isin(FEAT + ["residual (not these features)"]), SC["mean"] / gap, np.nan)
SC.to_csv("out/sim_shapley_calibrated.csv", index=False)
print(SC.round(3).to_string(index=False))

# ---------- B. rank correlation: persistence x reliability ----------
RC = pd.read_csv("out/balance_rankcorr.csv")
BSs = BS.set_index(["league", "season"])


def pairs(leagues):
    out = []
    for r in RC[RC.league.isin(leagues)].itertuples():
        a, b = (r.league, r.season), (r.league, r.season + 1)
        if a not in BSs.index or b not in BSs.index or BSs.loc[a, "short"] or BSs.loc[b, "short"]:
            continue
        rel = np.sqrt((1 - BSs.loc[a, "luck_share"]) * (1 - BSs.loc[b, "luck_share"]))
        out.append((r.rho, rel))
    return np.array(out)


pm, pe = pairs(["MLS"]), pairs(EU_NAMES)


def decomp(pm, pe):
    rm, Rm = pm[:, 0].mean(), pm[:, 1].mean(); re_, Re = pe[:, 0].mean(), pe[:, 1].mean()
    km, ke = rm / Rm, re_ / Re                      # implied true-talent persistence
    rel_part = 0.5 * ((km * Re - km * Rm) + (ke * Re - ke * Rm))
    per_part = 0.5 * ((ke * Rm - km * Rm) + (ke * Re - km * Re))
    return np.array([rm, re_, Rm, Re, km, ke, rel_part, per_part])


pt = decomp(pm, pe)
bs = np.array([decomp(pm[rng.integers(0, len(pm), len(pm))], pe[rng.integers(0, len(pe), len(pe))]) for _ in range(4000)])
keys = ["r_mls", "r_eu", "reliability_mls", "reliability_eu", "persistence_mls", "persistence_eu", "gap_from_reliability", "gap_from_persistence"]
RCJ = {k: {"mean": float(pt[x]), "lo": float(np.percentile(bs[:, x], 2.5)), "hi": float(np.percentile(bs[:, x], 97.5))} for x, k in enumerate(keys)}
RCJ["pairs"] = {"mls": len(pm), "eu": len(pe)}
json.dump(RCJ, open("out/sim_rankcorr.json", "w"), indent=1)
print({k: round(v["mean"], 3) for k, v in RCJ.items() if k != "pairs"})

# ---------- C. relegation (mechanical) ----------
# European promoted clubs: first-season ppg relative to league mean, in units of that season's talent SD
prom = []
for code in ["E0", "SP1", "I1", "D1", "F1", "N1", "P1", "B1", "T1", "SC0", "G1"]:
    prev = None
    for f in sorted(glob.glob(f"data/ext/odds_{code}_*.csv")):
        d = pd.read_csv(f, encoding="latin-1", usecols=lambda c: c in {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}).dropna()
        tm = pd.concat([pd.DataFrame({"t": d.HomeTeam, "p": 3 * (d.FTHG > d.FTAG) + (d.FTHG == d.FTAG)}),
                        pd.DataFrame({"t": d.AwayTeam, "p": 3 * (d.FTAG > d.FTHG) + (d.FTHG == d.FTAG)})]).groupby("t").p.mean()
        if prev is not None:
            new = [t for t in tm.index if t not in prev]
            for t in new:
                prom.append((tm[t] - tm.mean()) / tm.std(ddof=1))
        prev = set(tm.index)
PROM_Z = float(np.mean(prom))   # in observed-table SD units
L_M = float(mls_bs.luck_share.mean()); K_M = RCJ["persistence_mls"]["mean"]
print("European promoted clubs finish", round(PROM_Z, 2), "observed SDs below the league mean; n =", len(prom))


def run(releg, seasons=3000, n=30, burn=200):
    z = rng.normal(0, 1, n)                  # true talent, units of MLS talent SD
    sd_t, luck = [], []
    noise_sd = math.sqrt(L_M / (1 - L_M))  # chance spread relative to talent SD (MLS)
    for t in range(seasons):
        obs = z + rng.normal(0, noise_sd, n)
        if t >= burn:
            sd_t.append(z.std(ddof=1))
        z = K_M * z + math.sqrt(1 - K_M ** 2) * rng.normal(0, 1, n)
        if releg:
            bot = np.argsort(obs)[:3]
            # promoted club: placed so its expected observed finish matches European promoted clubs
            z[bot] = PROM_Z * math.sqrt(1 + noise_sd ** 2) + rng.normal(0, 0.5, 3)
    s = np.mean(sd_t)
    return s, noise_sd ** 2 / (noise_sd ** 2 + s ** 2)


s0, l0 = run(False); s1, l1 = run(True)
REL = {"talent_sd_ratio": s1 / s0, "luck_no_releg": l0, "luck_releg": l1, "promoted_z": PROM_Z, "n_promoted": len(prom),
       "persistence": K_M}
json.dump(REL, open("out/sim_relegation.json", "w"), indent=1)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in REL.items()})
