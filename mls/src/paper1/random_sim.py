"""Task 16 driver: Shapley decomposition of the luck-share gap and Monte Carlo format replays. Helpers in sim_core.py."""
from sim_core import *

# ---------- 1. analytic decomposition ----------
combos = [frozenset(c) for r in range(5) for c in itertools.combinations(FEAT, r)]
V = {c: [] for c in combos}
CAL = {}
checks = {}
base_pred, base_obs = [], []
for s in SEASONS:
    o = P[s]; dr = o["draws"]; n = len(o["teams"]); ids = {t: x for x, t in enumerate(o["teams"])}
    x = R[R.season == s]
    fix = (x.home.map(ids).values, x.away.map(ids).values, x.neutral.astype(float).values)
    idx = np.arange(D)
    k_eu = calibrate(dr, idx, n, T_EU, "k"); r_eu = calibrate(dr, idx, n, H_EU, "r")
    CAL[s] = {"k": k_eu, "r": r_eu}
    for c in combos:
        ls, _, _ = luck_share(dr, idx, n, fix, c, k_eu, r_eu)
        V[c].append(ls)
    obs = mls_bs[mls_bs.season == s].luck_share.iloc[0]
    base_pred.append(V[frozenset()][-1]); base_obs.append(obs)
    print(s, "k", round(float(np.median(k_eu)), 2), "r", round(float(np.median(r_eu)), 2),
          "luck actual", round(float(V[frozenset()][-1].mean()), 3), "obs", round(float(obs), 3),
          "all-EU", round(float(V[frozenset(FEAT)][-1].mean()), 3), flush=True)

Vm = {c: np.mean(np.stack(V[c]), 0) for c in combos}     # average over seasons, per draw
shap = {}
for f in FEAT:
    tot = np.zeros(D)
    others = [g for g in FEAT if g != f]
    for r_ in range(4):
        for S_ in itertools.combinations(others, r_):
            S_ = frozenset(S_)
            w = math.factorial(len(S_)) * math.factorial(3 - len(S_)) / math.factorial(4)
            tot += w * (Vm[S_] - Vm[S_ | {f}])          # reduction in luck share from switching f to Europe
    shap[f] = tot
base = Vm[frozenset()]; allf = Vm[frozenset(FEAT)]
q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
rows = [{"component": "MLS as played (model)", **dict(zip(["mean", "lo", "hi"], q(base)))}]
for f in FEAT:
    rows.append({"component": f, **dict(zip(["mean", "lo", "hi"], q(shap[f])))})
rows.append({"component": "MLS with all four switched", **dict(zip(["mean", "lo", "hi"], q(allf)))})
rows.append({"component": "Europe observed", "mean": LUCK_EU, "lo": np.nan, "hi": np.nan})
rows.append({"component": "residual (not these features)", **dict(zip(["mean", "lo", "hi"], q(allf - LUCK_EU)))})
SH = pd.DataFrame(rows)
gap = float(base.mean() - LUCK_EU)
SH["share_of_gap"] = np.where(SH.component.isin(FEAT + ["residual (not these features)"]), SH["mean"] / gap, np.nan)
SH.to_csv("out/sim_shapley.csv", index=False)
print(SH.round(3).to_string(index=False))

# posterior predictive check: model luck share vs observed, per season
bp = np.array([b.mean() for b in base_pred]); bo = np.array(base_obs)
lo_ = np.array([np.percentile(b, 5) for b in base_pred]); hi_ = np.array([np.percentile(b, 95) for b in base_pred])
checks["ppc_luck"] = {"seasons": SEASONS, "pred": bp.round(3).tolist(), "obs": bo.round(3).tolist(),
                      "mean_pred": float(bp.mean()), "mean_obs": float(bo.mean())}

# ---------- 2. Monte Carlo replays of formats ----------
SCEN = [("MLS as played: Supporters' Shield", frozenset(), "shield"),
        ("MLS as played: MLS Cup", frozenset(), "cup"),
        ("Balanced 34, no playoffs", frozenset({"schedule"}), "shield"),
        ("Balanced 38, no playoffs", frozenset({"schedule", "length"}), "shield"),
        ("+ European home advantage", frozenset({"schedule", "length", "home"}), "shield"),
        ("+ European talent spread", frozenset(FEAT), "shield")]
NREP = 25   # replays per posterior draw
DS = np.arange(0, D, 2)  # 200 draws
res = {name: [] for name, _, _ in SCEN}
shield_cup = []
luck_sim = []
for s in SEASONS:
    o = P[s]; dr = o["draws"]; n = len(o["teams"]); ids = {t: x for x, t in enumerate(o["teams"])}
    x = R[R.season == s]; fix = (x.home.map(ids).values, x.away.map(ids).values, x.neutral.astype(float).values)
    PO_N, fmt = PO[s]
    for name, combo, trophy in SCEN:
        hits = []
        k_all = CAL[s]["k"][DS] if "talent" in combo else np.ones(len(DS)); r_all = CAL[s]["r"][DS] if "home" in combo else np.ones(len(DS))
        if "schedule" in combo:
            L = 38 if "length" in combo else 34
            i0, j0, n0 = drr(n)
            ROUNDS = circle_rounds(n, i0, j0)
            NR_NEED = int(round(L * len(ROUNDS) / (2 * (n - 1))))
        else:
            i0, j0, n0 = fix
        pH, pD, pA = probs(dr, DS, i0, j0, n0, k_all, r_all)
        for di, d in enumerate(DS):
            # strongest team: best balanced expected ppg under this scenario's parameters
            bi, bj, bn = drr(n)
            bh, bd, ba = probs(dr, np.array([d]), bi, bj, bn, k_all[di:di + 1], r_all[di:di + 1])
            Eb, _, gb, _, _ = team_moments(bh, bd, ba, bi, bj, n)
            best = int(np.argmax(Eb[0]))
            if "schedule" in combo:   # random balanced slate: L games' worth of rounds from a circle-method double round robin
                tab = np.zeros((NREP, n))
                for rep in range(NREP):
                    rr = []
                    while len(rr) < NR_NEED:
                        rr.extend(rng.permutation(len(ROUNDS)).tolist())
                    pidx = np.concatenate([ROUNDS[q] for q in rr[:NR_NEED]])
                    tab[rep] = sim_table(pH[di:di + 1, pidx], pD[di:di + 1, pidx], i0[pidx], j0[pidx], n)[0]
            else:
                tab = sim_table(np.repeat(pH[di:di + 1], NREP, 0), np.repeat(pD[di:di + 1], NREP, 0), i0, j0, n)
            order = np.argsort(-tab, 1)
            if trophy == "shield":
                hits.extend((order[:, 0] == best).tolist())
            else:
                for rep in range(NREP):
                    ch = bracket(dr, d, k_all[di], r_all[di], list(order[rep]), fmt, PO_N)
                    hits.append(ch == best)
                    shield_cup.append(ch == order[rep, 0])
            if name == "MLS as played: Supporters' Shield":
                v = tab.var(1, ddof=1)
                luck_sim.append((s, float(np.mean(v))))
        res[name].append(np.mean(hits))
    print(s, {k_: round(v[-1], 3) for k_, v in res.items()}, flush=True)

FM = []
for name, combo, trophy in SCEN:
    v = np.array(res[name])
    bs = [rng.choice(v, len(v)).mean() for _ in range(4000)]
    FM.append({"format": name, "trophy": trophy, "p_best_wins": v.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
               "luck_share": float(Vm[combo].mean()), "luck_lo": float(np.percentile(Vm[combo], 2.5)),
               "luck_hi": float(np.percentile(Vm[combo], 97.5))})
FM = pd.DataFrame(FM)
FM.to_csv("out/sim_formats.csv", index=False)
print(FM.round(3).to_string(index=False))
checks["sim_shield_wins_cup"] = float(np.mean(shield_cup))
checks["actual_shield_wins_cup"] = 2 / 12
json.dump(checks, open("out/sim_checks.json", "w"), indent=1)
print("Shield winner wins Cup: simulated", round(np.mean(shield_cup), 3), "actual 2/12 =", round(2 / 12, 3))
