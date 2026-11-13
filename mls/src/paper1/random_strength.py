"""Fit the Dixon-Coles model to every MLS regular season; save posterior draws and team expected points."""
import sys, pickle
sys.path.insert(0, "src/paper1")
import numpy as np, pandas as pd
from dc_model import fit_nuts, outcome_probs
from numpyro.diagnostics import summary

M = pd.read_pickle("out/mls_matches.pkl")
R = M[~M.knockout].copy()
out, diag = {}, []
for s in range(2013, 2027):
    d = R[R.season == s]
    teams = sorted(set(d.home) | set(d.away))
    ids = {t: i for i, t in enumerate(teams)}
    m = fit_nuts(d.home.map(ids).values, d.away.map(ids).values, len(teams), d.hg.values, d.ag.values,
                 neutral=d.neutral.astype(float).values, warmup=1000, samples=1000, chains=2, seed=s)
    smp = {k: np.asarray(v) for k, v in m.get_samples().items()}
    sm = summary({k: v for k, v in m.get_samples(group_by_chain=True).items() if not k.startswith("z_")})
    rh = max(float(np.max(v["r_hat"])) for v in sm.values()); ne = min(float(np.min(v["n_eff"])) for v in sm.values())
    # thin to 400 draws; centre att/def as in the model
    idx = np.linspace(0, len(smp["c"]) - 1, 400).astype(int)
    dr = {k: v[idx] for k, v in smp.items()}
    dr["att"] = dr["att"] - dr["att"].mean(1, keepdims=True)
    dr["def"] = dr["def"] - dr["def"].mean(1, keepdims=True)
    out[s] = {"teams": teams, "draws": dr}
    diag.append({"season": s, "matches": len(d), "teams": len(teams), "max_rhat": rh, "min_ess": ne,
                 "home": float(smp["home"].mean()), "s_att": float(smp["s_att"].mean()), "s_def": float(smp["s_def"].mean()),
                 "s_hc": float(smp["s_hc"].mean())})
    print(diag[-1], flush=True)


def expected_ppg_balanced(dr, n):
    """Per draw: each team's expected points per game over a balanced double round robin."""
    D = len(dr["c"])
    i, j = np.where(~np.eye(n, dtype=bool))
    E = np.zeros((D, n))
    for k in range(D):
        lh = np.exp(dr["c"][k] + dr["home"][k] + dr["home_club"][k][i] + dr["att"][k][i] - dr["def"][k][j])
        la = np.exp(dr["c"][k] + dr["att"][k][j] - dr["def"][k][i])
        pH, pD, pA = outcome_probs(lh, la, np.full(len(i), dr["rho"][k]))
        pts_home = 3 * pH + pD; pts_away = 3 * pA + pD
        tot = np.bincount(i, pts_home, n) + np.bincount(j, pts_away, n)
        E[k] = tot / (2 * (n - 1))
    return E


for s, o in out.items():
    o["exp_ppg"] = expected_ppg_balanced(o["draws"], len(o["teams"]))
pickle.dump(out, open("out/dc_posteriors.pkl", "wb"))
pd.DataFrame(diag).to_csv("out/dc_diagnostics.csv", index=False)
print(pd.DataFrame(diag).round(3).to_string())
print({s: round(float(o["exp_ppg"].std(1).mean()), 3) for s, o in out.items()})
