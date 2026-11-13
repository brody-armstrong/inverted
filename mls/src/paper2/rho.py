"""Stage 2: estimate rho (weak-link vs strong-link) with frozen, cross-fitted player ratings; plus parameter recovery.

Team quality of a lineup of n players with ratings r_i (xGD per 90):
    q_i = 1 + (r_i - r_floor) / s            (positive, linear in r; r_floor = 1st percentile, s = rating SD)
    Q = n * [ r_floor + s * ( M_rho(q) - 1 ) ],  M_rho(q) = ( mean(q^rho) )^(1/rho)   (geometric mean at rho = 0)
rho = 1 is the plain sum of ratings. rho < 1 weights weaker players more (weak-link); rho > 1 stronger players (strong-link).
Stint model (weight = duration):  y = Z*gamma + beta * (Q_home - Q_away) + noise,  profiled over a grid of rho.
Ratings for each stint come from a fit that excluded its match (5-fold by match), so they never see the outcome.
Interval: bootstrap over matches. Usage: python src/paper2/rho.py <rating_col> [--recover]
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

exec(open("src/paper2/ratings.py").read().split('if __name__ == "__main__":')[0])   # design(), fit(), ST, prior pieces

RCOL = sys.argv[1] if len(sys.argv) > 1 else "rating"
RHOS = np.round(np.arange(-6, 6.01, 0.25), 2)
LAM = json.load(open("out/ratings_meta.json"))["lambda"]
rng = np.random.default_rng(42)

K = pd.read_pickle("out/ratings_mls_goals.pkl")
X, Z, keys = design(ST)
y, w = ST.y.values, ST.dur.values
kidx = pd.Series(range(len(keys)), index=list(keys))
mu = K.set_index("key").prior.reindex(list(keys)).fillna(0).values.copy()
extra = (K.set_index("key")[["finish90", "gk90"]].sum(axis=1).reindex(list(keys)).fillna(0).values if RCOL == "rating_goals" else 0.0)


def lineup_index(st):
    H = [[kidx[f"{p}|{s}"] for p in hp] for hp, s in zip(st.home_players, st.season)]
    A = [[kidx[f"{p}|{s}"] for p in ap] for ap, s in zip(st.away_players, st.season)]
    return H, A


H, A = lineup_index(ST)


def crossfit_ratings(yv):
    """Out-of-fold rating vectors: R[f] for stints in fold f."""
    R = np.zeros((len(ST), 0)).tolist()
    out = np.empty(len(ST), dtype=object)
    for tr, te in GroupKFold(5).split(X, yv, ST.event.values):
        beta, _ = fit(X[tr], Z[tr], yv[tr], w[tr], mu, LAM)
        beta = beta + extra
        for i in te:
            out[i] = beta
    return out


def Q_matrix(beta_per_stint):
    """Q_home - Q_away for every stint and every rho on the grid."""
    allb = np.concatenate([b for b in {id(b): b for b in beta_per_stint}.values()])
    floor, s = np.percentile(allb, 1), allb.std()
    D = np.zeros((len(ST), len(RHOS)))
    for i in range(len(ST)):
        b = beta_per_stint[i]
        for side, idx, sign in ((0, H[i], 1), (1, A[i], -1)):
            q = 1 + (b[idx] - floor) / s
            q = np.clip(q, 0.05, None)
            n = len(idx)
            for j, r in enumerate(RHOS):
                m = np.exp(np.mean(np.log(q))) if r == 0 else np.mean(q ** r) ** (1 / r)
                D[i, j] += sign * n * (floor + s * (m - 1))
    return D


def profile(D, yv, idx=None):
    """Weighted SSE for each rho (gamma, beta re-estimated); returns SSE array and beta at each rho."""
    idx = np.arange(len(yv)) if idx is None else idx
    sw = np.sqrt(w[idx])
    sse, betas = [], []
    Zi = Z[idx] * sw[:, None]; yi = yv[idx] * sw
    for j in range(len(RHOS)):
        Xd = np.c_[Zi, D[idx, j] * sw]
        coef, *_ = np.linalg.lstsq(Xd, yi, rcond=None)
        r = yi - Xd @ coef
        sse.append(r @ r); betas.append(coef[-1])
    return np.array(sse), np.array(betas)


def estimate(yv, nboot=200, label=""):
    bps = crossfit_ratings(yv)
    D = Q_matrix(bps)
    sse, betas = profile(D, yv)
    j = int(np.argmin(sse))
    ev = ST.event.values; uev = np.unique(ev); pos = pd.Series(np.arange(len(ev))).groupby(ev).apply(np.array)
    boots = []
    for b in range(nboot):
        pick = rng.choice(uev, len(uev))
        idx = np.concatenate(pos.loc[pick].values)
        s_b, _ = profile(D, yv, idx)
        boots.append(RHOS[int(np.argmin(s_b))])
    boots = np.array(boots)
    res = {"label": label, "rho_hat": float(RHOS[j]), "beta_at_rho": float(betas[j]), "beta_at_1": float(betas[list(RHOS).index(1.0)]),
           "rho_lo": float(np.percentile(boots, 2.5)), "rho_hi": float(np.percentile(boots, 97.5)),
           "share_boot_below_1": float(np.mean(boots < 1)), "sse_curve": (sse - sse.min()).round(3).tolist()}
    print(f"[{label}] rho {res['rho_hat']} (95% {res['rho_lo']} to {res['rho_hi']}), share of bootstraps < 1: {res['share_boot_below_1']:.2f}, "
          f"beta {res['beta_at_rho']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    if "--recover" in sys.argv:
        # Parameter recovery: true player ratings = current ratings scaled to a plausible true spread; outcomes simulated
        # from the stint model with a known rho and the observed residual noise; the prior is the truth plus noise
        # (calibrated so prior and truth correlate like the real prior and ratings do). Then the full procedure re-runs.
        b_true0 = np.asarray(K.set_index("key")[RCOL].reindex(list(keys)).fillna(0).values)
        g_fit = np.linalg.lstsq(Z * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        resid_sd = np.sqrt(np.average((y - Z @ g_fit) ** 2, weights=w))
        out = []
        for true_rho in (-2.0, 1.0, 3.0):
            for scale in (1.0, 2.0):
                bt = b_true0 * scale
                floor, s = np.percentile(bt, 1), bt.std()
                Dt = np.zeros(len(ST))
                for i in range(len(ST)):
                    for idx, sign in ((H[i], 1), (A[i], -1)):
                        q = np.clip(1 + (bt[idx] - floor) / s, 0.05, None)
                        m = np.exp(np.mean(np.log(q))) if true_rho == 0 else np.mean(q ** true_rho) ** (1 / true_rho)
                        Dt[i] += sign * len(idx) * (floor + s * (m - 1))
                ysim = Z @ g_fit + Dt + rng.normal(0, resid_sd, len(ST)) * np.sqrt(np.mean(w) / w)
                mu_saved = mu.copy()
                mu[:] = 0.48 * bt / bt.std() * mu_saved.std() + rng.normal(0, mu_saved.std() * 0.88, len(mu))
                r = estimate(ysim, nboot=100, label=f"true rho {true_rho}, spread x{scale}")
                r.update({"true_rho": true_rho, "scale": scale}); out.append(r)
                mu[:] = mu_saved
        json.dump(out, open(f"out/rho_recovery_{RCOL}.json", "w"), indent=1)
    else:
        res = estimate(y, label=f"MLS {RCOL}")
        json.dump(res, open(f"out/rho_mls_{RCOL}.json", "w"), indent=1)
