"""Simulation-calibrated rho (Addendum 2 of the preregistration).

1. Spread: find the true rating spread at which a simulated league reproduces the real split-half correlation of
   plain plus-minus ratings (0.154).
2. Calibration grid: for each true rho in GRID, simulate REPS leagues at that spread and record the naive estimate.
3. Confidence set for the real data: every true rho whose simulated estimates (mean +/- 1.96 sd) contain the real estimate.
4. Recovery check of the calibrated estimator on fresh simulations, including spreads x0.5 and x2 of the calibrated one.
Usage: python src/paper2/rho_calib.py spread | grid <rating_col> | real <rating_col> | check <rating_col>
Outputs accumulate in out/rho_calib_<rating_col>.json.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

exec(open("src/paper2/ratings.py").read().split('if __name__ == "__main__":')[0])   # ST, design(), fit()

MODE = sys.argv[1]; RCOL = sys.argv[2] if len(sys.argv) > 2 else "rating"
OUT = pathlib.Path(f"out/rho_calib_{RCOL}.json")
STATE = json.load(open(OUT)) if OUT.exists() else {}
RHOS = np.round(np.arange(-6, 6.01, 0.25), 2)
GRID = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
LAM = json.load(open("out/ratings_meta.json"))["lambda"]
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 7)

K = pd.read_pickle("out/ratings_mls_goals.pkl")
X, Z, keys = design(ST)
y, w = ST.y.values, ST.dur.values
klist = list(keys); kidx = {k: i for i, k in enumerate(klist)}
mu_real = K.set_index("key").prior.reindex(klist).fillna(0).values.copy()
extra = (K.set_index("key")[["finish90", "gk90"]].sum(axis=1).reindex(klist).fillna(0).values if RCOL == "rating_goals" else np.zeros(len(klist)))
b_true0 = K.set_index("key")[RCOL].reindex(klist).fillna(0).values.copy()

PAD = 12
def pad(lists):
    M = np.full((len(lists), PAD), -1, dtype=int)
    for i, l in enumerate(lists):
        M[i, :len(l)] = l[:PAD]
    return M
HM = pad([[kidx[f"{p}|{s}"] for p in hp] for hp, s in zip(ST.home_players, ST.season)])
AM = pad([[kidx[f"{p}|{s}"] for p in ap] for ap, s in zip(ST.away_players, ST.season)])
FOLD = np.zeros(len(ST), dtype=int)
for f, (_, te) in enumerate(GroupKFold(5).split(X, y, ST.event.values)):
    FOLD[te] = f
g_fit = np.linalg.lstsq(Z * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
RESID_SD = np.sqrt(np.average((y - Z @ g_fit) ** 2, weights=w))
# goals outcome (used to judge the finishing-adjusted version; finishing shows up in goals, not in xG)
yg = 90 * (ST.g_h - ST.g_a).values / ST.dur.values
gg_fit = np.linalg.lstsq(Z * np.sqrt(w)[:, None], yg * np.sqrt(w), rcond=None)[0]
RESID_SD_G = np.sqrt(np.average((yg - Z @ gg_fit) ** 2, weights=w))
GOALS = RCOL == "rating_goals"
b_x0 = K.set_index("key")["rating"].reindex(klist).fillna(0).values.copy()   # xG part of the truth




def lineup_ratings(Bfold):
    """Ratings of the players in each lineup, using that stint's out-of-fold rating vector. Returns (home, away) arrays with NaN pads."""
    Hr = np.where(HM >= 0, Bfold[FOLD[:, None], np.clip(HM, 0, None)], np.nan)
    Ar = np.where(AM >= 0, Bfold[FOLD[:, None], np.clip(AM, 0, None)], np.nan)
    return Hr, Ar


def Qdiff(Hr, Ar, rhos, floor, s):
    out = np.zeros((len(Hr), len(rhos)))
    for side, R, sign in ((0, Hr, 1), (1, Ar, -1)):
        q = np.clip(1 + (R - floor) / s, 0.05, None)
        n = np.sum(~np.isnan(R), axis=1)
        for j, r in enumerate(rhos):
            if r == 0:
                m = np.exp(np.nanmean(np.log(q), axis=1))
            else:
                m = np.nanmean(q ** r, axis=1) ** (1 / r)
            out[:, j] += sign * n * (floor + s * (m - 1))
    return out


def naive_rho(yv, mu, extra_v, yprof=None):
    """Cross-fitted ratings (fit on yv) -> profile over rho on yprof (default yv); returns rho_hat."""
    yprof = yv if yprof is None else yprof
    B = np.zeros((5, len(klist)))
    for f in range(5):
        tr = FOLD != f
        beta, _ = fit(X[tr], Z[tr], yv[tr], w[tr], mu, LAM)
        B[f] = beta + extra_v
    floor, s = np.percentile(B, 1), B.std()
    Hr, Ar = lineup_ratings(B)
    D = Qdiff(Hr, Ar, RHOS, floor, s)
    sw = np.sqrt(w); Zi = Z * sw[:, None]; yi = yprof * sw
    sse = []
    for j in range(len(RHOS)):
        Xd = np.c_[Zi, D[:, j] * sw]
        coef, *_ = np.linalg.lstsq(Xd, yi, rcond=None)
        r = yi - Xd @ coef; sse.append(r @ r)
    sse = np.array(sse)
    return float(RHOS[int(np.argmin(sse))]), sse - sse.min()


def simulate(true_rho, scale):
    """Returns (y to fit ratings on, y to profile on, prior, finishing added to ratings, truth)."""
    bx = b_x0 * scale                       # xG-creation truth
    bt = bx + (extra if GOALS else 0)       # full truth (finishing known, as in the real estimate)
    noise = lambda sd: rng.normal(0, sd, len(ST)) * np.sqrt(np.mean(w) / w)
    def D_of(b):
        floor, s = np.percentile(b, 1), b.std()
        Hr, Ar = lineup_ratings(np.tile(b, (5, 1)))
        return Qdiff(Hr, Ar, [true_rho], floor, s)[:, 0]
    y_fit = Z @ g_fit + D_of(bx) + noise(RESID_SD)
    y_prof = (Z @ gg_fit + D_of(bt) + noise(RESID_SD_G)) if GOALS else y_fit
    mu = 0.48 * bx / bx.std() * mu_real.std() + rng.normal(0, mu_real.std() * 0.88, len(klist))
    return y_fit, y_prof, mu, (extra if GOALS else np.zeros(len(klist))), bt


def split_half_plain(yv):
    half = ST.groupby("season").event.transform(lambda e: pd.factorize(e)[0] % 2).values
    rs = []
    for h in (0, 1):
        m = half == h
        beta, _ = fit(X[m], Z[m], yv[m], w[m], np.zeros(len(klist)), 2000)
        mins = np.asarray(abs(X[m]).T @ w[m]).ravel()
        rs.append((beta, mins))
    ok = (rs[0][1] >= 450) & (rs[1][1] >= 450)
    return float(np.corrcoef(rs[0][0][ok], rs[1][0][ok])[0, 1])


if MODE == "spread":
    real = split_half_plain(y)
    res = {"real_split_half_plain": real}
    for sc in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        ysim, *_ = simulate(1.0, sc)        # first output = xG outcome in both versions
        res[str(sc)] = split_half_plain(ysim)
        print("scale", sc, "simulated split-half", round(res[str(sc)], 3), "| real", round(real, 3), flush=True)
    xs = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0]); ys = np.array([res[str(x)] for x in xs])
    res["calibrated_scale"] = float(np.interp(real, ys, xs)) if np.all(np.diff(ys) > 0) else float(xs[np.argmin(abs(ys - real))])
    STATE["spread"] = res
    print("calibrated scale", round(res["calibrated_scale"], 2))

elif MODE == "grid":
    sc = STATE["spread"]["calibrated_scale"]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    G = STATE.setdefault("grid", {})
    for tr in GRID:
        for _ in range(reps):
            yf, yp, mu, ex, _ = simulate(tr, sc)
            G.setdefault(str(tr), []).append(naive_rho(yf, mu, ex, yp)[0])
        print("true", tr, "estimates", G[str(tr)], flush=True)

elif MODE == "real":
    r, curve = naive_rho(y, mu_real, extra, yg if GOALS else y)
    STATE["real_naive"] = r; STATE["real_curve"] = curve.round(3).tolist()
    print("real naive rho", r)

elif MODE == "check":
    # fresh simulations at spreads x0.5, x1, x2 of the calibrated one, true rho in {-2, 1, 3}; does the calibrated set cover?
    sc = STATE["spread"]["calibrated_scale"]
    C = STATE.setdefault("check", [])
    for mult in (0.5, 1.0, 2.0):
        for tr in (-2.0, 1.0, 3.0):
            yf, yp, mu, ex, _ = simulate(tr, sc * mult)
            C.append({"true": tr, "mult": mult, "naive": naive_rho(yf, mu, ex, yp)[0]})
            print(C[-1], flush=True)

json.dump(STATE, open(OUT, "w"), indent=1)
