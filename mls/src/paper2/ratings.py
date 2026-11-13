"""Stage 1: player-season ratings for MLS from stints (plus-minus on xG difference, pulled toward a box-score prior).

Rating = a player's contribution to his team's non-penalty xG difference per 90 minutes while he is on the field.
Model per stint s (weight = duration):
    90*(xG_home - xG_away)/dur = home + man_adv*(n_home-n_away) + state terms + sum(home ratings) - sum(away ratings) + noise
Ridge penalty pulls each rating toward a prior mu built from that season's g+ by action type, position and age.
The prior's mapping is fitted with the target season left out (no self-leakage). Penalty strength by 5-fold CV grouped by match.
Outputs: out/ratings_mls.pkl (player-season ratings), out/ratings_meta.json, plus split-half stability.
Usage: python src/paper2/ratings.py [--split-half]
"""
import json, glob, sys
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ST = pd.read_pickle("out/stints_usa.1.pkl")
ST = ST[ST.season.between(2013, 2026) & (ST.dur >= 1) & (ST.n_home >= 9) & (ST.n_away >= 9)].reset_index(drop=True)
SP = pd.read_pickle("out/stint_players_usa.1.pkl")[["pid", "asa_id"]].drop_duplicates("pid")
ASA = dict(zip(SP.pid, SP.asa_id))
ST["y"] = 90 * (ST.xg_h - ST.xg_a) / ST.dur

# ---------- box-score prior features per (asa player, season) ----------
rows = []
for f in sorted(glob.glob("data/raw/mls_ga_20*.json")):
    s = int(f[-9:-5])
    for r in json.load(open(f)):
        d = {"asa_id": r["player_id"], "season": s, "minutes": r["minutes_played"], "pos": r["general_position"]}
        for a in r["data"]:
            d["ga_" + a["action_type"]] = a["goals_added_above_avg"]
        rows.append(d)
GA = pd.DataFrame(rows).fillna(0)
GA = GA.groupby(["asa_id", "season"]).agg({**{c: "sum" for c in GA.columns if c.startswith("ga_") or c == "minutes"},
                                           "pos": lambda v: v.iloc[0]}).reset_index()
ACT = [c for c in GA.columns if c.startswith("ga_")]
for c in ACT:
    GA[c + "_p96"] = GA[c] / GA.minutes.clip(lower=90) * 96
born = {p["player_id"]: p.get("birth_date") for p in json.load(open("data/raw/players.json"))}
GA["age"] = [(pd.Timestamp(f"{s}-07-01") - pd.Timestamp(born[a])).days / 365.25 if born.get(a) else np.nan for a, s in zip(GA.asa_id, GA.season)]


def design(st):
    """Sparse player-season design (+1 home, -1 away) and unpenalised covariates."""
    keys = {}
    r_i, c_i, v_i = [], [], []
    for i, (hp, ap, s) in enumerate(zip(st.home_players, st.away_players, st.season)):
        for p in hp:
            k = keys.setdefault(f"{p}|{s}", len(keys)); r_i.append(i); c_i.append(k); v_i.append(1.0)
        for p in ap:
            k = keys.setdefault(f"{p}|{s}", len(keys)); r_i.append(i); c_i.append(k); v_i.append(-1.0)
    X = sparse.csr_matrix((v_i, (r_i, c_i)), shape=(len(st), len(keys)))
    sd = np.sign(st.score_diff.values)
    Z = np.c_[np.ones(len(st)), (st.n_home - st.n_away).values, (sd > 0), (sd < 0)].astype(float)
    return X, Z, keys


def fit(X, Z, y, w, mu, lam):
    """Weighted ridge around mu; covariates scaled up so their penalty is negligible."""
    y2 = y - X @ mu
    A = sparse.hstack([X, sparse.csr_matrix(Z * 100.0)]).tocsr()
    m = Ridge(alpha=lam, fit_intercept=False, solver="sparse_cg", max_iter=5000, tol=1e-6).fit(A, y2, sample_weight=w)
    b = m.coef_
    return mu + b[: X.shape[1]], b[X.shape[1]:] * 100.0


def cv_lambda(X, Z, y, w, mu, groups, grid):
    best = None
    for lam in grid:
        err = 0.0
        for tr, te in GroupKFold(5).split(X, y, groups):
            beta, g = fit(X[tr], Z[tr], y[tr], w[tr], mu, lam)
            pred = X[te] @ beta + Z[te] @ g
            err += np.sum(w[te] * (y[te] - pred) ** 2)
        print(f"   lambda {lam:>7}: weighted CV error {err / w.sum():.4f}", flush=True)
        if best is None or err < best[1]:
            best = (lam, err)
    return best[0]


def prior(keys, beta0, wmin, seasons_all):
    """Map plain-ratings to box features; cross-fit by season so a season's prior never uses its own ratings."""
    K = pd.DataFrame({"key": list(keys)}); K[["pid", "season"]] = K.key.str.split("|", expand=True); K["season"] = K.season.astype(int)
    K["beta0"] = beta0; K["wmin"] = wmin; K["asa_id"] = K.pid.map(ASA)
    K = K.merge(GA, on=["asa_id", "season"], how="left")
    F = [c + "_p96" for c in ACT]
    K["has_box"] = K[F[0]].notna().astype(float)
    for c in F:
        K[c] = K[c].fillna(0).clip(-1.5, 1.5)
    K["age_c"] = (K.age.fillna(K.age.median()) - 27) / 5; K["age2"] = K.age_c ** 2
    P = pd.get_dummies(K.pos.fillna("NA"), prefix="pos", dtype=float)
    Xp = pd.concat([K[F + ["has_box", "age_c", "age2"]], P], axis=1).values
    mu = np.zeros(len(K))
    for s in seasons_all:
        tr = (K.season != s).values & (K.wmin.values >= 300); te = (K.season == s).values
        m = Ridge(alpha=10.0).fit(Xp[tr], K.beta0.values[tr], sample_weight=np.sqrt(K.wmin.values[tr]))
        mu[te] = m.predict(Xp[te])
    return mu, K


def run(st, label, grid=(4000, 8000, 16000, 32000, 64000, 128000, 256000)):
    X, Z, keys = design(st)
    y, w = st.y.values, st.dur.values
    groups = st.event.values
    # minutes per player-season (for weighting the prior fit)
    wmin = np.asarray(abs(X).T @ w).ravel()
    print(f"[{label}] stints {len(st)}, player-seasons {len(keys)}", flush=True)
    lam0 = cv_lambda(X, Z, y, w, np.zeros(len(keys)), groups, grid) if label == "full" else 2000
    beta0, _ = fit(X, Z, y, w, np.zeros(len(keys)), lam0)
    mu, K = prior(keys, beta0, wmin, sorted(st.season.unique()))
    lam = cv_lambda(X, Z, y, w, mu, groups, grid) if label == "full" else LAM_FULL
    beta, g = fit(X, Z, y, w, mu, lam)
    # approximate standard error (diagonal): sigma^2 / (sum of weights + lambda)
    resid = y - X @ beta - Z @ g; sigma2 = np.sum(w * resid ** 2) / w.sum() * np.mean(w)
    se = np.sqrt(sigma2 / (np.asarray(abs(X).T @ w).ravel() + lam))
    K["rating"] = beta; K["plain"] = beta0; K["prior"] = mu; K["se"] = se; K["minutes"] = wmin
    return K, {"lambda_plain": lam0, "lambda": lam, "home": g[0], "man_adv": g[1], "leading": g[2], "trailing": g[3]}


if __name__ == "__main__":
    K, meta = run(ST, "full")
    LAM_FULL = meta["lambda"]
    K.to_pickle("out/ratings_mls.pkl")
    json.dump({k: float(v) for k, v in meta.items()}, open("out/ratings_meta.json", "w"), indent=1)
    print(json.dumps({k: round(float(v), 3) for k, v in meta.items()}))
    big = K[K.minutes >= 900]
    print("player-seasons >= 900 min:", len(big), "| rating SD", round(big.rating.std(), 3), "| plain SD", round(big.plain.std(), 3),
          "| corr(rating, prior)", round(np.corrcoef(big.rating, big.prior)[0, 1], 3), "| corr(plain, prior)", round(np.corrcoef(big.plain, big.prior)[0, 1], 3))
    if "--split-half" in sys.argv:
        ST["half"] = ST.groupby("season").event.transform(lambda e: pd.factorize(e)[0] % 2)
        out = []
        for h in (0, 1):
            Kh, _ = run(ST[ST.half == h].reset_index(drop=True), f"half{h}")
            out.append(Kh[["key", "rating", "plain", "minutes"]].rename(columns={"rating": f"r{h}", "plain": f"p{h}", "minutes": f"m{h}"}))
        J = out[0].merge(out[1], on="key")
        J = J[(J.m0 >= 450) & (J.m1 >= 450)]
        res = {"n": int(len(J)), "split_half_rating": float(np.corrcoef(J.r0, J.r1)[0, 1]),
               "split_half_plain": float(np.corrcoef(J.p0, J.p1)[0, 1])}
        json.dump(res, open("out/ratings_split_half.json", "w"), indent=1)
        print("split-half:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})
