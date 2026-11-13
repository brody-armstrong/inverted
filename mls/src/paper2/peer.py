"""Identical pipeline for MLS and peer leagues (P5 comparison, P9): market-based player ratings and absence costs.

Outcome per match: the closing market's expected goal difference (Pinnacle closing 1X2, margin removed, inverted
through independent Poisson scoring rates). Player exposure: minutes on the field from ESPN lineups (nominal clock).
Ratings: ridge plus-minus at match level, player-season keys, home intercept; shrinkage by 5-fold CV.
Then (same definitions as the MLS xG analysis):
  P5 shares: Cov(contribution of rank group, Q) / Var(Q), Q = minutes-weighted sum of ratings x 11.
  Absence: outside-cause absences of regulars (call-ups within 3 days, red-card suspensions); per-absence cost and
  per-rating-point cost by rank group, team-season and opponent-season fixed effects, bootstrap by team-season.
Usage: python src/paper2/peer.py <espn_league> <football-data code or MLS> [--ratings-only]
Writes out/peer_<league>.json and out/peer_ratings_<league>.pkl.
--ratings-only stops after section D (no absence study) and writes out/peer_<league>_ratings_only.json; the simulator
needs only these ratings, and section E needs out/national_team_squads.pkl.
"""
import sys, gzip, json, glob, re, unicodedata
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import poisson
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

LG, CODE = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(17)


def nominal(t):
    m = re.match(r"(\d+)'", t or "")
    return min(int(m.group(1)), 90) if m else None


# ---------- A. ESPN matches and player minutes ----------
matches, mins, reds = [], [], set()
for f in sorted(glob.glob(f"data/espn/{LG}/*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        if len(r["rosters"]) < 22 or len(r["teams"]) != 2:
            continue
        home = [k for k, v in r["teams"].items() if v["home"]][0]; away = [k for k, v in r["teams"].items() if not v["home"]][0]
        d = pd.Timestamp(r["date"]).tz_convert("UTC")
        season = d.year if LG == "usa.1" else (d.year + 1 if d.month >= 7 else d.year)
        matches.append({"event": r["id"], "date": d, "season": season, "home": home, "away": away, "hn": r["teams"][home]["name"],
                        "an": r["teams"][away]["name"], "hg": int(r["teams"][home]["score"] or 0), "ag": int(r["teams"][away]["score"] or 0),
                        "stype": r.get("season_type") or ""})
        for p in r["rosters"]:
            subs = [nominal(x["t"]) for x in p["plays"] if x["sub"]]
            rc = [nominal(x["t"]) for x in p["plays"] if x["rc"]]
            if rc:
                reds.add((r["id"], p["pid"]))
            on = 0 if p["starter"] else (subs[0] if p["subIn"] and subs else None)
            if on is None:
                continue
            off = 90
            if p["subOut"] and subs:
                off = subs[-1] if (p["starter"] or len(subs) > 1) else 90
            if rc and rc[0] is not None:
                off = min(off, rc[0])
            if off is not None and off > on:
                mins.append({"event": r["id"], "pid": p["pid"], "team": p["team"], "min": off - on, "name": p["name"]})
MT = pd.DataFrame(matches); MN = pd.DataFrame(mins)
if LG == "usa.1":
    MT = MT[MT.stype.str.contains("regular", na=False)]
MT = MT[MT.season.between(2013, 2026)]

# ---------- B. market expected goal difference ----------
G = np.arange(0, 11); cache = {}
def solve(ph, pa):
    k = (round(ph, 3), round(pa, 3))
    if k not in cache:
        f = lambda x: np.array([np.tril(np.outer(poisson.pmf(G, np.exp(x[0])), poisson.pmf(G, np.exp(x[1]))), -1).sum(),
                                np.triu(np.outer(poisson.pmf(G, np.exp(x[0])), poisson.pmf(G, np.exp(x[1]))), 1).sum()]) - [ph, pa]
        x = least_squares(f, [0.4, 0.1]).x; cache[k] = np.exp(x[0]) - np.exp(x[1])
    return cache[k]

if CODE == "MLS":
    MK = pd.read_pickle("out/market_gd_mls.pkl")[["event", "mkt_gd"]]
    MT = MT.merge(MK, on="event", how="inner")
else:
    O = []
    for f in sorted(glob.glob(f"data/ext/odds_{CODE}_*.csv")):
        d = pd.read_csv(f, encoding="latin-1")
        d = d.dropna(subset=["HomeTeam", "FTHG"])
        h, dr, a = (pd.to_numeric(d[c], errors="coerce") for c in ("PSCH", "PSCD", "PSCA"))
        for c0, c1, c2 in (("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")):
            if c0 in d:
                h = h.fillna(pd.to_numeric(d[c0], errors="coerce")); dr = dr.fillna(pd.to_numeric(d[c1], errors="coerce")); a = a.fillna(pd.to_numeric(d[c2], errors="coerce"))
        inv = pd.concat([1 / h, 1 / dr, 1 / a], axis=1); pr = inv.div(inv.sum(1), axis=0)
        O.append(pd.DataFrame({"d": pd.to_datetime(d.Date, dayfirst=True, errors="coerce"), "fh": d.HomeTeam, "fa": d.AwayTeam,
                               "hg": d.FTHG.astype(int), "ag": d.FTAG.astype(int), "ph": pr.iloc[:, 0], "pa": pr.iloc[:, 2]}))
    O = pd.concat(O).dropna()
    O["mkt_gd"] = [solve(a, b) for a, b in zip(O.ph, O.pa)]
    MT["d"] = MT.date.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None).dt.normalize()
    j = MT.merge(O, on=["d", "hg", "ag"])
    amb = j.groupby("event").fh.transform("size")
    pairs = pd.concat([j[amb == 1][["hn", "fh"]].set_axis(["e", "f"], axis=1), j[amb == 1][["an", "fa"]].set_axis(["e", "f"], axis=1)])
    NM = pairs.groupby("e").f.agg(lambda v: v.value_counts().index[0]).to_dict()
    MT["fh"] = MT.hn.map(NM); MT["fa"] = MT.an.map(NM)
    L = MT.merge(O.rename(columns={"d": "d2", "hg": "hg2", "ag": "ag2"}), on=["fh", "fa"])
    L = L[(L.d - L.d2).abs().dt.days <= 2].drop_duplicates("event")
    MT = L.drop(columns=["d2", "hg2", "ag2", "fh", "fa", "ph", "pa"])
MN = MN[MN.event.isin(MT.event)]
print(f"{LG}: matches with lineups and market line {len(MT)}; player-match rows {len(MN)}", flush=True)

# ---------- C. ratings (match-level ridge on market goal difference) ----------
MN = MN.merge(MT[["event", "season", "home"]], on="event")
MN["key"] = MN.pid + "|" + MN.season.astype(str)
MN["sgn"] = np.where(MN.team == MN.home, 1.0, -1.0)
keys = {k: i for i, k in enumerate(MN.key.unique())}; ev_i = {e: i for i, e in enumerate(MT.event)}
X = sparse.csr_matrix((MN.sgn * MN["min"] / 90, (MN.event.map(ev_i), MN.key.map(keys))), shape=(len(MT), len(keys)))
y = MT.mkt_gd.values
A = sparse.hstack([X, sparse.csr_matrix(np.ones((len(MT), 1)) * 100)]).tocsr()
best = None
for lam in (1, 3, 10, 30, 100):
    err = 0
    for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
        m = Ridge(alpha=lam, fit_intercept=False).fit(A[tr], y[tr]); err += np.sum((y[te] - m.predict(A[te])) ** 2)
    best = (lam, err) if best is None or err < best[1] else best
m = Ridge(alpha=best[0], fit_intercept=False).fit(A, y)
R = pd.DataFrame({"key": list(keys), "rating": m.coef_[:-1]})
R[["pid", "season"]] = R.key.str.split("|", expand=True); R["season"] = R.season.astype(int)
base_err = np.sum((y - y.mean()) ** 2)
print(f"   lambda {best[0]}; CV R^2 {1 - best[1] / base_err:.3f}; home intercept {m.coef_[-1] * 100:.3f}", flush=True)

# ---------- D. team quality and P5 shares ----------
PM = MN.groupby(["season", "team", "pid", "key"])["min"].sum().reset_index().merge(R[["key", "rating"]], on="key")
PM["tmin"] = PM.groupby(["season", "team"])["min"].transform("sum")
PM["c"] = PM.rating * PM["min"] / PM.tmin * 11
PM["regular"] = PM["min"] >= 900
PM["rank"] = PM[PM.regular].groupby(["season", "team"]).rating.rank(ascending=False, method="first")
PM["grp"] = np.where(PM["rank"] <= 3, "1-3", np.where(PM["rank"] <= 11, "4-11", "rest"))
Q = PM.groupby(["season", "team"]).c.sum()
C = PM.pivot_table(index=["season", "team"], columns="grp", values="c", aggfunc="sum").fillna(0)
shares = {g: float(np.cov(C[g], Q.loc[C.index])[0, 1] / Q.loc[C.index].var()) for g in ("1-3", "4-11", "rest") if g in C}
q_sd = float(Q.groupby(level=0).std().mean())

# ---------- ratings only (the simulator needs sections A-D; section E needs out/national_team_squads.pkl) ----------
if "--ratings-only" in sys.argv:
    R.to_pickle(f"out/peer_ratings_{LG}.pkl"); PM.to_pickle(f"out/peer_pm_{LG}.pkl")
    res = {"league": LG, "matches": int(len(MT)), "lambda": best[0], "cv_r2": float(1 - best[1] / base_err),
           "team_seasons": int(len(Q)), "team_quality_sd_within_season": q_sd, "P5_shares": shares}
    json.dump(res, open(f"out/peer_{LG}_ratings_only.json", "w"), indent=1)
    print(json.dumps(res, default=float))
    sys.exit(0)

# ---------- E. absences ----------
NT = pd.read_pickle("out/national_team_squads.pkl")
ntd = NT.groupby("pid").date.apply(lambda s: np.array(sorted(s.values), dtype="datetime64[D]"))
TM = pd.concat([MT.assign(team=MT.home, opp=MT.away, home_f=1, gd=MT.mkt_gd), MT.assign(team=MT.away, opp=MT.home, home_f=0, gd=-MT.mkt_gd)])
TM = TM.sort_values(["team", "date"]).reset_index(drop=True)
TM["prev"] = TM.groupby(["team", "season"]).event.shift(1)
TM["dd"] = TM.date.dt.tz_localize(None).dt.normalize().values.astype("datetime64[D]")
played = set(zip(MN.event, MN.pid))
regs = PM[PM.regular].copy()
nonreg = PM[~PM.regular]
repl = nonreg.groupby(["season", "team"]).apply(lambda g: np.average(g.rating, weights=g["min"]) if g["min"].sum() > 0 else 0,
                                                 include_groups=False).rename("repl")
regs = regs.merge(repl, on=["season", "team"], how="left").fillna({"repl": 0}); regs["above"] = regs.rating - regs.repl
grp = lambda rk: "r1" if rk == 1 else ("r2_3" if rk <= 3 else ("r4_5" if rk <= 5 else ("r6_11" if rk <= 11 else None)))
RG = {k: v for k, v in regs.groupby(["season", "team"])}
ab = []
for t in TM.itertuples():
    g = RG.get((t.season, t.team))
    if g is None:
        continue
    for p in g.itertuples():
        if (t.event, p.pid) in played:
            continue
        a = ntd.get(p.pid)
        cu = a is not None and np.any(np.abs((a - np.datetime64(t.dd, "D")).astype(int)) <= 3)
        su = isinstance(t.prev, str) and (t.prev, p.pid) in reds
        if (cu or su) and grp(p.rank):
            ab.append({"i": t.Index, "g": grp(p.rank), "above": p.above, "why": "callup" if cu else "susp"})
AB = pd.DataFrame(ab)
GR = ["r1", "r2_3", "r4_5", "r6_11"]
for g in GR:
    sub = AB[AB.g == g] if len(AB) else AB
    TM["N_" + g] = sub.groupby("i").size().reindex(TM.index).fillna(0) if len(sub) else 0.0
    TM["L_" + g] = sub.groupby("i").above.sum().reindex(TM.index).fillna(0) if len(sub) else 0.0
TM["ts"] = TM.team + "_" + TM.season.astype(str); TM["os"] = TM.opp + "_" + TM.season.astype(str)


def fe(df, cols):
    Y = df.gd.values.astype(float); Xc = df[["home_f"] + cols].values.astype(float)
    a = pd.factorize(df.ts)[0]; b = pd.factorize(df.os)[0]
    def dm(v):
        v = v.copy()
        for _ in range(30):
            v -= (np.bincount(a, v) / np.bincount(a))[a]; v -= (np.bincount(b, v) / np.bincount(b))[b]
        return v
    Yd = dm(Y); Xd = np.column_stack([dm(Xc[:, j]) for j in range(Xc.shape[1])])
    return np.linalg.lstsq(Xd, Yd, rcond=None)[0][1:]


NC = ["N_" + g for g in GR]; LC = ["L_" + g for g in GR]
n_est, l_est = fe(TM, NC), fe(TM, LC)
tsu = TM.ts.unique(); idx_by = {t: np.where(TM.ts.values == t)[0] for t in tsu}
bn, bl = [], []
for _ in range(300):
    pick = rng.choice(tsu, len(tsu)); ix = np.concatenate([idx_by[t] for t in pick])
    B = TM.iloc[ix].copy(); B["ts"] = B.ts + "_" + np.repeat(np.arange(len(pick)), [len(idx_by[t]) for t in pick]).astype(str)
    bn.append(fe(B, NC)); bl.append(fe(B, LC))
bn, bl = np.array(bn), np.array(bl)
ratio = bl[:, 3] / bl[:, 0]        # per-point cost ranks 6-11 / rank 1
res = {"league": LG, "matches": int(len(MT)), "lambda": best[0], "cv_r2": float(1 - best[1] / base_err),
       "team_seasons": int(len(Q)), "team_quality_sd_within_season": q_sd, "P5_shares": shares,
       "absences": AB.groupby(["g", "why"]).size().unstack(fill_value=0).to_dict() if len(AB) else {},
       "per_absence": {g: [float(n_est[j]), float(np.percentile(bn[:, j], 2.5)), float(np.percentile(bn[:, j], 97.5))] for j, g in enumerate(GR)},
       "per_point": {g: [float(l_est[j]), float(np.percentile(bl[:, j], 2.5)), float(np.percentile(bl[:, j], 97.5))] for j, g in enumerate(GR)},
       "P9_ratio_r6_11_over_r1": [float(l_est[3] / l_est[0]), float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))]}
R.to_pickle(f"out/peer_ratings_{LG}.pkl"); PM.to_pickle(f"out/peer_pm_{LG}.pkl")
json.dump(res, open(f"out/peer_{LG}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k in ("matches", "cv_r2", "team_quality_sd_within_season", "P5_shares")}, default=float))
print("   per absence:", {g: round(v[0], 3) for g, v in res["per_absence"].items()}, "| P9 ratio:", [round(x, 2) for x in res["P9_ratio_r6_11_over_r1"]])
