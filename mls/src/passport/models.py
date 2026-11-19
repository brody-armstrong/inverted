"""Stage 3: reliability gate, empirical-Bayes shrinkage, age-adjusted percentiles, readiness models,
backtest, outcome (b), calibration, final refit, 2027 predictions. Follows PREREGISTRATION.md.
Usage: python -I models.py <db_path> <pathway_model.json> <out_dir>
"""
import sys, os, json, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd
import duckdb
import statsmodels.api as sm

warnings.filterwarnings("ignore")
DB, PATHWAY, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261118)

GROUP = {"ST": "Strikers", "W": "Wingers", "AM": "Midfield", "CM": "Midfield", "DM": "Midfield",
         "FB": "Fullbacks", "CB": "Centre-backs"}
ACTIONS = ["dribbling", "passing", "receiving", "shooting", "interrupting"]
KPIS = ["gplus"] + [f"ga_{a}" for a in ACTIONS] + ["xg", "xa"]
LABEL = {"gplus": "g+ total", "ga_dribbling": "g+ dribbling", "ga_passing": "g+ passing", "ga_receiving": "g+ receiving",
         "ga_shooting": "g+ shooting", "ga_interrupting": "g+ interrupting", "xg": "xG", "xa": "xA"}
MIN_NP, MIN_HALF, MLS_CAP, REL_PASS = 900, 450, 500, 0.40
TRAIN, TEST = [2022, 2023], [2024]
T2 = "zeQZe4DqKw"
B = 500
Z90 = 1.6449

con = duckdb.connect(DB, read_only=True)
pm = con.execute("SELECT * FROM player_match WHERE NOT knockout").df()
players = con.execute("SELECT * FROM players").df().set_index("player_id")
mls = con.execute("SELECT * FROM mls_season").df()
con.close()

pm["gplus"] = pm[[f"ga_{a}" for a in ACTIONS + ["fouling"]]].sum(axis=1)
pm["season"] = pm.season.astype(int)
pm = pm[pm.position != "GK"]

# ---------- player-season table (MLS NEXT Pro, regular season) ----------
pos = (pm.groupby(["player_id", "season", "position"]).minutes.sum().reset_index()
       .sort_values("minutes").groupby(["player_id", "season"]).position.last())
ps = pm.groupby(["player_id", "season"]).agg(minutes=("minutes", "sum"), **{k: (k, "sum") for k in KPIS}).reset_index()
ps = ps.merge(pos.rename("position").reset_index(), on=["player_id", "season"])
ps["group"] = ps.position.map(GROUP)
ps = ps[ps.group.notna()]
for k in KPIS:
    ps[k + "_96"] = ps[k] / ps.minutes * 96
bd = players.birth_date
ps["age"] = (pd.to_datetime(ps.season.astype(str) + "-07-01") - pd.to_datetime(ps.player_id.map(bd))).dt.days / 365.25
t2min = pm[pm.team_id == T2].groupby(["player_id", "season"]).minutes.sum()

# ---------- 1. reliability (odd/even matches, training seasons) ----------
tr = pm[pm.season.isin(TRAIN)].sort_values(["player_id", "season", "date", "game_id"]).copy()
tr["half"] = tr.groupby(["player_id", "season"]).cumcount() % 2
h = tr.groupby(["player_id", "season", "half"]).agg(minutes=("minutes", "sum"), **{k: (k, "sum") for k in KPIS}).reset_index()
h = h[h.minutes >= MIN_HALF]
both = h.groupby(["player_id", "season"]).half.nunique()
h = h.set_index(["player_id", "season"]).loc[both[both == 2].index].reset_index()
rel = []
for k in KPIS:
    h[k + "_96"] = h[k] / h.minutes * 96
    w = h.pivot_table(index=["player_id", "season"], columns="half", values=k + "_96")
    r = w[0].corr(w[1])
    sb = 2 * r / (1 + r)
    rel.append(dict(kpi=k, label=LABEL[k], n_player_seasons=len(w), r_half=round(r, 3), r_full_sb=round(sb, 3),
                    passes=bool(sb >= REL_PASS)))
rel = pd.DataFrame(rel)
PASS = rel[rel.passes].kpi.tolist()
rel.to_csv(f"{OUT}/reliability.csv", index=False)
print(rel.to_string(index=False))

# ---------- 2. empirical-Bayes shrinkage per KPI x group x season (players with >= 900 minutes) ----------
pool = ps[ps.minutes >= MIN_NP].copy()
dev = pm.merge(pool[["player_id", "season"] + [k + "_96" for k in KPIS]], on=["player_id", "season"])
for k in KPIS:
    dev[k + "_d2"] = (dev[k] - dev[k + "_96"] * dev.minutes / 96) ** 2
v = dev.groupby(["player_id", "season"])[[k + "_d2" for k in KPIS]].sum()
pool = pool.merge(v.reset_index(), on=["player_id", "season"])
prior = []
for k in KPIS:
    pool[k + "_v"] = pool[k + "_d2"] / (pool.minutes / 96) ** 2
    for (g, s), idx in pool.groupby(["group", "season"]).groups.items():
        y, vv = pool.loc[idx, k + "_96"], pool.loc[idx, k + "_v"]
        mu = y.mean()
        tau2 = max(y.var() - vv.mean(), 0.05 * y.var())   # floor keeps a little between-player spread
        post_v = 1 / (1 / vv + 1 / tau2)
        post_m = post_v * (y / vv + mu / tau2)
        pool.loc[idx, k + "_eb"] = post_m
        pool.loc[idx, k + "_sd"] = np.sqrt(post_v)
        pool.loc[idx, k + "_pct"] = y.rank(pct=True) * 100
        # age-adjusted percentile: residual of the shrunk estimate on age, age^2
        a = pool.loc[idx, "age"]
        ok = a.notna()
        if ok.sum() > 10:
            X = np.column_stack([np.ones(ok.sum()), a[ok], a[ok] ** 2])
            beta, *_ = np.linalg.lstsq(X, post_m[ok], rcond=None)
            res = pd.Series(np.nan, index=idx)
            res[ok[ok].index] = post_m[ok] - X @ beta
            pool.loc[idx, k + "_agepct"] = res.rank(pct=True) * 100
        prior.append(dict(kpi=k, group=g, season=s, n=len(idx), mu=mu, tau=np.sqrt(tau2), median_se=np.sqrt(vv.median())))
pd.DataFrame(prior).to_csv(f"{OUT}/shrinkage_priors.csv", index=False)
scale = []
for k in KPIS:
    for g, d in pool[pool.season == 2026].groupby("group"):
        scale.append(dict(kpi=LABEL[k], group=g, p05=d[k + "_eb"].quantile(.05), p50=d[k + "_eb"].median(), p95=d[k + "_eb"].quantile(.95)))
pd.DataFrame(scale).to_csv(f"{OUT}/kpi_scale_2026.csv", index=False)
pool["gplus_z"] = pool.groupby(["group", "season"]).gplus_eb.transform(lambda x: (x - x.mean()) / x.std())

# ---------- 3. cohorts and outcomes ----------
mls_min = mls.groupby(["player_id", "season"]).minutes.sum()
mls_ga = mls.groupby(["player_id", "season"]).ga_total.sum()


def cum_mls(pid, upto):
    s = mls_min.get(pid)
    return 0 if s is None else float(s[s.index <= upto].sum())


def mls_in(pid, s):
    return float(mls_min.get((pid, s), 0.0))


coh = pool.copy()
coh["mls_cum"] = [cum_mls(p, s) for p, s in zip(coh.player_id, coh.season)]
coh = coh[coh.mls_cum < MLS_CAP].copy()
coh["mls_S"] = [mls_in(p, s) for p, s in zip(coh.player_id, coh.season)]
coh["m1"] = [mls_in(p, s + 1) for p, s in zip(coh.player_id, coh.season)]
coh["m2"] = [mls_in(p, s + 2) for p, s in zip(coh.player_id, coh.season)]
coh["y"] = ((coh.m1 + coh.m2) >= MLS_CAP).astype(int)
coh["log_np"] = np.log(coh.minutes)
coh["log_mls"] = np.log1p(coh.mls_S)
coh["raw_pct"] = coh.gplus_pct
coh = coh.dropna(subset=["age"])
print("\ncohort sizes:", coh.groupby("season").size().to_dict(),
      "| missing birth dates dropped:", int(pool[pool.age.isna()].shape[0]))

FEATS = {"new": ["age", "gplus_z", "log_np", "log_mls", "g_Fullbacks", "g_Midfield", "g_Strikers", "g_Wingers"],
         "raw_pct": ["raw_pct"], "minutes": ["log_np", "log_mls"]}
for g in ["Fullbacks", "Midfield", "Strikers", "Wingers"]:
    coh["g_" + g] = (coh.group == g).astype(float)


def person_periods(df, last_season):
    rows = []
    for r in df.itertuples():
        if r.season + 1 <= last_season:
            rows.append(dict(r._asdict(), period=1, event=int(r.m1 >= MLS_CAP)))
            if r.m1 < MLS_CAP and r.season + 2 <= last_season:
                rows.append(dict(r._asdict(), period=2, event=int(r.m1 + r.m2 >= MLS_CAP)))
    pp = pd.DataFrame(rows)
    pp["p2"] = (pp.period == 2).astype(float)
    return pp


def fit(pp, feats):
    X = sm.add_constant(pp[["p2"] + feats].astype(float), has_constant="add")
    return sm.Logit(pp.event.astype(float), X).fit(disp=0, method="lbfgs", maxiter=500)


def hazards(m, df, feats):
    out = []
    for p2 in (0.0, 1.0):
        X = sm.add_constant(df[feats].astype(float).assign(p2=p2)[["p2"] + feats], has_constant="add")
        out.append(m.predict(X))
    return np.asarray(out[0]), np.asarray(out[1])


def brier(p, y): return float(np.mean((p - y) ** 2))
def logloss(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------- 4. backtest: train 2022-23, test 2024 ----------
train = coh[coh.season.isin(TRAIN)]
test = coh[coh.season.isin(TEST)].copy()
pp_train = person_periods(train, last_season=2025)
bt, calib, models = [], [], {}
for name, feats in FEATS.items():
    m = fit(pp_train, feats)
    models[name] = m
    h1, h2 = hazards(m, test, feats)
    p = 1 - (1 - h1) * (1 - h2)
    test["p_" + name] = p
    bt.append(dict(model=name, brier=brier(p, test.y.values), logloss=logloss(p, test.y.values),
                   mean_pred=float(p.mean()), observed=float(test.y.mean()), n=len(test)))
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
    for b in range(5):
        sel = q.values == b
        calib.append(dict(model=name, bin=b + 1, n=int(sel.sum()), mean_pred=float(p[sel].mean()), observed=float(test.y.values[sel].mean())))
bt = pd.DataFrame(bt)
# bootstrap intervals for the differences (test players resampled)
diffs = {n: [] for n in ("raw_pct", "minutes")}
for _ in range(1000):
    i = rng.integers(0, len(test), len(test))
    t = test.iloc[i]
    for n in diffs:
        diffs[n].append(logloss(t.p_new.values, t.y.values) - logloss(t["p_" + n].values, t.y.values))
bt["ll_diff_vs_new_90ci"] = bt.model.map(lambda n: None if n == "new" else
                                          [round(float(np.percentile(diffs[n], 5)), 4), round(float(np.percentile(diffs[n], 95)), 4)])
new_ok = all(bt.set_index("model").loc["new", m] < bt.set_index("model").loc[b, m]
             for b in ("raw_pct", "minutes") for m in ("brier", "logloss"))
best_base = bt[bt.model != "new"].sort_values("logloss").model.iloc[0]
SELECTED = "new" if new_ok else best_base
bt.to_csv(f"{OUT}/backtest_a.csv", index=False)

# ---------- 4b. EXPLORATORY (not pre-registered, added 18 Nov after the backtest): test players with
# zero career MLS minutes through S, where "already in MLS" cannot carry the prediction ----------
z = test[test.mls_cum == 0]
ex = []
for name in FEATS:
    ex.append(dict(model=name, n=len(z), events=int(z.y.sum()), brier=brier(z["p_" + name].values, z.y.values),
                   logloss=logloss(z["p_" + name].values, z.y.values), mean_pred=float(z["p_" + name].mean()),
                   observed=float(z.y.mean())))
ex = pd.DataFrame(ex)
zd = {n: [] for n in ("raw_pct", "minutes")}
rng_ex = np.random.default_rng(20261119)   # own stream: the exploratory check must not shift the pre-registered draws
for _ in range(1000):
    i = rng_ex.integers(0, len(z), len(z))
    t = z.iloc[i]
    for n in zd:
        zd[n].append(logloss(t.p_new.values, t.y.values) - logloss(t["p_" + n].values, t.y.values))
ex["ll_diff_vs_new_90ci"] = ex.model.map(lambda n: None if n == "new" else
                                          [round(float(np.percentile(zd[n], 5)), 4), round(float(np.percentile(zd[n], 95)), 4)])
ex_ok = all(ex.set_index("model").loc["new", m] < ex.set_index("model").loc[b, m]
            for b in ("raw_pct", "minutes") for m in ("brier", "logloss"))
ex.to_csv(f"{OUT}/backtest_a_zero_mls_exploratory.csv", index=False)
print("\nEXPLORATORY: test players with zero career MLS minutes\n", ex.round(4).to_string(index=False), "\nnew beats both:", ex_ok)
pd.DataFrame(calib).to_csv(f"{OUT}/calibration_test.csv", index=False)
print("\nbacktest (a), test cohort 2024:\n", bt.round(4).to_string(index=False))
print("new model passes:", new_ok, "| selected for the page:", SELECTED)
coef_tab = pd.DataFrame({n: m.params for n, m in models.items()}).round(3)
coef_tab.to_csv(f"{OUT}/hazard_coefficients_train.csv")

# ---------- 5. outcome (b) ----------
pw = json.load(open(PATHWAY))["models"]["mlsnp"]
c = np.array(pw["coef"]["gplus_96"]); groups = pw["groups"]


def pathway_pred(df):
    X = np.column_stack([np.ones(len(df)), df.gplus_96.values] + [(df.group == g).astype(float).values for g in groups[1:]])
    return X @ c


def mls_gplus96(pid, s):
    mins = sum(float(mls_min.get((pid, t), 0)) for t in (s + 1, s + 2))
    ga = sum(float(mls_ga.get((pid, t), 0) or 0) for t in (s + 1, s + 2))
    return ga / mins * 96 if mins else np.nan


for df in (train, test):
    df.loc[:, "mls_g96"] = [mls_gplus96(p, s) if y else np.nan for p, s, y in zip(df.player_id, df.season, df.y)]
tb, sb = train[train.y == 1].dropna(subset=["mls_g96"]), test[test.y == 1].dropna(subset=["mls_g96"])
Xb = sm.add_constant(tb[["gplus_eb", "age"]])
mb = sm.WLS(tb.mls_g96, Xb, weights=(tb.m1 + tb.m2)).fit()
res_b = dict(n_train=len(tb), n_test=len(sb))
if len(sb) >= 10:
    pn = mb.predict(sm.add_constant(sb[["gplus_eb", "age"]], has_constant="add"))
    pb = pathway_pred(sb)
    res_b.update(rmse_new=float(np.sqrt(np.mean((pn - sb.mls_g96) ** 2))), rmse_pathway=float(np.sqrt(np.mean((pb - sb.mls_g96) ** 2))))
    res_b["supported"] = res_b["rmse_new"] < res_b["rmse_pathway"]
else:
    res_b["result"] = "insufficient data"
res_b["coef"] = {k: round(float(v), 4) for k, v in mb.params.items()}
json.dump(res_b, open(f"{OUT}/backtest_b.json", "w"), indent=1)
print("\noutcome (b):", res_b)

# ---------- 6. final model on all observed person-periods, bootstrap intervals ----------
feats = FEATS[SELECTED]
pp_all = person_periods(coh[coh.season <= 2025], last_season=2026)
final = fit(pp_all, feats)
cur = coh[coh.season == 2026].copy()
pids = pp_all.player_id.unique()
draws1, draws2 = [], []
pool_stats = pool.groupby(["group", "season"]).gplus_eb.agg(["mean", "std"])
for b in range(B):
    take = rng.choice(pids, len(pids), replace=True)
    bp = pp_all.set_index("player_id").loc[take].reset_index()   # player-level bootstrap of the training data
    pred_df = cur.copy()
    if "gplus_z" in feats:   # the predicted player's own g+ estimate is drawn from its posterior
        gm = pred_df.gplus_eb + rng.standard_normal(len(pred_df)) * pred_df.gplus_sd
        st = pool_stats.loc[list(zip(pred_df.group, pred_df.season))]
        pred_df["gplus_z"] = (gm.values - st["mean"].values) / st["std"].values
    try:
        mb_ = fit(bp, feats)
    except Exception:
        continue
    h1, h2 = hazards(mb_, pred_df, feats)
    draws1.append(h1); draws2.append(h2)
d1, d2 = np.array(draws1), np.array(draws2)
h1, h2 = hazards(final, cur, feats)
cur["p2027"] = h1
cur["p2027_lo"], cur["p2027_hi"] = np.percentile(d1, 5, axis=0), np.percentile(d1, 95, axis=0)
two = 1 - (1 - d1) * (1 - d2)
cur["p_two"] = 1 - (1 - h1) * (1 - h2)
cur["p_two_lo"], cur["p_two_hi"] = np.percentile(two, 5, axis=0), np.percentile(two, 95, axis=0)
cur["name"] = cur.player_id.map(players.name)
cur["timbers2"] = cur.apply(lambda r: t2min.get((r.player_id, 2026), 0) >= MIN_NP, axis=1)
print(f"\nbootstrap refits used: {len(d1)} of {B}")

stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
pred = cur.sort_values("p2027", ascending=False)[["player_id", "name", "group", "age", "minutes", "mls_S", "mls_cum",
                                                   "p2027", "p2027_lo", "p2027_hi", "timbers2"]].round(3)
pred.insert(0, "model", SELECTED); pred.insert(0, "written_utc", stamp)
pred.to_csv(f"{OUT}/predictions_2027.csv", index=False)

# ---------- 7. passport rows for the Timbers2 regulars ----------
regs = cur[cur.timbers2].copy()
proj_b = (mb.predict(sm.add_constant(regs[["gplus_eb", "age"]], has_constant="add")) if res_b.get("supported")
          else pd.Series(pathway_pred(regs), index=regs.index))
regs["proj_mls_g96"] = proj_b.values
regs["proj_source"] = "new model (b)" if res_b.get("supported") else "pathway model (published)"
kp = []
for r in regs.itertuples():
    for k in KPIS:
        kp.append(dict(player_id=r.player_id, player=r.name, group=r.group, kpi=LABEL[k], reliable=k in PASS,
                       raw_per96=getattr(r, k + "_96"), shrunk_per96=getattr(r, k + "_eb"),
                       lo90=getattr(r, k + "_eb") - Z90 * getattr(r, k + "_sd"),
                       hi90=getattr(r, k + "_eb") + Z90 * getattr(r, k + "_sd"),
                       raw_pct=getattr(r, k + "_pct"), age_adj_pct=getattr(r, k + "_agepct")))
kp = pd.DataFrame(kp).round(4)
kp.to_csv(f"{OUT}/passport_kpis.csv", index=False)
regs_out = regs[["player_id", "name", "group", "position", "age", "minutes", "mls_S", "mls_cum", "p_two", "p_two_lo",
                 "p_two_hi", "p2027", "p2027_lo", "p2027_hi", "proj_mls_g96", "proj_source"]].round(3)
regs_out.to_csv(f"{OUT}/passport_players.csv", index=False)
tl = ps[ps.player_id.isin(regs.player_id)][["player_id", "season", "minutes"]].assign(tier="MLS NEXT Pro")
tl = pd.concat([tl, mls[mls.player_id.isin(regs.player_id)].groupby(["player_id", "season"]).minutes.sum().reset_index().assign(tier="MLS")])
tl.to_csv(f"{OUT}/passport_timeline.csv", index=False)

summary = dict(written_utc=stamp, selected=SELECTED, new_passes=new_ok, exploratory_zero_mls_new_beats_both=ex_ok, reliable_kpis=PASS,
               cohorts=coh.groupby("season").size().to_dict(), test_base_rate=float(test.y.mean()),
               regulars=len(regs), outcome_b=res_b, n_bootstrap=len(d1))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print("\n", regs_out[["name", "group", "age", "minutes", "mls_S", "p_two", "p_two_lo", "p_two_hi", "p2027"]].to_string(index=False))
