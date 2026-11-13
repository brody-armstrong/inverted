"""Task 17: does a rule change cause the spread to move, and what drives MLS home advantage?

A. Before/after with controls. Season outcomes (talent SD = sqrt(observed var - chance var), luck share) for MLS vs the
   11 European leagues. delta = (MLS post - pre) - (Europe post - pre).
   Inference by permutation: every European league in turn plays "treated"; plus placebo event years for MLS.
   Events: TAM (announced mid-2015; first full season 2016), U22 Initiative (2022).
B. Match-level home advantage, MLS regular season, non-neutral venues. Outcomes: goal difference, xG difference, and the
   closing market's expected points difference (does the market price the same things?).
   Spec 1: home team-season and away team-season fixed effects (strength absorbed); identifies effects that vary within
           a team's season: distance, time zones, big altitude climbs, turf the away side doesn't play on, rest, crowd fill,
           empty stadium (the 2021 empty-to-full transition, and 2020's few matches with fans).
   Spec 2: team fixed effects + each side's season xG difference per game; identifies empty stadiums across 2020-21 too.
C. Ghost-game benchmark: European home points share, June 2020 - May 2021 vs other seasons.
Writes out/tests_did.csv, out/tests_did_scan.csv, out/tests_home.csv, out/tests_ghost.json.
"""
import glob, json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(3)
BS = pd.read_csv("out/balance_seasons.csv")
EU = ["Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1", "Eredivisie", "Liga Portugal", "Belgian Pro League",
      "Süper Lig", "Scottish Premiership", "Greek Super League"]
B = BS[BS.league.isin(EU + ["MLS"]) & ~BS.short].copy()
B["talent_sd"] = np.sqrt((B.var_obs - B.var_luck).clip(lower=0))


def did(df, treated, pre, post, y):
    t = df[df.league == treated]; c = df[(df.league != treated)]
    d_t = t[t.season.isin(post)][y].mean() - t[t.season.isin(pre)][y].mean()
    cc = c.groupby("league").apply(lambda g: g[g.season.isin(post)][y].mean() - g[g.season.isin(pre)][y].mean(), include_groups=False)
    return d_t - cc.mean()


EVENTS = {"TAM (2016 on)": (range(2013, 2016), range(2016, 2022)), "U22 Initiative (2022 on)": (range(2016, 2022), range(2022, 2027))}
rows = []
for ev, (pre, post) in EVENTS.items():
    for y in ("talent_sd", "luck_share"):
        d_mls = did(B[B.league.isin(EU + ["MLS"])], "MLS", pre, post, y)
        plac = [did(B[B.league.isin(EU)], lg, pre, post, y) for lg in EU]   # each European league as fake treated
        p = (1 + sum(abs(x) >= abs(d_mls) for x in plac)) / (1 + len(plac))
        rows.append({"event": ev, "outcome": y, "delta_mls": d_mls, "placebo_sd": np.std(plac), "perm_p": p,
                     "mls_pre": B[(B.league == "MLS") & B.season.isin(pre)][y].mean(),
                     "mls_post": B[(B.league == "MLS") & B.season.isin(post)][y].mean()})
DID = pd.DataFrame(rows)
DID.to_csv("out/tests_did.csv", index=False)
print(DID.round(3).to_string(index=False))

# placebo-date scan: split MLS at every year with >= 3 seasons each side (window 3 years before/after)
scan = []
for e in range(2016, 2025):
    pre, post = range(e - 3, e), range(e, e + 3)
    for y in ("talent_sd",):
        d = did(B, "MLS", pre, post, y)
        scan.append({"split_year": e, "outcome": y, "delta": d})
SCAN = pd.DataFrame(scan); SCAN.to_csv("out/tests_did_scan.csv", index=False)
print(SCAN.round(3).to_string(index=False))

# ---------- B. home advantage model ----------
m = pd.read_pickle("out/mls_matches.pkl")
m = m[~m.knockout.astype(bool) & ~m.neutral.astype(bool)].copy()
m["gd"] = m.hg - m.ag
m["xgd"] = m.hxg - m.axg
m["mkt"] = 3 * (m.mkt_H - m.mkt_A)                              # market expected points difference
m["pd_"] = np.select([m.hg > m.ag, m.hg == m.ag], [3, 1], 0) - np.select([m.ag > m.hg, m.hg == m.ag], [3, 1], 0)
m["dist1k"] = m.travel_km / 1000
m["alt_climb"] = (m.elev_gain_m > 1000).astype(float)            # away side climbs > 1 km (Colorado, RSL)
m["turf_mm"] = m.turf_mismatch.astype(float)
m["rest_diff"] = m.home_rest.clip(upper=14) - m.away_rest.clip(upper=14)
m["no_fans"] = (m.attendance == 0) & m.season.isin([2020, 2021])
m.loc[(m.attendance == 0) & ~m.season.isin([2020, 2021]), "attendance"] = np.nan   # zeros later are missing data
m["fill_c"] = m.fill.clip(upper=1.2).where(~m.no_fans, np.nan)
m["hts"] = m.home + "_" + m.season.astype(str); m["ats"] = m.away + "_" + m.season.astype(str)
# season xG difference per game for each side (spec 2 strength control)
side = pd.concat([pd.DataFrame({"t": m.home, "s": m.season, "x": m.xgd}), pd.DataFrame({"t": m.away, "s": m.season, "x": -m.xgd})])
xs = side.groupby(["t", "s"]).x.mean()
m["h_str"] = [xs.get((t, s), np.nan) for t, s in zip(m.home, m.season)]
m["a_str"] = [xs.get((t, s), np.nan) for t, s in zip(m.away, m.season)]

X = "dist1k + tz_crossed + alt_climb + turf_mm + rest_diff + no_fans"
res = []
FITS = {}
for y in ("gd", "xgd", "pd_", "mkt"):
    d = m.dropna(subset=[y, "dist1k", "tz_crossed", "rest_diff"])
    f1 = smf.ols(f"{y} ~ {X} + C(hts) + C(ats)", d).fit(cov_type="HC1")   # clusters = FE groups would be degenerate
    f2 = smf.ols(f"{y} ~ {X} + h_str + a_str + C(home) + C(away)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.hts)[0]})
    dd = d.dropna(subset=["fill_c"])
    f3 = smf.ols(f"{y} ~ dist1k + tz_crossed + alt_climb + turf_mm + rest_diff + fill_c + C(hts) + C(ats)", dd).fit(cov_type="HC1")
    if spec2_keep := (y in ("pd_", "mkt")):
        FITS[y] = f2
    for spec, f in (("team-season FE", f1), ("team FE + xG strength", f2), ("crowd fill, team-season FE", f3)):
        for v in ["dist1k", "tz_crossed", "alt_climb", "turf_mm", "rest_diff", "no_fans[T.True]", "fill_c"]:
            if v in f.params:
                res.append({"outcome": y, "spec": spec, "term": v, "coef": f.params[v], "se": f.bse[v], "p": f.pvalues[v], "n": int(f.nobs)})
HA = pd.DataFrame(res)
HA.to_csv("out/tests_home.csv", index=False)
pd.set_option("display.width", 200)
print(HA.round(3).to_string(index=False))

# raw home edge by crowd status
raw = m.groupby(m.no_fans).agg(n=("gd", "size"), gd=("gd", "mean"), xgd=("xgd", "mean"), pd_=("pd_", "mean"), mkt=("mkt", "mean"))
print(raw.round(3))
# 2021 only: empty early-season matches vs full later in the same season
r21 = m[m.season == 2021].groupby("no_fans").agg(n=("gd", "size"), gd=("gd", "mean"), xgd=("xgd", "mean"), pd_=("pd_", "mean"), mkt=("mkt", "mean"))
print(r21.round(3))

# ---------- C. European ghost games ----------
g = []
for f in glob.glob("data/ext/odds_*_*.csv"):
    if "USA" in f:
        continue
    d = pd.read_csv(f, encoding="latin-1", usecols=lambda c: c in {"Date", "FTHG", "FTAG", "PSCH", "PSCA", "PSCD"}).dropna(subset=["FTHG"])
    d["date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce")
    g.append(d)
g = pd.concat(g)
g["ghost"] = g.date.between("2020-06-01", "2021-05-31")
g["hp"] = 3 * (g.FTHG > g.FTAG) + (g.FTHG == g.FTAG); g["ap"] = 3 * (g.FTAG > g.FTHG) + (g.FTHG == g.FTAG)
for c in ("PSCH", "PSCD", "PSCA"):
    g[c] = pd.to_numeric(g[c], errors="coerce")
inv = 1 / g[["PSCH", "PSCD", "PSCA"]]
pr = inv.div(inv.sum(1), axis=0)
g["mkt_hp"] = 3 * pr.PSCH + pr.PSCD; g["mkt_ap"] = 3 * pr.PSCA + pr.PSCD
GH = {}
for k, x in g.groupby("ghost"):
    GH["ghost" if k else "crowds"] = {"matches": int(len(x)), "home_points_share": float(x.hp.sum() / (x.hp.sum() + x.ap.sum())),
                                      "home_ppg_minus_away": float((x.hp - x.ap).mean()), "home_gd": float((x.FTHG - x.FTAG).mean()),
                                      "market_home_ppg_minus_away": float((x.mkt_hp - x.mkt_ap).mean())}
# MLS equivalents
for k, x in m.groupby("no_fans"):
    GH["mls_" + ("empty" if k else "crowds")] = {"matches": int(len(x)), "home_ppg_minus_away": float(x.pd_.mean()),
                                                 "home_gd": float(x.gd.mean()), "home_xgd": float(x.xgd.mean()),
                                                 "market_home_ppg_minus_away": float(x.mkt.mean())}
json.dump(GH, open("out/tests_ghost.json", "w"), indent=1)
print(json.dumps(GH, indent=1))

# ---------- D. how much of MLS's excess home edge do travel, altitude and turf explain? ----------
# Counterfactual: give every MLS away side European-scale travel (median ~300 km, no time zones, no altitude climb, no turf
# mismatch) and predict the change in the home edge with the spec-2 coefficients. Europe's own edge is the yardstick.
eu_edge = GH["crowds"]["home_ppg_minus_away"]
mls_edge = float(m[~m.no_fans].pd_.mean())
out = {"mls_edge_ppg": mls_edge, "eu_edge_ppg": eu_edge, "excess": mls_edge - eu_edge}
for y in ("pd_", "mkt"):
    f = FITS[y]; b = f.params; Vc = f.cov_params()
    terms = ["dist1k", "tz_crossed", "alt_climb", "turf_mm"]
    dx = np.array([m.dist1k.mean() - 0.3, m.tz_crossed.mean(), m.alt_climb.mean(), m.turf_mm.mean()])
    eff = float(dx @ b[terms].values); se = float(np.sqrt(dx @ Vc.loc[terms, terms].values @ dx))
    out[f"travel_part_{y}"] = {"ppg": eff, "lo": eff - 1.96 * se, "hi": eff + 1.96 * se, "share_of_excess": eff / (mls_edge - eu_edge)}
# crowd natural experiment: change in home edge without fans, MLS vs Europe (bootstrap over matches)
def boot_diff(a, b, B=4000):
    a, b = np.asarray(a), np.asarray(b)
    return np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)])
mm = m.dropna(subset=["pd_"])
d_mls = boot_diff(mm[mm.no_fans].pd_, mm[~mm.no_fans].pd_)
gg = g.dropna(subset=["hp"])
d_eu = boot_diff((gg[gg.ghost].hp - gg[gg.ghost].ap), (gg[~gg.ghost].hp - gg[~gg.ghost].ap), B=1000)
d_mls_x = boot_diff(mm[mm.no_fans].xgd.dropna(), mm[~mm.no_fans].xgd.dropna())
out["no_fans_change_mls_ppg"] = {"mean": float(mm[mm.no_fans].pd_.mean() - mm[~mm.no_fans].pd_.mean()), "lo": float(np.percentile(d_mls, 2.5)), "hi": float(np.percentile(d_mls, 97.5))}
out["no_fans_change_mls_xgd"] = {"mean": float(mm[mm.no_fans].xgd.mean() - mm[~mm.no_fans].xgd.mean()), "lo": float(np.percentile(d_mls_x, 2.5)), "hi": float(np.percentile(d_mls_x, 97.5))}
out["no_fans_change_eu_ppg"] = {"mean": float(np.mean(d_eu)), "lo": float(np.percentile(d_eu, 2.5)), "hi": float(np.percentile(d_eu, 97.5))}
json.dump(out, open("out/tests_home_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
