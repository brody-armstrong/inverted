"""MLS wage bills 2013-2026: how much spending buys, star-heavy vs balanced payrolls, trend over time."""
import json, pathlib
import numpy as np, pandas as pd

RAW = pathlib.Path("data/raw")
load = lambda n: json.loads((RAW / f"{n}.json").read_text())

# team points from match results (regular season)
rows = []
for s in range(2013, 2027):
    for x in load(f"mls_games_{s}"):
        if x.get("knockout_game") or x.get("status") != "FullTime":
            continue
        for tid, gf, ga in [(x["home_team_id"], x["home_score"], x["away_score"]), (x["away_team_id"], x["away_score"], x["home_score"])]:
            rows.append({"season": s, "team_id": tid, "pts": 3 if gf > ga else (1 if gf == ga else 0), "gd": gf - ga})
R = pd.DataFrame(rows).groupby(["season", "team_id"]).agg(pts=("pts", "sum"), gd=("gd", "sum"), n=("pts", "size")).reset_index()
R["ppg"], R["gdpg"] = R.pts / R.n, R.gd / R.n

# payroll: each season's latest MLSPA release, guaranteed compensation
pay = []
for s in range(2013, 2027):
    d = pd.DataFrame(load(f"mls_sal_{s}"))
    d = d[d.mlspa_release == d.mlspa_release.max()]
    d["team_id"] = d.team_id.apply(lambda t: t[-1] if isinstance(t, list) else t)
    d = d.dropna(subset=["team_id", "guaranteed_compensation"])
    for tid, t in d.groupby("team_id"):
        c = t.guaranteed_compensation.sort_values(ascending=False)
        pay.append({"season": s, "team_id": tid, "release": t.mlspa_release.iloc[0], "payroll": c.sum(), "top3": c.head(3).sum(),
                    "rest": c.iloc[3:].sum(), "n_players": len(c), "median": c.median(), "top1": c.iloc[0]})
PAY = pd.DataFrame(pay)
D = R.merge(PAY, on=["season", "team_id"])
D = D[D.season != 2020]
lg = D.groupby("season")
D["rel_pay"] = D.payroll / lg.payroll.transform("mean")
D["rel_rest"] = D.rest / lg.rest.transform("mean")
D["rel_top3"] = D.top3 / lg.top3.transform("mean")
D["top3_share"] = D.top3 / D.payroll
D["log_rel_pay"] = np.log(D.rel_pay)

# xG difference for 2017+ (ASA team xgoals) to separate spending from luck
xg = []
for s in range(2017, 2027):
    for t in load(f"mls_team_xgoals_{s}"):
        xg.append({"season": s, "team_id": t["team_id"], "xgdpg": t["xgoal_difference"] / t["count_games"]})
D = D.merge(pd.DataFrame(xg), on=["season", "team_id"], how="left")

names = {t["team_id"]: t["team_short_name"] for t in load("teams")}
D["team"] = D.team_id.map(names)


def z(c):
    return (c - c.mean()) / c.std()


out = {}
# 1. how much does spending explain? (within-season)
zz = D.assign(zp=D.groupby("season").ppg.transform(z), zl=D.groupby("season").log_rel_pay.transform(z))
out["r_pay_ppg"] = float(np.corrcoef(zz.zl, zz.zp)[0, 1])
sub = D.dropna(subset=["xgdpg"])
zx = sub.assign(zx=sub.groupby("season").xgdpg.transform(z), zl=sub.groupby("season").log_rel_pay.transform(z))
out["r_pay_xgd"] = float(np.corrcoef(zx.zl, zx.zx)[0, 1])
# points per doubling of payroll relative to league average
b = np.polyfit(D.log_rel_pay, D.ppg, 1)
out["ppg_per_doubling"] = float(b[0] * np.log(2))
out["pts_per_doubling_34"] = float(b[0] * np.log(2) * 34)

# by era
eras = {"2013-16": (2013, 2016), "2017-19": (2017, 2019), "2021-23": (2021, 2023), "2024-26": (2024, 2026)}
out["eras"] = {}
for k, (a, c) in eras.items():
    e = zz[(zz.season >= a) & (zz.season <= c)]
    ee = D[(D.season >= a) & (D.season <= c)]
    out["eras"][k] = {"r": float(np.corrcoef(e.zl, e.zp)[0, 1]), "n": int(len(e)),
                      "spread_max_min": float(ee.groupby("season").apply(lambda t: t.payroll.max() / t.payroll.min(), include_groups=False).mean()),
                      "median_payroll": float(ee.payroll.median()), "top3_share": float(ee.top3_share.mean())}

# 2. stars vs depth: regress ppg on log relative top-3 and log relative rest (within-season z)
zz["zt"] = D.groupby("season").rel_top3.transform(lambda c: z(np.log(c)))
zz["zr"] = D.groupby("season").rel_rest.transform(lambda c: z(np.log(c)))
X = np.column_stack([np.ones(len(zz)), zz.zt, zz.zr])
beta = np.linalg.lstsq(X, zz.zp, rcond=None)[0]
out["stars_vs_depth_beta"] = {"top3": float(beta[1]), "rest": float(beta[2])}
out["r_top3"] = float(np.corrcoef(zz.zt, zz.zp)[0, 1])
out["r_rest"] = float(np.corrcoef(zz.zr, zz.zp)[0, 1])
# concentration at similar total spend: residual of ppg on payroll vs top3 share
res = zz.zp - np.polyval(np.polyfit(zz.zl, zz.zp, 1), zz.zl)
out["r_concentration_given_pay"] = float(np.corrcoef(D.top3_share, res)[0, 1])
D["conc_tier"] = pd.qcut(D.top3_share, 3, labels=["Balanced", "Middle", "Star-heavy"])
out["conc_tiers"] = {str(k): {"n": len(v), "top3_share": float(v.top3_share.mean()), "rel_pay": float(v.rel_pay.mean()),
                              "ppg": float(v.ppg.mean()), "resid": float(res[v.index].mean())} for k, v in D.groupby("conc_tier", observed=True)}

# 3. payroll quartiles -> outcome
D["pay_q"] = D.groupby("season").rel_pay.transform(lambda c: pd.qcut(c.rank(method="first"), 4, labels=False))
out["pay_quartiles"] = D.groupby("pay_q").agg(ppg=("ppg", "mean"), rel_pay=("rel_pay", "mean"),
                                              top_half=("ppg", lambda s: 0)).ppg.round(3).to_dict()
D["rank"] = D.groupby("season").ppg.rank(ascending=False)
D["n_teams"] = D.groupby("season").ppg.transform("size")
D["top_third"] = D["rank"] <= D.n_teams / 3
out["top_third_rate_by_pay_q"] = D.groupby("pay_q").top_third.mean().round(3).to_dict()

# 4. efficiency 2026: points vs payroll
cur = D[D.season == 2026].copy()
f = np.polyfit(D.log_rel_pay, D.ppg, 1)
cur["exp_pts"] = np.polyval(f, cur.log_rel_pay) * cur.n
cur["pts_vs_pay"] = cur.pts - cur.exp_pts
out["eff_2026"] = cur.sort_values("pts_vs_pay", ascending=False)[["team", "pts", "payroll", "top3_share", "exp_pts", "pts_vs_pay", "xgdpg"]].round(3).to_dict("records")
out["season_payroll"] = D.groupby("season").agg(median=("payroll", "median"), max=("payroll", "max"), min=("payroll", "min"),
                                                top3_share=("top3_share", "mean")).round(3).reset_index().to_dict("records")
out["fit"] = f.tolist()
out["releases"] = PAY.groupby("season").release.first().to_dict()
D.to_csv("out/mls_payroll.csv", index=False)
json.dump(out, open("out/spending.json", "w"), indent=1, default=str)

print("r(pay, ppg)", round(out["r_pay_ppg"], 2), "| r(pay, xGD)", round(out["r_pay_xgd"], 2), "| pts per doubling payroll", round(out["pts_per_doubling_34"], 1))
print("eras", json.dumps(out["eras"], indent=0))
print("stars vs depth", out["stars_vs_depth_beta"], "r_top3", round(out["r_top3"], 2), "r_rest", round(out["r_rest"], 2), "conc|pay", round(out["r_concentration_given_pay"], 2))
print("conc tiers", json.dumps(out["conc_tiers"], indent=0))
print("pay quartile ppg", out["pay_quartiles"], "top-third rate", out["top_third_rate_by_pay_q"])
print(pd.DataFrame(out["eff_2026"]).to_string())
print(pd.DataFrame(out["season_payroll"]).to_string())
print(out["releases"])
