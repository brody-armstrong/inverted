"""P5 (MLS part) and P6: where between-team differences in quality come from, and what money buys by roster rank.

Team quality Q (team-season) = sum over players of rating x (player minutes / team minutes) x 11  (plain sum, rho = 1,
because the rho approach could not be validated). Contribution of a rank group = the same sum over that group's players.
P5 share of group g = Cov(contribution_g, Q) / Var(Q)  (shares add to 1).
P6: points per $1M at a rank group = 34 x (points per game per unit Q) x (rating per $1M at that group's typical salary)
     x (that group's typical minute share x 11). Rating per $1M from rating = a + c*log(salary) + age + position (regulars),
     so the marginal value is c / salary. Intervals: bootstrap over team-seasons (Q->points) and players (pay curve).
Writes out/roster_rank.json.
"""
import json, glob
import numpy as np, pandas as pd

rng = np.random.default_rng(5)
RCOL = "rating"
K = pd.read_pickle("out/ratings_mls_goals.pkl")
ST = pd.read_pickle("out/stints_usa.1.pkl"); SP = pd.read_pickle("out/stint_players_usa.1.pkl")
M = pd.read_pickle("out/espn_matches_usa.1.pkl").set_index("event")
reg = M.index[M.season_type.str.contains("regular", na=False)]
ev = ST[ST.event.isin(reg)].groupby("event").season.first()
SP = SP[SP.event.isin(ev.index)].copy(); SP["season"] = SP.event.map(ev)
SP = SP[SP.season.between(2013, 2026)]
PM = SP.groupby(["season", "team", "pid"]).agg(min=("min", "sum"), asa_id=("asa_id", "first")).reset_index()
PM["key"] = PM.pid + "|" + PM.season.astype(str)
PM = PM.merge(K[["key", RCOL, "age"]], on="key", how="left").fillna({RCOL: 0})
PM["tmin"] = PM.groupby(["season", "team"])["min"].transform("sum")
PM["c"] = PM[RCOL] * PM["min"] / PM.tmin * 11
PM["regular"] = PM["min"] >= 900
PM["rank"] = PM[PM.regular].groupby(["season", "team"])[RCOL].rank(ascending=False, method="first")
PM["grp"] = np.where(PM["rank"] <= 3, "1-3", np.where(PM["rank"] <= 11, "4-11", "rest"))

Q = PM.groupby(["season", "team"]).c.sum().rename("Q")
C = PM.pivot_table(index=["season", "team"], columns="grp", values="c", aggfunc="sum").fillna(0)
TQ = C.join(Q)
shares = {g: float(np.cov(TQ[g], TQ.Q)[0, 1] / TQ.Q.var()) for g in ("1-3", "4-11", "rest")}

# points per game per unit Q (regular season)
rows = []
for e, r in M.loc[M.index.isin(ev.index)].iterrows():
    hp = 3 if r.hg > r.ag else (1 if r.hg == r.ag else 0); ap = 3 if r.ag > r.hg else (1 if r.hg == r.ag else 0)
    rows += [(ev[e], r.home_id, hp), (ev[e], r.away_id, ap)]
P = pd.DataFrame(rows, columns=["season", "team", "pts"]).groupby(["season", "team"]).pts.mean().rename("ppg")
TQ = TQ.join(P).dropna()


def slope(d):
    return np.polyfit(d.Q, d.ppg, 1)[0]


b_Q = slope(TQ)

# salaries: max guaranteed compensation per player-season
S = pd.concat([pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob("data/raw/mls_sal_*.json"))])
S = S.groupby(["player_id", "season_name"]).guaranteed_compensation.max().reset_index().rename(
    columns={"player_id": "asa_id", "season_name": "season", "guaranteed_compensation": "sal"})
R = PM[PM.regular].merge(S, on=["asa_id", "season"], how="inner")
R = R[R.sal > 0].copy()
R["ls"] = np.log(R.sal); R["agec"] = (R.age.fillna(R.age.median()) - 27) / 5


def pay_slope(d):
    X = np.c_[np.ones(len(d)), d.ls, d.agec, d.agec ** 2, d.season - 2020]
    return np.linalg.lstsq(X, d[RCOL].values, rcond=None)[0][1]


c_pay = pay_slope(R)
grp_sal = R.groupby("grp").sal.median()
grp_share = R.groupby("grp").apply(lambda d: (d["min"] / d.tmin).median(), include_groups=False)


def pts_per_m(bq, cp, g):
    return 34 * bq * (cp / (grp_sal[g] / 1e6)) * grp_share[g] * 11


est = {g: pts_per_m(b_Q, c_pay, g) for g in ("1-3", "4-11")}
ts = TQ.index.values; pl = np.arange(len(R)); bs = []
for _ in range(1000):
    bq = slope(TQ.iloc[rng.integers(0, len(TQ), len(TQ))])
    cp = pay_slope(R.iloc[rng.integers(0, len(R), len(R))])
    bs.append([pts_per_m(bq, cp, "1-3"), pts_per_m(bq, cp, "4-11")])
bs = np.array(bs)
res = {"P5_mls_shares_of_between_team_variance": shares, "team_seasons": int(len(TQ)),
       "ppg_per_unit_Q": float(b_Q), "rating_per_log_salary": float(c_pay),
       "median_salary": {g: float(v) for g, v in grp_sal.items()}, "median_minute_share": {g: float(v) for g, v in grp_share.items()},
       "P6_points_per_1M_per_season": {g: [float(est[g]), float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
                                        for j, g in enumerate(("1-3", "4-11"))},
       "P6_diff_depth_minus_stars": [float(np.mean(bs[:, 1] - bs[:, 0])), float(np.percentile(bs[:, 1] - bs[:, 0], 2.5)),
                                     float(np.percentile(bs[:, 1] - bs[:, 0], 97.5))]}
json.dump(res, open("out/roster_rank.json", "w"), indent=1)
print(json.dumps(res, indent=1))
