"""Absence study (P4, placebo): what does losing a player for an outside reason cost, per rating point, by roster rank?

Outside-cause absence of a regular (>= 900 regular-season minutes for that team that season):
  call-up  = did not play, and appeared in a national-team match-day squad within 3 days
  suspension = did not play, and received a red card in the team's previous match
Team-match outcome: non-penalty xG difference for the match (team's view, from stints).
Model (weighted by nothing; one row per team-match):
  xGD = team-season FE + opponent-season FE + home + sum_g beta_g * L_g + e
  L_g = sum over absent players in rank group g of (rating - replacement), replacement = mean rating of that team-season's
        non-regulars weighted by minutes. beta_g = xGD lost per rating point lost (1 = ratings fully accurate).
Groups: rank 1, ranks 2-3, ranks 4-5, ranks 6-11. P4: beta(6-11) > beta(1).
Placebo: permute the absence rows within team-season 500 times (absence counts kept, matches shuffled).
Usage: python src/paper2/absence.py <rating_col>
"""
import gzip, json, glob, sys
import numpy as np, pandas as pd

RCOL = sys.argv[1] if len(sys.argv) > 1 else "rating"
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else "xg"      # "xg" = match xG difference; "market" = closing-odds expected goal difference; "goals" = actual goal difference
rng = np.random.default_rng(11)
ST = pd.read_pickle("out/stints_usa.1.pkl"); SP = pd.read_pickle("out/stint_players_usa.1.pkl")
M = pd.read_pickle("out/espn_matches_usa.1.pkl").set_index("event")
K = pd.read_pickle("out/ratings_mls_goals.pkl")
NT = pd.read_pickle("out/national_team_squads.pkl")
reg_ev = M.index[M.season_type.str.contains("regular", na=False)]
ST = ST[ST.event.isin(reg_ev) & ST.season.between(2013, 2026)]
ev = ST.groupby("event").agg(season=("season", "first"), date=("date", "first"))
ev["d"] = ev.date.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
SP = SP[SP.event.isin(ev.index)].copy(); SP["season"] = SP.event.map(ev.season)

# red cards per event and player (from raw ESPN rosters)
reds = set()
for f in glob.glob("data/espn/usa.1/*.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        for p in r["rosters"]:
            if any(x["rc"] for x in p["plays"]):
                reds.add((r["id"], p["pid"]))

# team-match table with outcome
home_id = M.home_id.to_dict(); away_id = M.away_id.to_dict()
x = ST.groupby("event")[["xg_h", "xg_a"]].sum()
TM = pd.concat([pd.DataFrame({"event": x.index, "team": x.index.map(home_id), "opp": x.index.map(away_id), "home": 1, "xgd": x.xg_h - x.xg_a}),
                pd.DataFrame({"event": x.index, "team": x.index.map(away_id), "opp": x.index.map(home_id), "home": 0, "xgd": x.xg_a - x.xg_h})])
TM["season"] = TM.event.map(ev.season); TM["d"] = TM.event.map(ev.d)
if OUTCOME == "goals":
    GD = (M.hg - M.ag).astype(float)
    TM["xgd"] = np.where(TM.home == 1, TM.event.map(GD), -TM.event.map(GD))
    TM = TM.dropna(subset=["xgd"])
if OUTCOME == "market":
    MK = pd.read_pickle("out/market_gd_mls.pkl").set_index("event").mkt_gd
    TM["xgd"] = np.where(TM.home == 1, TM.event.map(MK), -TM.event.map(MK))
    TM = TM.dropna(subset=["xgd"])
TM = TM.sort_values(["team", "d"]).reset_index(drop=True)
TM["prev_event"] = TM.groupby(["team", "season"]).event.shift(1)

# regulars, ratings, ranks, replacement level
mins = SP.groupby(["season", "team", "pid"])["min"].sum().reset_index()
mins["key"] = mins.pid + "|" + mins.season.astype(str)
mins = mins.merge(K[["key", RCOL]], on="key", how="left").fillna({RCOL: 0})
regs = mins[mins["min"] >= 900].copy()
regs["rank"] = regs.groupby(["season", "team"])[RCOL].rank(ascending=False, method="first").astype(int)
nonreg = mins[mins["min"] < 900]
repl = nonreg.groupby(["season", "team"]).apply(lambda g: np.average(g[RCOL], weights=g["min"]) if g["min"].sum() > 0 else 0,
                                                 include_groups=False).rename("repl")
regs = regs.merge(repl, on=["season", "team"], how="left").fillna({"repl": 0})
regs["above"] = regs[RCOL] - regs.repl
played = set(zip(SP.event, SP.pid))
nt_dates = NT.groupby("pid").date.apply(lambda s: np.array(sorted(s.values), dtype="datetime64[D]"))

def group(rk):
    return "r1" if rk == 1 else ("r2_3" if rk <= 3 else ("r4_5" if rk <= 5 else ("r6_11" if rk <= 11 else None)))

rows = []
R = regs.set_index(["season", "team"])
for tm in TM.itertuples():
    try:
        team_regs = R.loc[(tm.season, tm.team)]
    except KeyError:
        continue
    team_regs = team_regs if isinstance(team_regs, pd.DataFrame) else team_regs.to_frame().T
    for p in team_regs.itertuples():
        if (tm.event, p.pid) in played:
            continue
        a = nt_dates.get(p.pid)
        callup = a is not None and np.any(np.abs((a - np.datetime64(tm.d, "D")).astype(int)) <= 3)
        susp = isinstance(tm.prev_event, str) and (tm.prev_event, p.pid) in reds
        if callup or susp:
            g = group(p.rank)
            if g:
                rows.append({"idx": tm.Index, "group": g, "above": p.above, "why": "callup" if callup else "suspension", "rank": p.rank})
AB = pd.DataFrame(rows)
for g in ("r1", "r2_3", "r4_5", "r6_11"):
    TM["L_" + g] = AB[AB.group == g].groupby("idx").above.sum().reindex(TM.index).fillna(0)
TM["ts"] = TM.team + "_" + TM.season.astype(str); TM["os"] = TM.opp + "_" + TM.season.astype(str)
G = ["L_r1", "L_r2_3", "L_r4_5", "L_r6_11"]


def ols_fe(df, cols):
    """Two-way FE by alternating demeaning (team-season, opponent-season), then OLS with home + cols."""
    Y = df.xgd.values.astype(float); Xc = df[["home"] + cols].values.astype(float)
    a = pd.factorize(df.ts)[0]; b = pd.factorize(df.os)[0]
    def demean(v):
        v = v.copy()
        for _ in range(30):
            v -= (np.bincount(a, v) / np.bincount(a))[a]
            v -= (np.bincount(b, v) / np.bincount(b))[b]
        return v
    Yd = demean(Y); Xd = np.column_stack([demean(Xc[:, j]) for j in range(Xc.shape[1])])
    coef, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
    e = Yd - Xd @ coef
    # cluster-robust by team-season
    XtX = np.linalg.inv(Xd.T @ Xd); meat = np.zeros((Xd.shape[1],) * 2)
    for c in np.unique(a):
        m = a == c; s = Xd[m].T @ e[m]; meat += np.outer(s, s)
    V = XtX @ meat @ XtX
    return dict(zip(["home"] + cols, coef)), dict(zip(["home"] + cols, np.sqrt(np.diag(V))))


coef, se = ols_fe(TM, G)
# placebo: permute absence loads among matches of the same team-season
plac = []
for _ in range(500):
    P = TM.copy()
    for gcol in G:
        P[gcol] = P.groupby("ts")[gcol].transform(lambda v: rng.permutation(v.values))
    c, _ = ols_fe(P, G); plac.append([c[g] for g in G])
plac = np.array(plac)
# bootstrap over team-seasons for the P4 contrast beta(6-11) - beta(1)
tsu = TM.ts.unique(); byts = {t: np.where(TM.ts.values == t)[0] for t in tsu}
bs = []
for _ in range(400):
    pick = rng.choice(tsu, len(tsu)); idx = np.concatenate([byts[t] for t in pick])
    B = TM.iloc[idx].copy(); B["ts"] = B.ts + "_" + np.repeat(np.arange(len(pick)), [len(byts[t]) for t in pick]).astype(str)
    c, _ = ols_fe(B, G); bs.append([c[g] for g in G])
for g in ("r1", "r2_3", "r4_5", "r6_11"):
    TM["N_" + g] = AB[AB.group == g].groupby("idx").size().reindex(TM.index).fillna(0)
cN, sN = ols_fe(TM, ["N_r1", "N_r2_3", "N_r4_5", "N_r6_11"])
bs = np.array(bs)
diff = bs[:, 3] - bs[:, 0]
res = {"rating": RCOL, "outcome": OUTCOME, "n_team_matches": int(len(TM)), "absences": AB.groupby(["group", "why"]).size().unstack(fill_value=0).to_dict(),
       "beta": {g: float(coef[g]) for g in G}, "se": {g: float(se[g]) for g in G},
       "boot_ci": {g: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))] for j, g in enumerate(G)},
       "placebo_sd": {g: float(plac[:, j].std()) for j, g in enumerate(G)},
       "placebo_p": {g: float(np.mean(np.abs(plac[:, j]) >= abs(coef[g]))) for j, g in enumerate(G)},
       "P4_diff_r6_11_minus_r1": [float(np.mean(diff)), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
       "P4_share_boot_diff_positive": float(np.mean(diff > 0)), "home": float(coef["home"]),
       "per_absence": {g: [float(cN[g]), float(sN[g])] for g in cN if g != "home"}}
json.dump(res, open(f"out/absence_{RCOL}_{OUTCOME}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
