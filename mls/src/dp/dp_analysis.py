"""What MLS clubs get from Designated Players, 2013-2026.

Inputs: out/dp_draft_2013_2026.csv (DP list with manual confirmations), ASA salaries, minutes, goals added (g+),
match results. Outputs out/dp_analysis.json and printed tables.
"""
import json, pathlib
import numpy as np, pandas as pd

RAW = pathlib.Path("data/raw")
load = lambda n: json.loads((RAW / f"{n}.json").read_text())
SEASONS = [s for s in range(2013, 2027) if s != 2020]
ACTIONS = ["Dribbling", "Fouling", "Interrupting", "Passing", "Receiving", "Shooting"]
names = {p["player_id"]: p for p in load("players")}
tn = {t["team_id"]: t["team_short_name"] for t in load("teams")}

# ---------- DP table ----------
DP = pd.read_csv("out/dp_draft_2013_2026.csv")
DP = DP[(DP.is_dp_final == "Y") & DP.season.isin(SEASONS)].copy()
DP["salary"] = DP.fall_comp.fillna(DP.spring_comp)
# minutes and clubs from match data for every DP row (rows added by confirmation had none)
mins = {}
for s_ in SEASONS:
    for x in load(f"mls_xgoals_{s_}"):
        t = x["team_id"] if isinstance(x["team_id"], list) else [x["team_id"]]
        mins[(s_, x["player_id"])] = (x["minutes_played"], set(t))
DP["mls_minutes"] = [mins.get((s_, p), (0, set()))[0] for s_, p in zip(DP.season, DP.player_id)]
DP["played_for_club"] = [t in mins.get((s_, p), (0, set()))[1] for s_, p, t in zip(DP.season, DP.player_id, DP.team_id)]
DP["slot_share"] = DP.share_of_season.fillna(1.0)
dead_status = DP.status.fillna("").str.contains("No MLS minutes|Played only for another club|Left before season")
DP["dead"] = dead_status | (DP.mls_minutes == 0) | ~DP.played_for_club
DP["birth_year"] = DP.birth_year.fillna(DP.player_id.map(lambda p: int(names[p]["birth_date"][:4]) if isinstance(p, str) and names.get(p, {}).get("birth_date") else np.nan))
DP["age"] = DP.season - DP.birth_year
first = DP.groupby(["player_id", "team_id"]).season.transform("min")
DP["age_at_signing"] = first - DP.birth_year
DP["age_band"] = pd.cut(DP.age_at_signing, [0, 23, 26, 29, 99], labels=["23 or younger", "24-26", "27-29", "30 or older"])
dp_keys = set(zip(DP.season, DP.player_id))

# ---------- player-seasons with salary + g+ ----------
rows = []
for s in SEASONS:
    sal = pd.DataFrame(load(f"mls_sal_{s}"))
    sal = sal[sal.mlspa_release == sal.mlspa_release.max()]
    sal = sal.dropna(subset=["guaranteed_compensation"]).drop_duplicates("player_id")
    comp = dict(zip(sal.player_id, sal.guaranteed_compensation))
    for x in load(f"mls_ga_{s}"):
        tot = sum(a["goals_added_above_avg"] for a in x["data"])
        rows.append({"season": s, "player_id": x["player_id"], "minutes": x["minutes_played"],
                     "position": x["general_position"], "gplus": tot, "gplus96": tot / x["minutes_played"] * 96 if x["minutes_played"] else np.nan,
                     "salary": comp.get(x["player_id"])})
PS = pd.DataFrame(rows)
PS = PS[PS.position != "GK"]
PS["is_dp"] = [(s, p) in dp_keys for s, p in zip(PS.season, PS.player_id)]
PS["group"] = np.where(PS.is_dp, "DP", np.where(PS.salary >= 1_000_000, "Non-DP, $1M+", np.where(PS.salary >= 400_000, "Non-DP, $400k-$1M", "Non-DP, under $400k")))
# season-relative salary so 2013 and 2026 money compare
PS["rel_salary"] = PS.salary / PS.groupby("season").salary.transform("median")

# ---------- team-season spending ----------
games = []
for s in SEASONS:
    for x in load(f"mls_games_{s}"):
        if x.get("knockout_game") or x.get("status") != "FullTime":
            continue
        for tid, gf, ga in [(x["home_team_id"], x["home_score"], x["away_score"]), (x["away_team_id"], x["away_score"], x["home_score"])]:
            games.append({"season": s, "team_id": tid, "pts": 3 if gf > ga else (1 if gf == ga else 0)})
T = pd.DataFrame(games).groupby(["season", "team_id"]).agg(pts=("pts", "sum"), n=("pts", "size")).reset_index()
T["ppg"] = T.pts / T.n
xg = []
for s in [s for s in SEASONS if s >= 2017]:
    for t in load(f"mls_team_xgoals_{s}"):
        xg.append({"season": s, "team_id": t["team_id"], "xgdpg": t["xgoal_difference"] / t["count_games"]})
T = T.merge(pd.DataFrame(xg), on=["season", "team_id"], how="left")

pay = []
for s in SEASONS:
    sal = pd.DataFrame(load(f"mls_sal_{s}"))
    sal = sal[sal.mlspa_release == sal.mlspa_release.max()].dropna(subset=["guaranteed_compensation"])
    sal["team_id"] = sal.team_id.apply(lambda t: t[-1] if isinstance(t, list) else t)
    sal = sal.dropna(subset=["team_id"])
    for tid, g in sal.groupby("team_id"):
        nondp = g[[(s, p) not in dp_keys for p in g.player_id]].guaranteed_compensation.sum()
        pay.append({"season": s, "team_id": tid, "nondp_pay": nondp})
PAY = pd.DataFrame(pay)
dpt = DP.assign(live=lambda d: d.salary * d.slot_share * (~d.dead), deadm=lambda d: d.salary * d.slot_share * d.dead,
                n_live=lambda d: d.slot_share * (~d.dead)) \
        .groupby(["season", "team_id"]).agg(dp_pay=("live", "sum"), dead_pay=("deadm", "sum"), dp_n=("n_live", "sum"),
                                             dp_missing_salary=("salary", lambda x: x.isna().sum())).reset_index()
TS = T.merge(PAY, on=["season", "team_id"], how="left").merge(dpt, on=["season", "team_id"], how="left")
TS[["dp_pay", "dead_pay", "dp_n"]] = TS[["dp_pay", "dead_pay", "dp_n"]].fillna(0)
TS["team"] = TS.team_id.map(tn)
for c in ["dp_pay", "nondp_pay", "dead_pay"]:
    TS[f"{c}_rel"] = TS[c] / TS.groupby("season")[c].transform("mean").replace(0, np.nan)

out = {}
# A. which money buys points: within-season demeaned regression, $M of each kind
def demean(df, cols):
    return df[cols] - df.groupby("season")[cols].transform("mean")
ts = TS.dropna(subset=["nondp_pay"]).copy()
ts["dp_m"], ts["nondp_m"], ts["dead_m"] = ts.dp_pay / 1e6, ts.nondp_pay / 1e6, ts.dead_pay / 1e6
for era, lo, hi in [("2013-2026", 2013, 2026), ("2013-2019", 2013, 2019), ("2021-2026", 2021, 2026)]:
    e = ts[(ts.season >= lo) & (ts.season <= hi)]
    Y = demean(e, ["ppg"]).ppg.values
    X = demean(e, ["dp_m", "nondp_m"]).values
    b = np.linalg.lstsq(np.column_stack([X]), Y, rcond=None)[0]
    # bootstrap by team-season
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(e), len(e))
        bs.append(np.linalg.lstsq(X[i], Y[i], rcond=None)[0])
    bs = np.array(bs)
    zp = e.groupby("season").ppg.transform(lambda c: (c - c.mean()) / c.std())
    out.setdefault("money", {})[era] = {
        "n": int(len(e)),
        "pts_per_1m_dp": float(b[0] * 34), "ci_dp": (np.percentile(bs[:, 0], [5, 95]) * 34).round(2).tolist(),
        "pts_per_1m_nondp": float(b[1] * 34), "ci_nondp": (np.percentile(bs[:, 1], [5, 95]) * 34).round(2).tolist(),
        "r_dp": float(np.corrcoef(e.groupby("season").dp_pay.transform(lambda c: (c - c.mean()) / c.std()), zp)[0, 1]),
        "r_nondp": float(np.corrcoef(e.groupby("season").nondp_pay.transform(lambda c: (c - c.mean()) / c.std()), zp)[0, 1]),
        "mean_dp_m": float(e.dp_m.mean()), "mean_nondp_m": float(e.nondp_m.mean())}

# B. what a DP produces vs other high earners (2013-2026, outfield, 900+ min)
p = PS[(PS.minutes >= 900) & PS.salary.notna()]
grp = p.groupby("group").agg(n=("gplus", "size"), gplus96=("gplus96", "mean"), gplus96_median=("gplus96", "median"),
                             hit_rate=("gplus96", lambda x: (x > 0).mean()), salary=("salary", "median"),
                             rel_salary=("rel_salary", "median"), gplus_season=("gplus", "mean"))
out["by_group"] = grp.round(4).reset_index().to_dict("records")
# availability: share of DP seasons with 900+ minutes vs other $1M+ players
allp = PS[PS.salary.notna()]
out["availability"] = allp[allp.salary >= 1_000_000].groupby("is_dp").minutes.apply(lambda m: float((m >= 1500).mean())).to_dict()
# goals added per $1M of salary (season-relative: per 10x-median salary unit)
for g, x in p.groupby("group"):
    out.setdefault("gplus_per_1m", {})[g] = float(x.gplus.sum() / (x.salary.sum() / 1e6))

# C. age at signing
dpp = DP.merge(PS[["season", "player_id", "gplus", "gplus96", "minutes", "position"]], on=["season", "player_id"], how="left")
dpp = dpp[~dpp.dead]
age = dpp.groupby("age_band", observed=True).agg(dp_seasons=("player", "size"), players=("player_id", "nunique"),
                                                 salary_median=("salary", "median"),
                                                 gplus96=("gplus96", "mean"), hit_rate=("gplus96", lambda x: (x > 0).mean()),
                                                 minutes=("minutes", "mean"))
age["gplus_per_1m"] = dpp.groupby("age_band", observed=True).apply(lambda x: x.gplus.sum() / (x.salary.sum() / 1e6), include_groups=False)
out["age_bands"] = age.round(4).reset_index().to_dict("records")
# rows for an age chart: one point per DP signing (first season at club), output over his DP spell
spell = dpp.groupby(["player_id", "team_id", "player", "club"], as_index=False).agg(
    age=("age_at_signing", "first"), seasons=("season", "nunique"), first=("season", "min"),
    gplus=("gplus", "sum"), minutes=("minutes", "sum"), salary=("salary", "mean"))
spell["gplus96"] = spell.gplus / spell.minutes * 96
spell.to_csv("out/dp_spells.csv", index=False)
out["spell_by_age"] = spell.groupby(pd.cut(spell.age, [0, 23, 26, 29, 99], labels=["23 or younger", "24-26", "27-29", "30 or older"]), observed=True) \
    .agg(spells=("player", "size"), seasons_per_spell=("seasons", "mean"), gplus96=("gplus96", "mean"),
         total_gplus=("gplus", "mean"), salary=("salary", "median")).round(3).reset_index().to_dict("records")
mia = dpp[dpp.club == "Miami"][["season", "player", "age", "salary", "minutes", "gplus96"]].sort_values(["season", "salary"], ascending=[True, False])
out["miami"] = mia.round(3).to_dict("records")

# D. dead DP money
dead = DP[DP.dead & DP.salary.notna()]
out["dead"] = {"seasons": int(len(dead)), "total_m": float((dead.salary * dead.slot_share).sum() / 1e6),
               "share_of_dp_spend": float((dead.salary * dead.slot_share).sum() / (DP.salary * DP.slot_share).sum()),
               "by_club": (dead.assign(m=dead.salary * dead.slot_share / 1e6).groupby("club").m.sum().sort_values(ascending=False).head(8).round(2).to_dict()),
               "examples": dead.sort_values("salary", ascending=False)[["season", "club", "player", "salary", "status"]].head(10).to_dict("records")}
out["coverage"] = {"dp_seasons": int(len(DP)), "dp_seasons_no_salary": int(DP.salary.isna().sum()),
                   "team_seasons": int(len(TS))}
TS.to_csv("out/dp_team_seasons.csv", index=False)
json.dump(out, open("out/dp_analysis.json", "w"), indent=1, default=str)

pd.set_option("display.width", 220)
print(json.dumps(out["money"], indent=1))
print(grp.round(3).to_string())
print("availability (share 1500+ min among $1M+ earners):", out["availability"])
print("g+ per $1M:", {k: round(v, 3) for k, v in out["gplus_per_1m"].items()})
print(age.round(3).to_string())
print(pd.DataFrame(out["spell_by_age"]).to_string())
print(mia.round(3).to_string())
print(json.dumps({k: v for k, v in out["dead"].items() if k != "examples"}, indent=1)); print(pd.DataFrame(out["dead"]["examples"]).to_string())
print(out["coverage"])
