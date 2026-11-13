"""Team analysis: what separates winning MLS teams, and where each 2026 squad falls short.

1. Team-season table 2017-2026 (style, chance quality, g+ for/against by action).
2. Drivers of points per game (within-season z-scores, pooled) and what predicts next season.
3. Squad quality by position group (minutes-weighted player g+/96) and its link to points.
4. 2026 gap report per club + targets from the MLS profiler and the pathway projections.
"""
import json, pathlib
import numpy as np, pandas as pd

RAW = pathlib.Path("data/raw")
load = lambda n: json.loads((RAW / f"{n}.json").read_text())
ACTIONS = ["Dribbling", "Fouling", "Interrupting", "Passing", "Receiving", "Shooting"]
GROUPS = {"CB": "Centre-backs", "FB": "Fullbacks", "DM": "Midfield", "CM": "Midfield",
          "AM": "Midfield", "W": "Wingers", "ST": "Strikers"}
GORDER = ["Centre-backs", "Fullbacks", "Midfield", "Wingers", "Strikers"]

# ---------- 1. team-season table ----------
rows = []
for s in range(2017, 2027):
    xg = {t["team_id"]: t for t in load(f"mls_team_xgoals_{s}")}
    xp = {t["team_id"]: t for t in load(f"mls_team_xpass_{s}")}
    ga = {t["team_id"]: {a["action_type"]: a for a in t["data"]} for t in load(f"mls_team_goals-added_{s}")}
    for tid, t in xg.items():
        g = t["count_games"]
        p, a = xp[tid], ga[tid]
        r = {"team_id": tid, "season": s, "games": g, "points": t["points"], "ppg": t["points"] / g,
             "xpts_pg": t["xpoints"] / g, "gd_pg": t["goal_difference"] / g, "xgd_pg": t["xgoal_difference"] / g,
             "xgf_pg": t["xgoals_for"] / g, "xga_pg": t["xgoals_against"] / g,
             "shots_for_pg": t["shots_for"] / g, "shots_against_pg": t["shots_against"] / g,
             "xg_per_shot": t["xgoals_for"] / t["shots_for"], "xga_per_shot": t["xgoals_against"] / t["shots_against"],
             "finishing_pg": (t["goals_for"] - t["xgoals_for"]) / g,
             "keeping_pg": (t["xgoals_against"] - t["goals_against"]) / g,
             "possession": p["attempted_passes_for"] / (p["attempted_passes_for"] + p["attempted_passes_against"]),
             "pass_acc_oe": p["passes_completed_over_expected_p100_for"],
             "opp_pass_acc_oe": p["passes_completed_over_expected_p100_against"],
             "verticality": p["avg_vertical_distance_for"], "opp_verticality": p["avg_vertical_distance_against"]}
        for act in ACTIONS:
            r[f"gpf_{act}"] = a[act]["goals_added_for"] / g
            r[f"gpa_{act}"] = a[act]["goals_added_against"] / g
            r[f"gpnet_{act}"] = r[f"gpf_{act}"] - r[f"gpa_{act}"]
        rows.append(r)
T = pd.DataFrame(rows)
names = {t["team_id"]: (t["team_short_name"], t["team_abbreviation"]) for t in load("teams")}
T["team"] = T.team_id.map(lambda i: names[i][0])
T["abbr"] = T.team_id.map(lambda i: names[i][1])
T = T[T.games >= 10]  # skip tiny samples

# ---------- 2. drivers of points ----------
FEATS = {"xgd_pg": "xG difference", "xgf_pg": "xG for", "xga_pg": "xG against", "shots_for_pg": "Shots for",
         "shots_against_pg": "Shots against", "xg_per_shot": "xG per shot", "xga_per_shot": "xG per shot conceded",
         "possession": "Possession (pass share)", "pass_acc_oe": "Pass accuracy over expected",
         "opp_pass_acc_oe": "Opponent pass accuracy over expected", "verticality": "Pass verticality",
         "opp_verticality": "Opponent pass verticality", "finishing_pg": "Finishing (G − xG)",
         "keeping_pg": "Shot-stopping (xGA − GA)",
         **{f"gpnet_{a}": f"Net g+ {a}" for a in ACTIONS}}
z = T.groupby("season")[list(FEATS) + ["ppg"]].transform(lambda c: (c - c.mean()) / c.std())
drivers = sorted(({"key": k, "label": v, "r": float(np.corrcoef(z[k], z.ppg)[0, 1])} for k, v in FEATS.items()),
                 key=lambda d: -abs(d["r"]))

# persistence: what predicts next season's points?
nxt = T[["team_id", "season", "ppg"]].assign(season=lambda x: x.season - 1).rename(columns={"ppg": "ppg_next"})
TP = T.merge(nxt, on=["team_id", "season"])
persist = {lab: float(np.corrcoef(TP[k], TP.ppg_next)[0, 1]) for k, lab in
           [("ppg", "Points per game"), ("gd_pg", "Goal difference"), ("xgd_pg", "xG difference"), ("xpts_pg", "Expected points")]}

# ---------- 3. squad quality by position group ----------
P = pd.read_pickle("out/all_seasons.pkl")
P = P[(P.league == "mls")].copy()
P["group"] = P.general_position.map(GROUPS)
P = P.dropna(subset=["group", "gplus_96"])
P["gplus_tot"] = P.gplus_96 * P.minutes_played / 96
sq = (P.groupby(["team_id", "season", "group"])
        .agg(gp=("gplus_tot", "sum"), mins=("minutes_played", "sum")).reset_index())
sq["q"] = sq.gp / sq.mins * 96  # minutes-weighted g+/96 of everyone who played there
SQ = sq.pivot_table(index=["team_id", "season"], columns="group", values="q").reset_index()
TS = T.merge(SQ, on=["team_id", "season"], how="left")
zz = TS.groupby("season")[GORDER + ["ppg"]].transform(lambda c: (c - c.mean()) / c.std()).dropna()
X = np.column_stack([np.ones(len(zz))] + [zz[g] for g in GORDER])
beta = np.linalg.lstsq(X, zz.ppg, rcond=None)[0][1:]
r2 = 1 - ((zz.ppg - X @ np.r_[0, beta]) ** 2).sum() / ((zz.ppg - zz.ppg.mean()) ** 2).sum()
group_weight = dict(zip(GORDER, beta.tolist()))
# convert to raw units: ppg per +0.01 g+/96 in that group
ppg_sd = TS.groupby("season").ppg.std().mean()
q_sd = {g: TS.groupby("season")[g].std().mean() for g in GORDER}
ppg_per_001 = {g: group_weight[g] * ppg_sd / q_sd[g] * 0.01 for g in GORDER}

# ---------- 4. 2026 gap report ----------
cur = TS[TS.season == 2026].copy().sort_values("ppg", ascending=False).reset_index(drop=True)
cur["rank"] = np.arange(1, len(cur) + 1)
top = cur.head(8)
prev = T[T.season == 2025].set_index("team_id")
cur["xgd_change"] = cur.apply(lambda r: r.xgd_pg - prev.xgd_pg.get(r.team_id, np.nan), axis=1)

mls26 = pd.read_csv("out/players_2026.csv")
roles_by_group = mls26.groupby("group").role.unique().to_dict()
role_min = mls26.groupby(["team_id", "group", "role"]).minutes_played.sum().reset_index()
group_min = mls26.groupby(["team_id", "group"]).minutes_played.sum().rename("gmin").reset_index()
role_min = role_min.merge(group_min, on=["team_id", "group"])
role_min["share"] = role_min.minutes_played / role_min.gmin
top_ids = set(top.team_id)
role_top = (role_min[role_min.team_id.isin(top_ids)].groupby(["group", "role"]).share.sum()
            / len(top_ids)).to_dict()
bottom_ids = set(cur.tail(8).team_id)
role_bot = (role_min[role_min.team_id.isin(bottom_ids)].groupby(["group", "role"]).share.sum()
            / len(bottom_ids)).to_dict()

PW = json.load(open("out/pathway.json"))
mls_idx = {pid: i for i, pid in enumerate(mls26.player_id)}
pros = pd.DataFrame(PW["prospects"]).reset_index().rename(columns={"index": "pi"})

teams_out = []
for _, t in cur.iterrows():
    gaps = []
    for g in GORDER:
        if pd.isna(t[g]):
            continue
        deficit = t[g] - top[g].mean()
        gaps.append({"group": g, "q": round(float(t[g]), 4), "top": round(float(top[g].mean()), 4),
                     "league": round(float(cur[g].mean()), 4),
                     "ppg_cost": round(float(min(deficit, 0) / 0.01 * ppg_per_001[g]), 3)})
    gaps.sort(key=lambda x: x["ppg_cost"])
    team_roles = role_min[role_min.team_id == t.team_id]
    recs = []
    for gp in gaps[:2]:
        g = gp["group"]
        if gp["ppg_cost"] >= 0:
            continue
        # role the top teams lean on that this team barely uses
        shares = {r: float(team_roles[(team_roles.group == g) & (team_roles.role == r)].share.sum()) for r in roles_by_group[g]}
        missing = max(roles_by_group[g], key=lambda r: role_top.get((g, r), 0) - shares[r])
        cand = mls26[(mls26.group == g) & (mls26.team_id != t.team_id) & (mls26.minutes_played >= 1500)
                     & (mls26.guaranteed_compensation <= 1_000_000) & (mls26.gplus_96_adj > gp["q"])]
        cand = cand.assign(role_fit=(cand.role == missing).astype(int))
        cand = cand.sort_values(["role_fit", "value_season"], ascending=False).head(4)
        pr = pros[(pros.g == g) & (pros.lg != "mlsnp") & (pros.a.fillna(99) <= 26)].sort_values("pct", ascending=False).head(3)
        used = set(); 
        recs.append({"group": g, "role_gap": missing, "role_share": round(shares[missing], 2),
                     "role_top": round(role_top.get((g, missing), 0), 2),
                     "mls": [mls_idx[pid] for pid in cand.player_id], "pathway": pr.pi.tolist()})
    teams_out.append({
        "id": t.team_id, "team": t.team, "abbr": t.abbr, "rank": int(t["rank"]), "pts": int(t.points),
        "ppg": round(float(t.ppg), 2), "xpts": round(float(t.xpts_pg * t.games), 1),
        "xgf": round(float(t.xgf_pg), 2), "xga": round(float(t.xga_pg), 2), "xgd": round(float(t.xgd_pg), 2),
        "fin": round(float(t.finishing_pg * t.games), 1), "keep": round(float(t.keeping_pg * t.games), 1),
        "poss": round(float(t.possession), 3), "vert": round(float(t.verticality), 2),
        "opp_acc": round(float(t.opp_pass_acc_oe), 2), "xg_shot": round(float(t.xg_per_shot), 3),
        "xgd_change": None if pd.isna(t.xgd_change) else round(float(t.xgd_change), 2),
        "gaps": gaps, "recs": recs})

# league-wide style percentiles for the team cards
STYLE_T = {"possession": "Possession", "verticality": "Directness", "shots_for_pg": "Shot volume",
           "xg_per_shot": "Shot quality", "shots_against_pg": "Shots conceded",
           "opp_pass_acc_oe": "Opponent passing comfort"}
for k in STYLE_T:
    cur[f"pct_{k}"] = cur[k].rank(pct=True).mul(100).round()
for i, t in enumerate(teams_out):
    t["style"] = [int(cur.loc[i, f"pct_{k}"]) for k in STYLE_T]

# best vs worst style contrast, pooled 2017-2026: top quarter vs bottom quarter by ppg within season
T["tier"] = T.groupby("season").ppg.transform(lambda c: pd.qcut(c.rank(method="first"), 4, labels=False))
contrast = []
for k, lab in FEATS.items():
    hi, lo = T[T.tier == 3][k].mean(), T[T.tier == 0][k].mean()
    contrast.append({"label": lab, "top": float(hi), "bottom": float(lo), "all": float(T[k].mean()),
                     "d": float((hi - lo) / T[k].std())})
contrast.sort(key=lambda c: -abs(c["d"]))

out = {"n_team_seasons": int(len(T)), "drivers": drivers, "persist": persist, "contrast": contrast,
       "group_weight": group_weight, "ppg_per_001": ppg_per_001, "squad_r2": float(r2),
       "role_top": {f"{g}|{r}": v for (g, r), v in role_top.items()},
       "role_bot": {f"{g}|{r}": v for (g, r), v in role_bot.items()},
       "style_labels": list(STYLE_T.values()), "teams": teams_out}
json.dump(out, open("out/teams.json", "w"), separators=(",", ":"))

print(len(T), "team-seasons")
print("DRIVERS:", [(d["label"], round(d["r"], 2)) for d in drivers[:12]])
print("PERSIST:", {k: round(v, 2) for k, v in persist.items()})
print("SQUAD weights (z):", {k: round(v, 2) for k, v in group_weight.items()}, "R2", round(r2, 2))
print("ppg per +0.01 g+/96:", {k: round(v, 3) for k, v in ppg_per_001.items()})
print("CONTRAST:", [(c["label"], round(c["d"], 2)) for c in contrast[:10]])
print("ROLES top vs bottom:", {f"{g}|{r}": (round(v, 2), round(role_bot.get((g, r), 0), 2)) for (g, r), v in role_top.items()})
for t in teams_out:
    g = t["gaps"][0]
    rec = t["recs"][0] if t["recs"] else {}
    print(f"{t['rank']:2d} {t['team']:15s} {t['pts']:3d}pts xGD {t['xgd']:+.2f}  worst {g['group']:12s} cost {g['ppg_cost']:+.3f}  "
          f"role gap {rec.get('role_gap')} ({rec.get('role_share')} vs {rec.get('role_top')}) "
          f"MLS: {[mls26.player_name.iloc[i] for i in rec.get('mls', [])][:3]}")
