"""Chain USL1 -> USLC -> MLS, test the chain on direct USL1 -> MLS moves, project 2026 players."""
import json
import numpy as np, pandas as pd

M = json.load(open("out/pathway_model.json"))
STATS = list(M["labels"])
GROUPS = {"CB": "Centre-backs", "FB": "Fullbacks", "DM": "Midfield", "CM": "Midfield",
          "AM": "Midfield", "W": "Wingers", "ST": "Strikers"}
d = pd.read_pickle("out/all_seasons.pkl")
d["group"] = d.general_position.map(GROUPS)


def wls(X, y, w):
    W = np.sqrt(w)[:, None]
    return np.linalg.lstsq(X * W, y * W.ravel(), rcond=None)[0]


def design(x, grp, groups):
    return np.column_stack([np.ones(len(x)), x] + [(grp == g).astype(float) for g in groups[1:]])


def fit_pairs(src, dst, src_min=900, dst_min=900):
    a = d[(d.league == src) & (d.minutes_played >= src_min)]
    b = d[(d.league == dst) & (d.minutes_played >= dst_min)]
    out = []
    for lag in (0, 1):
        bb = b.assign(season=b.season - lag)
        p = a.merge(bb, on=["player_id", "season"], suffixes=("_lo", "_hi")).assign(lag=lag)
        out.append(p)
    p = pd.concat(out).sort_values("lag").drop_duplicates(["player_id", "season"])
    p["w"] = 2 / (1 / p.minutes_played_lo + 1 / p.minutes_played_hi)
    p["grp"] = p.group_hi.fillna(p.group_lo)
    return p.dropna(subset=["grp"])


def project(df, coefs, groups, suffix=""):
    out = pd.DataFrame(index=df.index)
    for s in STATS:
        out[s] = design(df[s].values, df.group.values, groups) @ np.array(coefs[s])
    return out


# --- USL1 -> USLC step ---
p = fit_pairs("usl1", "uslc").dropna(subset=[f"{s}_lo" for s in STATS] + [f"{s}_hi" for s in STATS])
g1 = sorted(p.grp.unique())
step = {s: wls(design(p[f"{s}_lo"].values, p.grp.values, g1), p[f"{s}_hi"].values, p.w.values).tolist() for s in STATS}
factors_step = {s: float(np.average(p[f"{s}_hi"], weights=p.w) / np.average(p[f"{s}_lo"], weights=p.w))
                for s in ["xg_96", "xa_96", "passes_96", "int_96", "drib_96"]}
print("USL1 -> USLC pairs:", len(p), {k: round(v, 2) for k, v in factors_step.items()})

uslc = M["models"]["uslc"]


def chain_usl1(df):
    mid = project(df, step, g1)
    mid["group"] = df.group
    return project(mid, uslc["coef"], uslc["groups"])


# --- test the chain on direct USL1 -> MLS pairs (never used to fit the chain) ---
direct = fit_pairs("usl1", "mls", 900, 450).dropna(subset=[f"{s}_lo" for s in STATS] + [f"{s}_hi" for s in STATS])
lo = direct[[f"{s}_lo" for s in STATS]].copy()
lo.columns = STATS
lo["group"] = direct.grp.values
pred = chain_usl1(lo)
mls_pos_mean = d[(d.league == "mls") & (d.minutes_played >= 450)].groupby("group")[STATS].mean()
chain_test = {}
for s in STATS:
    y = direct[f"{s}_hi"].values
    e_chain = np.mean(np.abs(pred[s].values - y))
    e_mean = np.mean(np.abs(direct.grp.map(mls_pos_mean[s]).values - y))
    chain_test[s] = {"mae_chain": float(e_chain), "mae_position_mean": float(e_mean)}
print("chain test on", len(direct), "direct USL1->MLS moves:",
      {M["labels"][s]: f"{v['mae_chain']:.3f} vs {v['mae_position_mean']:.3f}" for s, v in chain_test.items()})

# --- 2026 projections ---
cur = d[(d.league != "mls") & (d.season == 2026) & (d.minutes_played >= 900)].dropna(subset=STATS + ["group"]).copy()
proj = []
for lg, sub in cur.groupby("league"):
    if lg == "usl1":
        pr = chain_usl1(sub)
    else:
        mdl = M["models"][lg]
        pr = project(sub, mdl["coef"], mdl["groups"])
    pr.columns = [f"proj_{c}" for c in pr.columns]
    proj.append(sub.join(pr))
cur = pd.concat(proj)

mls26 = d[(d.league == "mls") & (d.season == 2026) & (d.minutes_played >= 900)]
for s in STATS:
    cur[f"mlspct_{s}"] = [
        round(100 * (mls26[mls26.group == g][s] < v).mean()) if (mls26.group == g).any() else np.nan
        for g, v in zip(cur.group, cur[f"proj_{s}"])]
mls_minutes = d[(d.league == "mls") & (d.season == 2026)].set_index("player_id").minutes_played
cur["mls_min_2026"] = cur.player_id.map(mls_minutes).fillna(0).astype(int)

teams = {}
import pathlib
for lg in ["uslc", "usl1", "mlsnp"]:
    f = pathlib.Path(f"data/raw/{lg}_teams.json")
    if f.exists():
        teams.update({t["team_id"]: t["team_short_name"] for t in json.loads(f.read_text())})
cur["team"] = cur.team_id.map(teams)
cur.to_pickle("out/pathway_2026.pkl")
json.dump({"chain_test": chain_test, "n_direct": int(len(direct)), "usl1_uslc_pairs": int(len(p)),
           "usl1_uslc_factors": factors_step}, open("out/pathway_chain.json", "w"), indent=1)

cols = ["player_name", "league", "general_position", "age", "minutes_played", "gplus_96", "proj_gplus_96",
        "mlspct_gplus_96", "proj_xg_96", "mls_min_2026"]
print(cur.groupby("league").size().to_dict())
print(cur.sort_values("proj_gplus_96", ascending=False)[cols].head(25).round(3).to_string())
