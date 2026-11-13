"""Inverted — MLS player profiling.

Builds per-96 player fingerprints from ASA data, clusters players into
data-driven roles within position groups, computes quality percentiles,
nearest-neighbour comparables and value vs. salary.
"""
import json, sys, pathlib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import HuberRegressor

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
MIN_MINUTES = 900
RAW = pathlib.Path("data/raw")
load = lambda n: json.loads((RAW / f"{n}.json").read_text())

GROUPS = {"CB": "Centre-backs", "FB": "Fullbacks", "DM": "Midfield", "CM": "Midfield",
          "AM": "Midfield", "W": "Wingers", "ST": "Strikers"}
ROLE_K = {"Centre-backs": 3, "Fullbacks": 3, "Midfield": 4, "Wingers": 3, "Strikers": 3}
ACTIONS = ["Dribbling", "Fouling", "Interrupting", "Passing", "Receiving", "Shooting"]

# ---------- assemble ----------
players = pd.DataFrame(load("players"))[["player_id", "player_name", "birth_date", "nationality"]]
teams = pd.DataFrame(load("teams"))[["team_id", "team_short_name", "team_abbreviation"]]
xg = pd.DataFrame(load(f"xgoals_{SEASON}"))
xg["traded"] = xg.team_id.apply(lambda t: isinstance(t, list))
xg["team_id"] = xg.team_id.apply(lambda t: t[-1] if isinstance(t, list) else t)  # most recent club
xp = pd.DataFrame(load(f"xpass_{SEASON}")).drop(columns=["team_id", "general_position", "minutes_played"])
ga_rows = []
for p in load(f"ga_{SEASON}"):
    r = {"player_id": p["player_id"]}
    for a in p["data"]:
        r[f"ga_{a['action_type']}"] = a["goals_added_above_avg"]
        r[f"gaRaw_{a['action_type']}"] = a["goals_added_raw"]
        r[f"n_{a['action_type']}"] = a["count_actions"]
    ga_rows.append(r)
ga = pd.DataFrame(ga_rows)
sal = pd.DataFrame(load(f"sal_{SEASON}"))
sal = sal.sort_values("mlspa_release").groupby("player_id").last()[["guaranteed_compensation", "base_salary"]].reset_index()

df = (xg.merge(xp, on="player_id").merge(ga, on="player_id", how="left")
        .merge(players, on="player_id", how="left").merge(teams, on="team_id", how="left")
        .merge(sal, on="player_id", how="left"))
df = df[(df.minutes_played >= MIN_MINUTES) & (df.general_position != "GK")].copy()
df["group"] = df.general_position.map(GROUPS)
age_ref = pd.Timestamp(f"{SEASON}-07-01")
df["age"] = ((age_ref - pd.to_datetime(df.birth_date, errors="coerce")).dt.days / 365.25).round(1)

p96 = 96 / df.minutes_played
df["xg_96"] = df.xgoals * p96
df["shots_96"] = df.shots * p96
df["xa_96"] = df.xassists * p96
df["kp_96"] = df.key_passes * p96
df["passes_96"] = df.attempted_passes * p96
df["recv_96"] = df.n_Receiving * p96
df["drib_96"] = df.n_Dribbling * p96
df["int_96"] = df.n_Interrupting * p96
df["foul_96"] = df.n_Fouling * p96
for a in ACTIONS:
    df[f"ga_{a}_96"] = df[f"ga_{a}"] * p96
df["gplus_96"] = df[[f"ga_{a}_96" for a in ACTIONS]].sum(axis=1)
df["gplus_total"] = df[[f"ga_{a}" for a in ACTIONS]].sum(axis=1)
# Shrink g+/96 toward position-group mean: low-minute rates are noisy (≈ 900 min prior weight)
K_SHRINK = 900
gmean = df.groupby("group").gplus_96.transform("mean")
df["gplus_96_adj"] = (df.minutes_played * df.gplus_96 + K_SHRINK * gmean) / (df.minutes_played + K_SHRINK)

# STYLE = what a player does (drives roles). QUALITY = how well (drives percentiles).
STYLE = ["xg_96", "shots_96", "xa_96", "kp_96", "passes_96", "recv_96", "drib_96", "int_96",
         "avg_distance_yds", "avg_vertical_distance_yds", "share_team_touches"]
QUALITY = [f"ga_{a}_96" for a in ACTIONS] + ["passes_completed_over_expected_p100", "gplus_96"]
LABELS = {"xg_96": "xG", "shots_96": "Shots", "xa_96": "xA", "kp_96": "Key passes", "passes_96": "Passes",
          "recv_96": "Receptions", "drib_96": "Dribble actions", "int_96": "Interruptions",
          "avg_distance_yds": "Pass length", "avg_vertical_distance_yds": "Pass verticality",
          "share_team_touches": "Touch share", "passes_completed_over_expected_p100": "Pass accuracy vs xPass",
          "gplus_96": "g+ total", **{f"ga_{a}_96": f"g+ {a}" for a in ACTIONS}}

# ---------- roles (GMM on style z-scores within group) ----------
df["role_id"] = -1
role_desc = {}
for g, sub in df.groupby("group"):
    Z = StandardScaler().fit_transform(sub[STYLE].fillna(0))
    gm = GaussianMixture(ROLE_K[g], covariance_type="diag", n_init=20, random_state=7).fit(Z)
    lab = gm.predict(Z)
    df.loc[sub.index, "role_id"] = lab
    zdf = pd.DataFrame(Z, columns=STYLE, index=sub.index)
    for k in range(ROLE_K[g]):
        m = zdf[lab == k].mean().sort_values()
        role_desc[(g, k)] = {"n": int((lab == k).sum()),
                             "high": [LABELS[c] for c in m.index[::-1][:3]],
                             "low": [LABELS[c] for c in m.index[:2]],
                             "members": sub[lab == k].sort_values("minutes_played", ascending=False).player_name.head(6).tolist()}

# ---------- percentiles within group ----------
for c in STYLE + QUALITY:
    df[f"pct_{c}"] = df.groupby("group")[c].rank(pct=True).mul(100).round()

# ---------- comparables: cosine on style+quality z within group ----------
FEAT = STYLE + QUALITY[:-1]
comps = {}
for g, sub in df.groupby("group"):
    Z = StandardScaler().fit_transform(sub[FEAT].fillna(0))
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T
    ids = sub.player_id.values
    for i, pid in enumerate(ids):
        order = np.argsort(-S[i])[1:6]
        comps[pid] = [(ids[j], round(float(S[i, j]), 3)) for j in order]

# ---------- value: g+ vs what salary predicts, within group ----------
df["value_resid"] = np.nan
v = df.dropna(subset=["guaranteed_compensation"])
for g, sub in v.groupby("group"):
    X = np.log10(sub[["guaranteed_compensation"]].values)
    y = sub.gplus_96_adj.values
    h = HuberRegressor().fit(X, y)
    df.loc[sub.index, "gplus_expected_96"] = h.predict(X)
    df.loc[sub.index, "value_resid"] = y - h.predict(X)
df["value_season"] = df.value_resid * df.minutes_played / 96  # extra g+ over a season vs salary-expected

# Human role names: matched to each cluster's style signature (checked by hand for 2026)
ROLE_NAMES = {  # key order = matching priority
    "Centre-backs": {"Key passes": "Wide CB / hybrid", "Receptions": "Ball-playing CB", "Pass verticality": "Direct stopper"},
    "Fullbacks": {"xG": "Attacking wing-back", "Interruptions": "Defensive fullback", "Passes": "Possession fullback"},
    "Midfield": {"Touch share": "Deep playmaker hub", "xG": "Attacking 10", "Passes": "Circulator", "Interruptions": "Ball-winner"},
    "Wingers": {"Touch share": "Creator winger", "Shots": "Inside forward", "Interruptions": "Pressing wide forward"},
    "Strikers": {"xG": "Penalty-box striker", "Pass length": "Pressing target forward", "Passes": "Creative / false 9"},
}
for (g, k), d in role_desc.items():
    d["name"] = next((v for key, v in ROLE_NAMES[g].items() if key in d["high"]), f"{g} {k}")
df["role"] = [role_desc[(g, k)]["name"] for g, k in zip(df.group, df.role_id)]

df.to_csv(f"out/players_{SEASON}.csv", index=False)
json.dump({f"{g}|{k}": d for (g, k), d in role_desc.items()}, open(f"out/roles_{SEASON}.json", "w"), indent=1)
json.dump(comps, open(f"out/comps_{SEASON}.json", "w"))
print(len(df), "players")
for (g, k), d in sorted(role_desc.items()):
    print(f"{g:13s} {k} n={d['n']:3d} high={d['high']} low={d['low']}\n      {d['members']}")
