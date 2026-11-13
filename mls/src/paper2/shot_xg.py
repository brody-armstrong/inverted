"""Text-based xG model for ESPN shot commentary, and ESPN<->ASA match linkage.

Model: logistic regression of goal on pre-shot features (location x body part, assist type x body part, situation),
cross-validated by season; penalties use the observed conversion rate. Validation: per team-match xG vs ASA's xG.
Writes out/espn_shots_xg_usa.1.pkl (shots with xg), out/espn_asa_link.csv, out/xg_validation.json.
"""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score
from scipy.stats import pearsonr

S = pd.read_pickle("out/espn_shots_usa.1.pkl"); M = pd.read_pickle("out/espn_matches_usa.1.pkl")
M = M[~M.season_type.str.contains("all-star", na=False)]
M["season"] = M.date.dt.tz_convert("America/New_York").dt.year
S = S.merge(M[["event", "season"]], on="event")
sh = S[S.kind == "shot"].copy()
sh["lb"] = sh["loc"] + "|" + sh.body; sh["ab"] = sh.assist + "|" + sh.body; sh["sb"] = sh.situ + "|" + sh.body
FE = ["loc", "body", "assist", "situ", "lb", "ab", "sb"]
enc = OneHotEncoder(handle_unknown="ignore", min_frequency=30)

# season-wise cross-validation: each season predicted by a model fit on the others
sh["xg"] = np.nan
for s in sorted(sh.season.unique()):
    tr, te = sh.season != s, sh.season == s
    X = enc.fit_transform(sh.loc[tr, FE]); m = LogisticRegression(C=1.0, max_iter=2000).fit(X, sh.loc[tr, "goal"])
    sh.loc[te, "xg"] = m.predict_proba(enc.transform(sh.loc[te, FE]))[:, 1]
pen = S[S.kind == "penalty"]; pen_rate = pen.goal.mean()
S.loc[S.kind == "penalty", "xg"] = pen_rate
S.loc[sh.index, "xg"] = sh.xg
S.loc[S.kind == "own_goal", "xg"] = 0.0
base = np.full(len(sh), sh.goal.mean())
V = {"shots": int(len(sh)), "goal_rate": float(sh.goal.mean()), "penalty_rate": float(pen_rate),
     "cv_log_loss": float(log_loss(sh.goal, sh.xg)), "base_log_loss": float(log_loss(sh.goal, base)), "cv_auc": float(roc_auc_score(sh.goal, sh.xg)),
     "xg_sum_over_goals": float(sh.xg.sum() / sh.goal.sum())}

# ---------- link ESPN events to ASA games ----------
A = pd.read_pickle("out/mls_matches.pkl")[["game_id", "season", "kickoff_utc", "home", "away", "hg", "ag", "hxg", "axg", "knockout"]]
A["d"] = pd.to_datetime(A.kickoff_utc, utc=True).dt.tz_convert("America/New_York").dt.date
M["d"] = M.date.dt.tz_convert("America/New_York").dt.date
teams = {t["team_id"]: t["team_name"] for t in json.load(open("data/raw/teams.json"))}
A["home_n"] = A.home.map(teams); A["away_n"] = A.away.map(teams)
# pass 1: same day + same score -> learn the ESPN->ASA name map from unambiguous pairs
j = M.merge(A, on=["d", "hg", "ag"], suffixes=("", "_a"))
amb = j.groupby("event").game_id.transform("size")
pairs = pd.concat([j[amb == 1][["home", "home_n"]].set_axis(["e", "a"], axis=1), j[amb == 1][["away", "away_n"]].set_axis(["e", "a"], axis=1)])
NM = pairs.groupby("e").a.agg(lambda v: v.value_counts().index[0]).to_dict()
M["home_m"] = M.home.map(NM); M["away_m"] = M.away.map(NM)
L = M.merge(A.rename(columns={"d": "d_a", "season": "season_a", "hg": "hg_a", "ag": "ag_a", "home": "home_asa", "away": "away_asa"}), left_on=["home_m", "away_m"], right_on=["home_n", "away_n"])
L["gap"] = (pd.to_datetime(L.d) - pd.to_datetime(L.d_a)).abs().dt.days
L = L[L.gap <= 1].sort_values("gap").drop_duplicates("event").drop_duplicates("game_id")
link = L[["event", "game_id", "season_a"]].rename(columns={"season_a": "season"})
link.to_csv("out/espn_asa_link.csv", index=False)
V["linked"] = int(len(link)); V["asa_games"] = int(len(A)); V["espn_events"] = int(len(M))

# ---------- validation vs ASA team-match xG ----------
tm = S[S.team_id.notna()].groupby(["event", "is_home"]).xg.sum().unstack(fill_value=0).rename(columns={True: "e_h", False: "e_a"})
v = link.merge(tm, left_on="event", right_index=True).merge(A[["game_id", "hxg", "axg"]], on="game_id").dropna()
r_team = pearsonr(np.r_[v.e_h, v.e_a], np.r_[v.hxg, v.axg]).statistic
r_diff = pearsonr(v.e_h - v.e_a, v.hxg - v.axg).statistic
scale = float(np.r_[v.hxg, v.axg].sum() / np.r_[v.e_h, v.e_a].sum())
V.update({"validated_matches": int(len(v)), "r_team_match_xg": float(r_team), "r_xg_difference": float(r_diff), "asa_over_ours": scale})
by = v.groupby("season").apply(
    lambda x: pd.Series({"r": pearsonr(np.r_[x.e_h, x.e_a], np.r_[x.hxg, x.axg]).statistic, "n": len(x)}), include_groups=False)
V["r_by_season"] = {int(k): round(float(r), 3) for k, r in by.r.items()}
sh_out = S.copy(); sh_out.to_pickle("out/espn_shots_xg_usa.1.pkl")
json.dump(V, open("out/xg_validation.json", "w"), indent=1)
print(json.dumps(V, indent=1))
print("unmapped ESPN names:", sorted(set(M.home) - set(NM))[:10])
