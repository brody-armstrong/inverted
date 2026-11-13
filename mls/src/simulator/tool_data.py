"""Data for the Roster Investment Simulator (out/tool/data.json).

Rosters = end-of-2026 MLSPA list (latest release), projected to 2027 with the production model (market-based ratings).
Team-strength uncertainty is calibrated so that simulated points spread matches the backtest's actual error (2016-2026):
extra team SD = sqrt(RMSE^2 - mean match-luck variance), converted to strength units.
"""
import json, pathlib
import numpy as np, pandas as pd

P = pd.read_pickle("out/proj_players_2027.pkl")
T = pd.read_pickle("out/proj_teams_2027.pkl")
M = json.load(open("out/proj_model.json"))
BT = pd.read_pickle("out/proj_teams.pkl")
coef = np.array(M["coef"])

# calibration: points error in backtest vs MC luck spread
BT = BT.dropna(subset=["actual_pts"])
rmse = float(np.sqrt(np.mean((BT.proj_pts - BT.actual_pts) ** 2)))
luck_sd = float(np.mean((BT.pts_hi - BT.pts_lo) / 2.563))
extra_pts_sd = float(np.sqrt(max(rmse ** 2 - luck_sd ** 2, 0)))
slope = np.polyfit(BT.Q * BT.groupby("season").Q.transform(lambda v: 1) , BT.proj_pts, 1)[0]      # points per unit Q
extra_Q_sd = extra_pts_sd / slope

# rosters
P = P.copy()
P["proj"] = (np.c_[np.ones(len(P)), P.p1, P.p2, P.new, P.ls, P.ls * P.new, P.ag, P.ag ** 2] @ coef)
pos_map = {"GK": "GK", "D": "D", "M": "M", "F": "F", "D-M": "D", "M-F": "M", "F-M": "F", "M-D": "M"}
P["pos"] = P["pos"].map(lambda p: pos_map.get(str(p), "M") if isinstance(p, str) else "M")
P["age_y"] = P.age.round(0)
east = {"Atlanta United FC", "Charlotte FC", "Chicago Fire FC", "FC Cincinnati", "Columbus Crew", "D.C. United", "Inter Miami CF",
        "CF Montréal", "Nashville SC", "New England Revolution", "New York City FC", "Red Bull New York", "Orlando City SC",
        "Philadelphia Union", "Toronto FC"}
teams = []
for t in T.itertuples():
    pl = P[P.team_e == t.team].sort_values("proj", ascending=False)
    teams.append({"id": t.team, "name": t.name, "conf": "East" if t.name in east else "West",
                  "players": [{"n": r.name or "?", "pos": r.pos, "age": None if pd.isna(r.age_y) else int(r.age_y), "sal": int(r.sal),
                               "r": round(float(r.proj), 4), "new": int(r.new)} for r in pl.itertuples()]})
# schedule template (2026 regular season)
EM = pd.read_pickle("out/espn_matches_usa.1.pkl")
EM["season"] = EM.date.dt.tz_convert("America/New_York").dt.year
sch = EM[(EM.season == 2026) & EM.season_type.str.contains("regular", na=False) & ~EM.home.str.contains("All-Star") & ~EM.away.str.contains("All-Star")]
ids = {t["id"]: i for i, t in enumerate(teams)}
sched = [[ids[h], ids[a]] for h, a in zip(sch.home_id, sch.away_id) if h in ids and a in ids]
# new-signing curve: projected rating of a newly signed player at salary s and age a (no MLS history)
new_curve = {"b0": float(coef[0] + coef[3]), "bls": float(coef[4] + coef[5]), "bag": float(coef[6]), "bag2": float(coef[7])}
bt = json.load(open("out/proj_backtest.json")); fb = json.load(open("out/futures_backtest.json"))
data = {"teams": teams, "sched": sched, "curve": M["curve"], "h": M["h"], "c": M["c"], "total": M["total_goals"],
        "team_sd": extra_Q_sd, "new_curve": new_curve,
        "meta": {"matches": 6340, "seasons": "2013–2026", "backtest_spearman_model": round(bt["mean_spearman_model"], 2),
                 "backtest_spearman_naive": round(bt["mean_spearman_naive"], 2),
                 "futures": {k: {"model_vs_market": round(v["spearman_model_market"], 2), "market_vs_actual": round(v["market_vs_actual"], 2),
                                 "model_vs_actual": round(v["model_vs_actual"], 2)} for k, v in fb.items()},
                 "rmse_pts": round(rmse, 1)}}
pathlib.Path("out/tool").mkdir(exist_ok=True)
json.dump(data, open("out/tool/data.json", "w"), separators=(",", ":"))
print("teams", len(teams), "players", sum(len(t["players"]) for t in teams), "matches", len(sched), "| backtest RMSE", round(rmse, 1),
      "luck sd", round(luck_sd, 1), "extra pts sd", round(extra_pts_sd, 1), "-> team strength sd", round(extra_Q_sd, 3),
      "| size KB", pathlib.Path("out/tool/data.json").stat().st_size // 1024)
print("new signing projected rating at $0.5M/1.25M/2.5M/5M, age 27:",
      [round(new_curve["b0"] + new_curve["bls"] * np.log(s), 3) for s in (5e5, 1.25e6, 2.5e6, 5e6)])
