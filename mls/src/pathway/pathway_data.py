"""Assemble one long table of player-seasons across MLS, USLC, USL1 and MLS NEXT Pro."""
import json, pathlib
import pandas as pd

RAW = pathlib.Path("data/raw")
ACTIONS = ["Dribbling", "Fouling", "Interrupting", "Passing", "Receiving", "Shooting"]
LEAGUES = {"mls": range(2017, 2027), "uslc": range(2017, 2027), "usl1": range(2019, 2027), "mlsnp": range(2022, 2027)}


def load(n):
    return json.loads((RAW / f"{n}.json").read_text())


def season(lg, s):
    xg = pd.DataFrame(load(f"{lg}_xgoals_{s}"))
    if xg.empty:
        return None
    xg["team_id"] = xg.team_id.apply(lambda t: t[-1] if isinstance(t, list) else t)
    xp = pd.DataFrame(load(f"{lg}_xpass_{s}"))[["player_id", "attempted_passes", "passes_completed_over_expected_p100",
                                                 "avg_distance_yds", "avg_vertical_distance_yds", "share_team_touches"]]
    rows = []
    for p in load(f"{lg}_ga_{s}"):
        r = {"player_id": p["player_id"]}
        for a in p["data"]:
            r[f"ga_{a['action_type']}"] = a["goals_added_above_avg"]
            r[f"n_{a['action_type']}"] = a["count_actions"]
        rows.append(r)
    df = xg.merge(xp, on="player_id", how="left").merge(pd.DataFrame(rows), on="player_id", how="left")
    df["league"], df["season"] = lg, s
    return df


frames = [f for lg, ss in LEAGUES.items() for s in ss if (f := season(lg, s)) is not None]
d = pd.concat(frames, ignore_index=True)
d = d[d.general_position != "GK"].copy()

names = {}
for lg in LEAGUES:
    for p in load(f"{lg}_players"):
        names.setdefault(p["player_id"], (p["player_name"], p.get("birth_date")))
d["player_name"] = d.player_id.map(lambda i: names.get(i, (None, None))[0])
d["birth_date"] = d.player_id.map(lambda i: names.get(i, (None, None))[1])
d["age"] = (pd.to_datetime(d.season.astype(str) + "-07-01") - pd.to_datetime(d.birth_date, errors="coerce")).dt.days / 365.25

p96 = 96 / d.minutes_played
d["xg_96"] = d.xgoals * p96
d["shots_96"] = d.shots * p96
d["xa_96"] = d.xassists * p96
d["kp_96"] = d.key_passes * p96
d["passes_96"] = d.attempted_passes * p96
d["recv_96"] = d.n_Receiving * p96
d["drib_96"] = d.n_Dribbling * p96
d["int_96"] = d.n_Interrupting * p96
for a in ACTIONS:
    d[f"ga_{a}_96"] = d[f"ga_{a}"] * p96
d["gplus_96"] = d[[f"ga_{a}_96" for a in ACTIONS]].sum(axis=1, min_count=1)
d.to_pickle("out/all_seasons.pkl")

print(d.groupby(["league", "season"]).size().unstack(0).fillna(0).astype(int))
ids = {lg: set(d[d.league == lg].player_id) for lg in LEAGUES}
for lg in ["uslc", "usl1", "mlsnp"]:
    print(lg, "players also seen in MLS:", len(ids[lg] & ids["mls"]))
