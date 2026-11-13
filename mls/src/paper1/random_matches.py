"""Match-level MLS table 2013-2026 with venue, travel, rest, altitude, weather, attendance; plus odds."""
import json, math, pathlib
import numpy as np, pandas as pd

RAW, EXT = pathlib.Path("data/raw"), pathlib.Path("data/ext")
load = lambda p: json.loads(pathlib.Path(p).read_text())
teams = {t["team_id"]: t for t in load(RAW / "teams.json")}
stadia = {s["stadium_id"]: s for s in load(RAW / "mls_stadia.json")}
elev = load(EXT / "stadium_elevation.json")

rows = []
for s in range(2013, 2027):
    xg = {g["game_id"]: g for g in load(RAW / f"mls_games_xg_{s}.json")}
    for g in load(RAW / f"mls_games_{s}.json"):
        if g.get("status") != "FullTime":
            continue
        x = xg.get(g["game_id"], {})
        rows.append({"game_id": g["game_id"], "season": s, "kickoff_utc": g["date_time_utc"][:19], "home": g["home_team_id"],
                     "away": g["away_team_id"], "hg": g["home_score"], "ag": g["away_score"], "stadium": g.get("stadium_id"),
                     "attendance": g.get("attendance"), "knockout": bool(g.get("knockout_game")), "matchday": g.get("matchday"),
                     "hxg": x.get("home_team_xgoals"), "axg": x.get("away_team_xgoals")})
M = pd.DataFrame(rows)
M["kickoff_utc"] = pd.to_datetime(M.kickoff_utc)
M = M.sort_values("kickoff_utc").reset_index(drop=True)

# each team's home stadium = its most used home venue that season
home_st = M[~M.knockout].groupby(["season", "home"]).stadium.agg(lambda x: x.value_counts().index[0])


def st_attr(sid, k):
    return stadia.get(sid, {}).get(k)


def hav(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return np.nan
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


M["neutral"] = [st != home_st.get((s, h)) for s, h, st in zip(M.season, M.home, M.stadium)]
M["lat"] = M.stadium.map(lambda s: st_attr(s, "latitude"))
M["lon"] = M.stadium.map(lambda s: st_attr(s, "longitude"))
M["turf"] = M.stadium.map(lambda s: st_attr(s, "turf"))
M["capacity"] = M.stadium.map(lambda s: st_attr(s, "capacity"))
M["elev"] = M.stadium.map(lambda s: elev.get(s))
away_home_st = [home_st.get((s, a)) for s, a in zip(M.season, M.away)]
M["away_home_lat"] = [st_attr(x, "latitude") for x in away_home_st]
M["away_home_lon"] = [st_attr(x, "longitude") for x in away_home_st]
M["away_home_elev"] = [elev.get(x) if x else np.nan for x in away_home_st]
M["away_home_turf"] = [st_attr(x, "turf") for x in away_home_st]
M["travel_km"] = [hav(a, b, c, d) for a, b, c, d in zip(M.away_home_lat, M.away_home_lon, M.lat, M.lon)]
# time zones crossed: longitude difference / 15 degrees, rounded
M["tz_crossed"] = ((M.lon - M.away_home_lon).abs() / 15).round()
M["elev_gain_m"] = M.elev - M.away_home_elev
M["turf_mismatch"] = (M.turf == True) & (M.away_home_turf == False)

# rest days for each side (any competitive MLS match incl. playoffs)
long = pd.concat([M[["game_id", "kickoff_utc", "home"]].rename(columns={"home": "team"}).assign(side="h"),
                  M[["game_id", "kickoff_utc", "away"]].rename(columns={"away": "team"}).assign(side="a")]).sort_values("kickoff_utc")
long["rest"] = long.groupby("team").kickoff_utc.diff().dt.total_seconds() / 86400
rest = long.pivot(index="game_id", columns="side", values="rest")
M = M.merge(rest.rename(columns={"h": "home_rest", "a": "away_rest"}), left_on="game_id", right_index=True, how="left")

# kickoff weather (hourly, nearest hour) and away team's typical home temperature that month
wx = {}
for f in (EXT / "weather").glob("*.json"):
    d = load(f)
    h = d.get("hourly", {})
    if h:
        wx[f.stem] = pd.DataFrame({"t": pd.to_datetime(h["time"]), "temp": h["temperature_2m"], "rh": h["relative_humidity_2m"],
                                   "precip": h["precipitation"]}).set_index("t")
temps, rhs, prs = [], [], []
for sid, ko in zip(M.stadium, M.kickoff_utc):
    w = wx.get(sid)
    if w is None:
        temps.append(np.nan); rhs.append(np.nan); prs.append(np.nan); continue
    k = ko.round("h")
    if k in w.index:
        r = w.loc[k]; temps.append(r.temp); rhs.append(r.rh); prs.append(r.precip)
    else:
        temps.append(np.nan); rhs.append(np.nan); prs.append(np.nan)
M["temp_c"], M["rh"], M["precip_mm"] = temps, rhs, prs
# away team's home climate: mean temperature at its stadium in that calendar month (all years)
clim = {}
for sid, w in wx.items():
    clim[sid] = w.temp.groupby(w.index.month).mean().to_dict()
M["away_home_month_temp"] = [clim.get(x, {}).get(k.month) if x else np.nan for x, k in zip(away_home_st, M.kickoff_utc)]
M["temp_shock"] = M.temp_c - M.away_home_month_temp
M["fill"] = M.attendance / M.capacity

# odds: Pinnacle closing (PSC*) else market average closing
O = pd.read_csv(EXT / "odds_USA.csv")
O = O[O.League == "MLS"].copy()
O["date"] = pd.to_datetime(O.Date, dayfirst=True)
name_map = {}
for tid, t in teams.items():
    name_map[t["team_name"]] = tid
alias = {"CF Montreal": "CF Montréal", "Montreal Impact": "CF Montréal", "DC United": "D.C. United", "Los Angeles Galaxy": "LA Galaxy",
         "Los Angeles FC": "Los Angeles FC", "New York Red Bulls": "New York Red Bulls", "Minnesota": "Minnesota United FC",
         "Sporting Kansas City": "Sporting Kansas City", "Inter Miami": "Inter Miami CF", "Nashville SC": "Nashville SC",
         "St. Louis City": "St. Louis City SC", "San Diego FC": "San Diego FC", "Austin FC": "Austin FC", "Charlotte": "Charlotte FC"}
import difflib
tnames = list(name_map)


def to_id(n):
    n2 = alias.get(n, n)
    if n2 in name_map:
        return name_map[n2]
    m = difflib.get_close_matches(n2, tnames, n=1, cutoff=0.5)
    return name_map[m[0]] if m else None


O["home"], O["away"] = O.Home.map(to_id), O.Away.map(to_id)
unmatched = sorted(set(O.Home[O.home.isna()]) | set(O.Away[O.away.isna()]))
for c in ["H", "D", "A"]:
    O[f"p{c}_raw"] = 1 / pd.to_numeric(O[f"PSC{c}"], errors="coerce").fillna(pd.to_numeric(O[f"AvgC{c}"], errors="coerce"))
s = O[["pH_raw", "pD_raw", "pA_raw"]].sum(axis=1)
for c in ["H", "D", "A"]:
    O[f"mkt_{c}"] = O[f"p{c}_raw"] / s          # remove bookmaker margin
O["odds_src"] = np.where(pd.to_numeric(O.PSCH, errors="coerce").notna(), "Pinnacle closing", "market average closing")
M["date"] = M.kickoff_utc.dt.tz_localize(None).dt.normalize() if M.kickoff_utc.dt.tz is not None else M.kickoff_utc.dt.normalize()
# odds dates are UK dates; allow +-1 day
O1 = O[["date", "home", "away", "mkt_H", "mkt_D", "mkt_A", "odds_src"]].dropna(subset=["home", "away"])
mm = []
for shift in (0, -1, 1):
    t = O1.assign(date=O1.date + pd.Timedelta(days=shift))
    mm.append(M[["game_id", "date", "home", "away"]].merge(t, on=["date", "home", "away"]))
mm = pd.concat(mm).drop_duplicates("game_id")
M = M.merge(mm[["game_id", "mkt_H", "mkt_D", "mkt_A", "odds_src"]], on="game_id", how="left")
M.to_pickle("out/mls_matches.pkl")
print(len(M), "matches |", M.knockout.sum(), "playoff |", M.neutral.sum(), "neutral-ish |",
      "with odds:", M.mkt_H.notna().sum(), "| weather:", M.temp_c.notna().sum(), "| xG:", M.hxg.notna().sum(),
      "| attendance:", M.attendance.notna().sum())
print("unmatched odds team names:", unmatched)
print(M[["travel_km", "tz_crossed", "elev_gain_m", "home_rest", "away_rest", "temp_c", "temp_shock", "fill"]].describe().round(2).to_string())
print(M.groupby("season").agg(n=("game_id", "size"), odds=("mkt_H", lambda x: x.notna().mean()), att0=("attendance", lambda x: (x.fillna(-1) == 0).sum())).round(2).T.to_string())
