"""Parity study: luck share, year-to-year stability, roster continuity. MLS (ASA) vs other leagues (Transfermarkt)."""
import json, pathlib
import numpy as np, pandas as pd

RAW = pathlib.Path("data/raw"); TM = pathlib.Path("data/tm")
load = lambda n: json.loads((RAW / f"{n}.json").read_text())
NAMES = {"MLS": "MLS", "GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1",
         "NL1": "Eredivisie", "PO1": "Liga Portugal", "BE1": "Belgian Pro League", "TR1": "Süper Lig", "RU1": "Russian Premier Liga",
         "SC1": "Scottish Premiership", "GR1": "Greek Super League", "DK1": "Danish Superliga", "UKR1": "Ukrainian Premier Liga",
         "BRA1": "Brasileirão", "PL1": "Ekstraklasa", "SA1": "Saudi Pro League", "C1": "Swiss Super League", "RO1": "Romanian Superliga",
         "SER1": "Serbian SuperLiga", "TS1": "Czech First League"}


def team_table(g, season_col="season"):
    """g: home, away, hg, ag, season, league -> per team-season points, games, plus per-match point outcomes."""
    h = pd.DataFrame({"league": g.league, "season": g[season_col], "team": g.home, "gf": g.hg, "ga": g.ag, "home": 1})
    a = pd.DataFrame({"league": g.league, "season": g[season_col], "team": g.away, "gf": g.ag, "ga": g.hg, "home": 0})
    m = pd.concat([h, a])
    m["pts"] = np.select([m.gf > m.ga, m.gf == m.ga], [3, 1], 0)
    return m


# --- MLS matches (ASA, regular season) ---
rows = []
for s in range(2013, 2027):
    for x in load(f"mls_games_{s}"):
        if x.get("knockout_game") or x.get("status") != "FullTime":
            continue
        rows.append({"league": "MLS", "season": s, "home": x["home_team_id"], "away": x["away_team_id"],
                     "hg": x["home_score"], "ag": x["away_score"]})
G = [pd.DataFrame(rows)]
# --- other leagues (Transfermarkt) ---
tg = pd.read_csv(TM / "games.csv.gz", usecols=["competition_id", "season", "home_club_id", "away_club_id",
                                               "home_club_goals", "away_club_goals", "competition_type"])
tg = tg[(tg.competition_type == "domestic_league") & tg.competition_id.isin(NAMES)].dropna()
tg = tg.rename(columns={"competition_id": "league", "home_club_id": "home", "away_club_id": "away",
                        "home_club_goals": "hg", "away_club_goals": "ag"})
G.append(tg[["league", "season", "home", "away", "hg", "ag"]])
G = pd.concat(G, ignore_index=True)
M = team_table(G)

# complete-ish seasons only: every team played >= 85% of the season's modal games and >= 20 games
TS = M.groupby(["league", "season", "team"]).agg(pts=("pts", "sum"), n=("pts", "size"), gd=("gf", "sum"), ga=("ga", "sum")).reset_index()
TS["gd"] = TS.gd - TS.ga
TS["ppg"] = TS.pts / TS.n
mode_n = TS.groupby(["league", "season"]).n.transform("max")
TS = TS[(TS.n >= 0.85 * mode_n) & (mode_n >= 20)]
TS = TS[~((TS.league == "MLS") & (TS.season == 2020))]  # pandemic season: bubble tournament + short schedule

# --- A. luck share per league-season ---
ls = []
for (lg, s), t in TS.groupby(["league", "season"]):
    if len(t) < 10:
        continue
    mm = M[(M.league == lg) & (M.season == s) & M.team.isin(t.team)]
    var_match = mm.pts.var()                      # variance of one team-game's points (league's own W/D/L mix)
    var_luck = var_match / t.n.mean()             # binomial-style noise in a season's ppg
    var_obs = t.ppg.var()
    draws = (mm.pts == 1).mean()
    ls.append({"league": lg, "season": int(s), "teams": len(t), "games": float(t.n.mean()), "var_obs": var_obs,
               "var_luck": var_luck, "draw_rate": draws, "sd_obs": np.sqrt(var_obs),
               "sd_talent": np.sqrt(max(var_obs - var_luck, 0)), "luck_share": min(var_luck / var_obs, 1)})
LS = pd.DataFrame(ls)
lg_sum = (LS.groupby("league").agg(seasons=("season", "size"), first=("season", "min"), last=("season", "max"),
                                   luck_share=("luck_share", "mean"), sd_talent=("sd_talent", "mean"),
                                   sd_obs=("sd_obs", "mean"), games=("games", "mean"), draw_rate=("draw_rate", "mean"))
          .reset_index())
# games needed before talent explains half the spread: var_match / var_talent
var_match_lg = LS.assign(vm=LS.var_luck * LS.games).groupby("league").vm.mean()
lg_sum["games_to_half"] = lg_sum.league.map(var_match_lg) / lg_sum.sd_talent ** 2

# --- B. year-to-year stability ---
st = []
for lg, t in TS.groupby("league"):
    a = t[["season", "team", "ppg"]]
    b = a.assign(season=a.season - 1).rename(columns={"ppg": "ppg_next"})
    j = a.merge(b, on=["season", "team"])
    if len(j) >= 30:
        st.append({"league": lg, "pairs": len(j), "r": float(np.corrcoef(j.ppg, j.ppg_next)[0, 1])})
ST = pd.DataFrame(st)
lg_sum = lg_sum.merge(ST, on="league", how="left")

# --- C. roster continuity ---
# MLS: ASA player minutes by club (traded players credited to last club)
pm = []
for s in range(2013, 2027):
    for x in load(f"mls_xgoals_{s}"):
        tid = x["team_id"][-1] if isinstance(x["team_id"], list) else x["team_id"]
        pm.append({"league": "MLS", "season": s, "team": tid, "player": x["player_id"], "min": x["minutes_played"]})
PM = pd.DataFrame(pm)
# others: Transfermarkt appearances in domestic league games, season from games table
ap = pd.read_csv(TM / "appearances.csv.gz", usecols=["game_id", "player_id", "player_club_id", "competition_id", "minutes_played"])
ap = ap[ap.competition_id.isin(NAMES)]
gs = pd.read_csv(TM / "games.csv.gz", usecols=["game_id", "season"])
ap = ap.merge(gs, on="game_id")
PM2 = (ap.groupby(["competition_id", "season", "player_club_id", "player_id"]).minutes_played.sum().reset_index()
         .rename(columns={"competition_id": "league", "player_club_id": "team", "player_id": "player", "minutes_played": "min"}))
PM = pd.concat([PM, PM2], ignore_index=True)


def continuity(pm):
    prev = pm[["league", "season", "team", "player"]].assign(season=pm.season + 1, back=1)
    j = pm.merge(prev, on=["league", "season", "team", "player"], how="left")
    j["back"] = j.back.fillna(0)
    c = j.groupby(["league", "season", "team"]).apply(lambda x: (x["min"] * x.back).sum() / x["min"].sum(), include_groups=False)
    return c.rename("continuity").reset_index()


CO = continuity(PM)
first_season = PM.groupby("league").season.min()
CO = CO[CO.season > CO.league.map(first_season)]
CO = CO.merge(TS[["league", "season", "team"]], on=["league", "season", "team"])  # only clubs in that season's table
co_lg = CO.groupby("league").continuity.agg(["mean", "size"]).rename(columns={"mean": "continuity", "size": "co_n"}).reset_index()
lg_sum = lg_sum.merge(co_lg, on="league", how="left")
mls_co_by_season = CO[CO.league == "MLS"].groupby("season").continuity.mean()

# MLS: does continuity make results stickier?
mt = TS[TS.league == "MLS"][["season", "team", "ppg"]]
mt = mt.merge(mt.assign(season=mt.season + 1).rename(columns={"ppg": "ppg_prev"}), on=["season", "team"])
mt = mt.merge(CO[CO.league == "MLS"][["season", "team", "continuity"]], on=["season", "team"])
mt["co_tier"] = pd.qcut(mt.continuity, 3, labels=["Low", "Mid", "High"])
by_tier = {str(k): {"n": len(v), "r": float(np.corrcoef(v.ppg_prev, v.ppg)[0, 1]), "continuity": float(v.continuity.mean()),
                    "abs_change": float((v.ppg - v.ppg_prev).abs().mean())} for k, v in mt.groupby("co_tier", observed=True)}
X = np.column_stack([np.ones(len(mt)), mt.ppg_prev - mt.ppg_prev.mean(), mt.continuity - mt.continuity.mean(),
                     (mt.ppg_prev - mt.ppg_prev.mean()) * (mt.continuity - mt.continuity.mean())])
coef = np.linalg.lstsq(X, mt.ppg, rcond=None)[0]

lg_sum["name"] = lg_sum.league.map(NAMES)
lg_sum.to_csv("out/parity_leagues.csv", index=False)
LS.to_csv("out/parity_league_seasons.csv", index=False)
mt.to_csv("out/mls_continuity.csv", index=False)
json.dump({"by_tier": by_tier, "interaction_coef": coef.tolist(), "n": len(mt),
           "mls_continuity_by_season": mls_co_by_season.round(3).to_dict()}, open("out/parity_continuity.json", "w"), indent=1)

pd.set_option("display.width", 200)
print(lg_sum.sort_values("luck_share", ascending=False)[["name", "seasons", "first", "last", "games", "draw_rate", "sd_obs", "sd_talent",
      "luck_share", "games_to_half", "pairs", "r", "continuity", "co_n"]].round(3).to_string(index=False))
print("\nMLS luck share by season:", LS[LS.league == "MLS"].set_index("season").luck_share.round(2).to_dict())
print("MLS continuity by season:", mls_co_by_season.round(2).to_dict())
print("persistence by continuity tier:", json.dumps(by_tier, indent=0))
print("interaction model coef [int, prev, cont, prev*cont]:", np.round(coef, 3))
