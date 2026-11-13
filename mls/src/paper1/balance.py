"""Task 15: standard competitive-balance metrics, MLS vs 11 European leagues vs NFL/NBA/NHL/MLB.

Per league-season:
  luck share   = (per-game outcome variance / games) / observed variance of points-per-game across teams
                 (the share of the table's spread a league of equal teams would produce by chance)
  Noll-Scully  = observed SD / chance SD = 1 / sqrt(luck share)
  rank corr    = Spearman correlation of points-per-game between consecutive seasons (same clubs)
  best wins    = did the team with the best regular-season record win the championship?
Intervals: bootstrap over seasons (4,000 draws).
Writes out/balance_seasons.csv, out/balance_summary.csv, out/balance_titles.csv.
"""
import json, glob, pathlib, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

EXT = pathlib.Path("data/ext"); RAW = pathlib.Path("data/raw")
rng = np.random.default_rng(7)
EURO = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1", "N1": "Eredivisie",
        "P1": "Liga Portugal", "B1": "Belgian Pro League", "T1": "Süper Lig", "SC0": "Scottish Premiership", "G1": "Greek Super League"}

TEAM = []   # league, season, team, g, pts (in that league's points units), var_match contributions


def soccer_tables(g, league):
    """g: season, home, away, hg, ag -> team rows + per-team-game points."""
    out = []
    for s, x in g.groupby("season"):
        hp = np.select([x.hg > x.ag, x.hg == x.ag], [3, 1], 0)
        ap = np.select([x.ag > x.hg, x.hg == x.ag], [3, 1], 0)
        pts = np.r_[hp, ap]
        gd = np.r_[x.hg - x.ag, x.ag - x.hg]
        t = (pd.DataFrame({"team": np.r_[x.home, x.away], "pts": pts, "win": pts == 3, "gd": gd})
             .groupby("team").agg(sum=("pts", "sum"), size=("pts", "size"), wins=("win", "sum"), gd=("gd", "sum")))
        out.append(pd.DataFrame({"league": league, "season": s, "team": t.index, "g": t["size"].values,
                                 "pts": t["sum"].values, "wins": t.wins.values, "gdiff": t.gd.values,
                                 "var_match": pts.var(ddof=0), "family": "soccer"}))
    return pd.concat(out)


# --- MLS (regular season; 2020 = MLS is Back + shortened season, kept but flagged) ---
m = pd.read_pickle("out/mls_matches.pkl")
m = m[~m.knockout.astype(bool)]
TEAM.append(soccer_tables(m[["season", "home", "away", "hg", "ag"]], "MLS"))

# --- Europe (football-data.co.uk, 2012-13 .. 2025-26; season labelled by the year it ends) ---
for code, name in EURO.items():
    fs = sorted(glob.glob(str(EXT / f"odds_{code}_*.csv")))
    g = []
    for f in fs:
        yy = re.search(r"_(\d{4})\.csv", f).group(1)
        d = pd.read_csv(f, encoding="latin-1", usecols=lambda c: c in {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}).dropna()
        d["season"] = 2000 + int(yy[2:])
        g.append(d.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"}))
    TEAM.append(soccer_tables(pd.concat(g), name))

# --- NFL (nflverse; regular season; ties = half a win) ---
n = pd.read_csv(EXT / "nfl_games.csv", usecols=["season", "game_type", "home_team", "away_team", "home_score", "away_score"])
n = n[(n.season.between(2013, 2025)) & n.home_score.notna()]
FR = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
for c in ("home_team", "away_team"):
    n[c] = n[c].replace(FR)
reg = n[n.game_type == "REG"]
rows = []
for s, x in reg.groupby("season"):
    hw = np.where(x.home_score > x.away_score, 1, np.where(x.home_score == x.away_score, .5, 0))
    w = np.r_[hw, 1 - hw]
    t = pd.DataFrame({"team": np.r_[x.home_team, x.away_team], "w": w}).groupby("team").w.agg(["sum", "size"])
    rows.append(pd.DataFrame({"league": "NFL", "season": s, "team": t.index, "g": t["size"].values, "pts": t["sum"].values,
                              "var_match": w.var(ddof=0), "family": "NA"}))
TEAM.append(pd.concat(rows))
sb = n[n.game_type == "SB"].assign(champ=lambda d: np.where(d.home_score > d.away_score, d.home_team, d.away_team))
CHAMP = {("NFL", int(r.season)): r.champ for r in sb.itertuples()}

# --- NBA (ESPN standings; season labelled by end year) ---
rows = []
for f in sorted(glob.glob(str(EXT / "nba_standings_*.json"))):
    y = int(re.search(r"_(\d{4})", f).group(1)); d = json.load(open(f))
    for conf in d["children"]:
        for e in conf["standings"]["entries"]:
            st = {s["name"]: s.get("value") for s in e["stats"]}
            rows.append({"league": "NBA", "season": y, "team": e["team"]["abbreviation"], "g": st["wins"] + st["losses"],
                         "pts": st["wins"], "family": "NA"})
nba = pd.DataFrame(rows); nba["var_match"] = 0.25
TEAM.append(nba)
NBA_CH = {2014: "SA", 2015: "GS", 2016: "CLE", 2017: "GS", 2018: "GS", 2019: "TOR", 2020: "LAL", 2021: "MIL",
          2022: "GS", 2023: "DEN", 2024: "BOS", 2025: "OKC", 2026: "NY"}   # 2026: Knicks beat Spurs 4-1 (web, June 2026)
CHAMP.update({("NBA", y): c for y, c in NBA_CH.items()})

# --- NHL (points: W=2, OTL=1; points-per-game is the "pts" analogue) ---
rows = []
for f in sorted(glob.glob(str(EXT / "nhl_standings_*.json"))):
    y = int(re.search(r"_(\d{8})", f).group(1)[4:]); d = json.load(open(f))["standings"]
    for t in d:
        ab = t["teamAbbrev"]["default"]; ab = {"ARI": "UTA", "PHX": "UTA"}.get(ab, ab)
        rows.append({"league": "NHL", "season": y, "team": ab, "g": t["gamesPlayed"], "pts": t["points"],
                     "w": t["wins"], "otl": t["otLosses"], "family": "NA"})
nhl = pd.DataFrame(rows)
for y, x in nhl.groupby("season"):   # outcome mix: share of team-games that are W (2), OTL (1), L (0)
    G = x.g.sum(); pw, po = x.w.sum() / G, x.otl.sum() / G
    mu = 2 * pw + po
    nhl.loc[x.index, "var_match"] = 4 * pw + po - mu ** 2
TEAM.append(nhl.drop(columns=["w", "otl"]))
for f in sorted(glob.glob(str(EXT / "nhl_bracket_*.json"))):
    y = int(re.search(r"_(\d{4})", f).group(1)); fin = [s for s in json.load(open(f))["series"] if s["playoffRound"] == 4][0]
    win = fin["topSeedTeam"] if fin["winningTeamId"] == fin["topSeedTeam"]["id"] else fin["bottomSeedTeam"]
    CHAMP[("NHL", y)] = {"ARI": "UTA"}.get(win["abbrev"], win["abbrev"])

# --- MLB ---
rows = []
for f in sorted(glob.glob(str(EXT / "mlb_standings_*.json"))):
    y = int(re.search(r"_(\d{4})", f).group(1))
    for r in json.load(open(f))["records"]:
        for t in r["teamRecords"]:
            rows.append({"league": "MLB", "season": y, "team": t["team"]["id"], "g": t["wins"] + t["losses"], "pts": t["wins"],
                         "family": "NA"})
mlb = pd.DataFrame(rows); mlb["var_match"] = 0.25
TEAM.append(mlb)
for f in sorted(glob.glob(str(EXT / "mlb_ws_*.json"))):
    y = int(re.search(r"_(\d{4})", f).group(1)); w = {}
    for dt in json.load(open(f))["dates"]:
        for gm in dt["games"]:
            for side in ("home", "away"):
                if gm["teams"][side].get("isWinner"):
                    nm = gm["teams"][side]["team"]["id"]; w[nm] = w.get(nm, 0) + 1
    CHAMP[("MLB", y)] = max(w, key=w.get)

# --- MLS Cup (ASA has no shootout results, so winners are listed; checked against final scores) ---
teams = {t["team_id"]: t["team_name"] for t in json.load(open(RAW / "teams.json"))}
CUP = {2013: "Sporting Kansas City", 2014: "LA Galaxy", 2015: "Portland Timbers", 2016: "Seattle Sounders FC",
       2017: "Toronto FC", 2018: "Atlanta United FC", 2019: "Seattle Sounders FC", 2020: "Columbus Crew",
       2021: "New York City FC", 2022: "Los Angeles FC", 2023: "Columbus Crew", 2024: "LA Galaxy", 2025: "Inter Miami CF"}
name2id = {}
for k, v in teams.items():
    name2id[v] = k
for y, nm in CUP.items():
    hit = [k for k, v in teams.items() if v.lower().replace("fc", "").strip() in nm.lower() or nm.lower() in v.lower()]
    CHAMP[("MLS", y)] = hit
T = pd.concat(TEAM, ignore_index=True)
T["ppg"] = T.pts / T.g
mode_g = T.groupby(["league", "season"]).g.transform("max")
T = T[T.g >= 0.85 * mode_g]

# ---------- per-season metrics ----------
S = (T.groupby(["league", "season"]).agg(teams=("team", "size"), g=("g", "mean"), var_obs=("ppg", lambda v: v.var(ddof=1)),
                                         var_match=("var_match", "first"), family=("family", "first")).reset_index())
S["var_luck"] = S.var_match / S.g
S["luck_share"] = (S.var_luck / S.var_obs).clip(upper=1)
S["noll_scully"] = np.sqrt(S.var_obs / S.var_luck)
S["short"] = ((S.league == "MLS") & (S.season == 2020)) | ((S.league == "MLB") & (S.season == 2020)) | \
             ((S.league == "NHL") & S.season.isin([2020, 2021])) | ((S.league == "NBA") & S.season.isin([2020, 2021]))


# best regular-season record wins the championship (ties at the top share credit)
def best_wins(lg, y):
    c = CHAMP.get((lg, y))
    if c is None:
        return np.nan
    x = T[(T.league == lg) & (T.season == y)]
    if lg == "MLS":   # Supporters' Shield: points-per-game (2020 decided on ppg), then wins, then goal difference
        top = x.sort_values(["ppg", "wins", "gdiff"], ascending=False).team.iloc[0]
        return float(top in c)
    top = x[x.ppg == x.ppg.max()].team.tolist()
    return np.mean([t == c for t in top])


S["best_wins_title"] = [best_wins(r.league, r.season) if (r.league, r.season) in CHAMP else (1.0 if r.family == "soccer" else np.nan)
                        for r in S.itertuples()]
S.loc[(S.league == "MLS") & ~S.season.isin(CUP), "best_wins_title"] = np.nan   # 2026 MLS Cup not yet played


# year-to-year rank correlation
def rank_pairs(lg):
    x = T[T.league == lg][["season", "team", "ppg"]]
    j = x.merge(x.assign(season=x.season - 1), on=["season", "team"], suffixes=("", "_next"))
    return j


RC = []
for lg in S.league.unique():
    j = rank_pairs(lg)
    for s, p in j.groupby("season"):
        if len(p) >= 8:
            RC.append({"league": lg, "season": s, "n": len(p), "rho": spearmanr(p.ppg, p.ppg_next).statistic})
RC = pd.DataFrame(RC)


# ---------- bootstrap intervals ----------
def boot_luck(lg, B=4000):
    """Resample seasons. Each season's luck share already carries its own sampling noise, so the spread
    across seasons is the honest interval for the league's average (team-level resampling biases a variance ratio upward)."""
    v = S[(S.league == lg) & ~S.short].luck_share.values
    return np.array([rng.choice(v, len(v)).mean() for _ in range(B)])


def boot_mean(v, B=2000):
    v = np.asarray(v); return np.array([rng.choice(v, len(v)).mean() for _ in range(B)])


rows = []
for lg, ss in S.groupby("league"):
    full = ss[~ss.short]
    bl = boot_luck(lg)
    rc = RC[(RC.league == lg)]
    br = boot_mean(rc.rho) if len(rc) > 2 else np.array([np.nan])
    bw = full.best_wins_title.dropna()
    rows.append({"league": lg, "family": ss.family.iloc[0], "seasons": f"{ss.season.min()}–{ss.season.max()}", "n_seasons": len(full),
                 "teams": round(full.teams.mean(), 1), "games": round(full.g.mean(), 1),
                 "luck_share": full.luck_share.mean(), "luck_lo": np.percentile(bl, 2.5), "luck_hi": np.percentile(bl, 97.5),
                 "noll_scully": full.noll_scully.mean(),
                 "rank_corr": rc.rho.mean(), "rank_lo": np.nanpercentile(br, 2.5), "rank_hi": np.nanpercentile(br, 97.5),
                 "best_wins_title": bw.mean() if len(bw) else np.nan, "titles_n": len(bw)})
SUM = pd.DataFrame(rows).sort_values("luck_share", ascending=False)
euro = SUM[SUM.league.isin(EURO.values())]
SUM.to_csv("out/balance_summary.csv", index=False)
S.to_csv("out/balance_seasons.csv", index=False)
RC.to_csv("out/balance_rankcorr.csv", index=False)

# MLS Shield vs Cup table
ms = []
for y in sorted(CUP):
    x = T[(T.league == "MLS") & (T.season == y)].sort_values(["ppg", "wins", "gdiff"], ascending=False)
    ms.append({"season": y, "shield": teams.get(x.team.iloc[0]), "cup": CUP[y], "shield_won_cup": best_wins("MLS", y)})
pd.DataFrame(ms).to_csv("out/balance_titles.csv", index=False)

pd.set_option("display.width", 220)
print(SUM.round(3).to_string(index=False))
print("\nEurope pooled: luck", round(euro.luck_share.mean(), 3), "NS", round(euro.noll_scully.mean(), 2), "rank", round(euro.rank_corr.mean(), 3))
print(pd.DataFrame(ms).to_string(index=False))
