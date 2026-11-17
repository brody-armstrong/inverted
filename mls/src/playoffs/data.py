"""Match table for the playoff simulator, 2023-2026 (the seasons played under the current playoff format).

Inputs (all public; fetched by src/simulator/fetch_inputs.py plus the ASA games and teams endpoints):
  data/raw/mls_games_<s>.json   ASA games, 2023-2026
  data/raw/teams.json           ASA teams
  data/ext/odds_USA.csv         football-data.co.uk MLS closing odds
Closing odds follow src/paper1/random_matches.py: Pinnacle closing, market average closing where Pinnacle is missing,
bookmaker margin removed. Each priced match is inverted to home and away Poisson scoring rates whose win and loss
probabilities match the market (the inversion used in src/paper2/market_gd_mls.py and peer.py).

Output: out/playoffs/matches.pkl
Usage:  python3 src/playoffs/data.py
"""
import difflib, json, pathlib
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

RAW, EXT, OUT = pathlib.Path("data/raw"), pathlib.Path("data/ext"), pathlib.Path("out/playoffs")
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = [2023, 2024, 2025, 2026]
teams = {t["team_id"]: t for t in json.loads((RAW / "teams.json").read_text())}

rows = []
for s in SEASONS:
    for g in json.loads((RAW / f"mls_games_{s}.json").read_text()):
        if g.get("status") != "FullTime":
            continue
        rows.append({"game_id": g["game_id"], "season": s, "kickoff_utc": g["date_time_utc"][:19], "home": g["home_team_id"],
                     "away": g["away_team_id"], "hg": g["home_score"], "ag": g["away_score"],
                     "knockout": bool(g.get("knockout_game")), "matchday": g.get("matchday")})
M = pd.DataFrame(rows)
M["kickoff_utc"] = pd.to_datetime(M.kickoff_utc)
M = M.sort_values("kickoff_utc").reset_index(drop=True)
M["date"] = M.kickoff_utc.dt.normalize()

# ---- closing odds (same matching as src/paper1/random_matches.py) ----
O = pd.read_csv(EXT / "odds_USA.csv")
O = O[(O.League == "MLS") & O.Season.isin(SEASONS)].copy()
O["date"] = pd.to_datetime(O.Date, dayfirst=True)
name_map = {t["team_name"]: tid for tid, t in teams.items()}
alias = {"CF Montreal": "CF Montréal", "DC United": "D.C. United", "Los Angeles Galaxy": "LA Galaxy",
         "Inter Miami": "Inter Miami CF", "St. Louis City": "St. Louis City SC", "Charlotte": "Charlotte FC",
         "Atlanta Utd": "Atlanta United FC", "New York City": "New York City FC", "Chicago Fire": "Chicago Fire FC",
         "Houston Dynamo": "Houston Dynamo FC", "Minnesota United": "Minnesota United FC", "Portland Timbers": "Portland Timbers FC",
         "Seattle Sounders": "Seattle Sounders FC", "Vancouver Whitecaps": "Vancouver Whitecaps FC", "Orlando City": "Orlando City SC"}


def to_id(n):
    n2 = alias.get(n, n)
    if n2 in name_map:
        return name_map[n2]
    m = difflib.get_close_matches(n2, list(name_map), n=1, cutoff=0.5)
    return name_map[m[0]] if m else None


O["home"], O["away"] = O.Home.map(to_id), O.Away.map(to_id)
assert O.home.notna().all() and O.away.notna().all(), "unmatched odds team names"
for c in "HDA":
    O[f"p{c}_raw"] = 1 / pd.to_numeric(O[f"PSC{c}"], errors="coerce").fillna(pd.to_numeric(O[f"AvgC{c}"], errors="coerce"))
s = O[["pH_raw", "pD_raw", "pA_raw"]].sum(axis=1)
for c in "HDA":
    O[f"mkt_{c}"] = O[f"p{c}_raw"] / s
mm = []
for shift in (0, -1, 1):          # odds dates are UK dates
    t = O[["date", "home", "away", "mkt_H", "mkt_D", "mkt_A"]].assign(date=O.date + pd.Timedelta(days=shift))
    mm.append(M[["game_id", "date", "home", "away"]].merge(t, on=["date", "home", "away"]))
mm = pd.concat(mm).drop_duplicates("game_id")
M = M.merge(mm[["game_id", "mkt_H", "mkt_D", "mkt_A"]], on="game_id", how="left")

# ---- market-implied Poisson rates ----
G = np.arange(0, 11)
cache = {}


def rates(ph, pa):
    k = (round(ph, 3), round(pa, 3))
    if k not in cache:
        def f(x):
            p = np.outer(poisson.pmf(G, np.exp(x[0])), poisson.pmf(G, np.exp(x[1])))
            return np.array([np.tril(p, -1).sum(), np.triu(p, 1).sum()]) - [ph, pa]
        cache[k] = np.exp(least_squares(f, [0.4, 0.1]).x)
    return cache[k]


ok = M.mkt_H.notna()
R = np.array([rates(h, a) for h, a in zip(M.loc[ok, "mkt_H"], M.loc[ok, "mkt_A"])])
M.loc[ok, "mkt_lh"], M.loc[ok, "mkt_la"] = R[:, 0], R[:, 1]
M.to_pickle(OUT / "matches.pkl")
print(M.groupby(["season", "knockout"]).agg(n=("game_id", "size"), with_odds=("mkt_H", lambda x: int(x.notna().sum()))).to_string())
