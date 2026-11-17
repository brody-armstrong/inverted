"""Fetch the ASA data for the profile lens (public; American Soccer Analysis API).

  data/raw/mls_team_{xgoals,xpass,goals-added}_<s>.json   via src/profiles/fetch_teams.py (2017-2026, regular season)
  data/raw/mls_games_xg_<s>.json        match xG, 2017-2026
  data/raw/ga_teams_<s>.json            player goals added by season, split by team, regular season
  data/raw/ga_window_<s>.json           player goals added (minutes) in the last 10 weeks of the regular season, 2023-2026
  data/raw/mls_games_<s>.json           games (for knockout flags and regular-season end dates), 2017-2026
  data/raw/players.json                 player names
Usage: python3 src/playoffs/fetch_profiles.py   (after src/playoffs/data.py has run once, for 2023-2026 end dates)
"""
import json, time, urllib.request, pathlib, subprocess
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

RAW = pathlib.Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://app.americansocceranalysis.com/api/v1/mls"
SEASONS = [s for s in range(2017, 2027)]


def get(path, name):
    f = RAW / f"{name}.json"
    if f.exists():
        return name, "cached"
    for _ in range(4):
        try:
            with urllib.request.urlopen(f"{BASE}/{path}", timeout=180) as r:
                d = json.load(r); f.write_text(json.dumps(d)); return name, len(d)
        except Exception as e:
            err = e; time.sleep(4)
    return name, f"FAILED {err}"


subprocess.run(["python3", "src/profiles/fetch_teams.py"], check=True)
jobs = [("players", "players")]
for s in SEASONS:
    jobs += [(f"games/xgoals?season_name={s}", f"mls_games_xg_{s}"), (f"games?season_name={s}", f"mls_games_{s}"),
             (f"players/goals-added?season_name={s}&split_by_teams=true&stage_name=Regular%20Season", f"ga_teams_{s}")]
with ThreadPoolExecutor(4) as ex:
    print(" ".join(f"{a}:{b}" for a, b in ex.map(lambda j: get(*j), jobs)))

# last 10 weeks of each regular season (dates from the games table)
win = []
for s in (2023, 2024, 2025, 2026):
    g = pd.DataFrame(json.loads((RAW / f"mls_games_{s}.json").read_text()))
    g = g[(g.status == "FullTime") & ~g.knockout_game.fillna(False).astype(bool)]
    end = pd.to_datetime(g.date_time_utc.str[:10]).max()
    start = end - pd.Timedelta(days=70)
    win.append((f"players/goals-added?start_date={start.date()}&end_date={end.date()}&split_by_teams=true", f"ga_window_{s}"))
    print(s, "window", start.date(), "to", end.date())
with ThreadPoolExecutor(4) as ex:
    print(" ".join(f"{a}:{b}" for a, b in ex.map(lambda j: get(*j), win)))
