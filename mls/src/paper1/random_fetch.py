"""Raw data for 'Is MLS built to be random?'
- ASA match xG per season (MLS 2013-2026), stadia (already saved)
- football-data.co.uk closing odds: MLS (USA.csv) and European leagues (mmz4281 season files)
- Open-Meteo: stadium elevation; hourly kickoff weather
- North American leagues: NFL games (nflverse), MLB/NHL/NBA standings
"""
import json, time, pathlib, urllib.request, urllib.parse, io, csv
import pandas as pd

RAW = pathlib.Path("data/raw"); EXT = pathlib.Path("data/ext"); EXT.mkdir(exist_ok=True)


def get(url, path=None, tries=4, binary=False):
    p = pathlib.Path(path) if path else None
    if p and p.exists():
        return p.read_bytes() if binary else p.read_text()
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if p:
                p.write_bytes(b)
            return b if binary else b.decode("utf-8-sig", errors="replace")
        except Exception as e:
            last = e; time.sleep(3 + 3 * i)
    raise RuntimeError(f"{url}: {last}")


# 1. ASA match xG
for s in range(2013, 2027):
    get(f"https://app.americansocceranalysis.com/api/v1/mls/games/xgoals?season_name={s}", RAW / f"mls_games_xg_{s}.json")
print("ASA xG done")

# 2. odds
get("https://www.football-data.co.uk/new/USA.csv", EXT / "odds_USA.csv")
EU = ["E0", "SP1", "I1", "D1", "F1", "N1", "P1", "B1", "T1", "SC0", "G1"]
for y in range(12, 26):
    code = f"{y:02d}{y+1:02d}"
    for lg in EU:
        try:
            get(f"https://www.football-data.co.uk/mmz4281/{code}/{lg}.csv", EXT / f"odds_{lg}_{code}.csv")
        except Exception as e:
            print("odds miss", lg, code, e)
print("odds done")

# 3. stadium elevation
st = json.load(open(RAW / "mls_stadia.json"))
st = [s for s in st if s.get("latitude") is not None]
el_path = EXT / "stadium_elevation.json"
if not el_path.exists():
    out = {}
    for i in range(0, len(st), 50):
        chunk = st[i:i + 50]
        q = "latitude=" + ",".join(str(s["latitude"]) for s in chunk) + "&longitude=" + ",".join(str(s["longitude"]) for s in chunk)
        r = json.loads(get("https://api.open-meteo.com/v1/elevation?" + q))
        for s, e in zip(chunk, r["elevation"]):
            out[s["stadium_id"]] = e
    json.dump(out, open(el_path, "w"))
print("elevation done")

# 4. kickoff weather: one archive call per stadium covering all its match dates
games = []
for s in range(2013, 2027):
    for g in json.load(open(RAW / f"mls_games_{s}.json")):
        games.append(g)
G = pd.DataFrame(games)
G["date"] = G.date_time_utc.str[:10]
wx_dir = EXT / "weather"; wx_dir.mkdir(exist_ok=True)
stmap = {s["stadium_id"]: s for s in st}
for sid, g in G.groupby("stadium_id"):
    if sid not in stmap:
        continue
    f = wx_dir / f"{sid}.json"
    if f.exists():
        continue
    a, b = g.date.min(), min(g.date.max(), "2026-11-09")
    # the day after covers evening kickoffs that fall after midnight UTC
    b = (pd.Timestamp(b) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    q = urllib.parse.urlencode({"latitude": stmap[sid]["latitude"], "longitude": stmap[sid]["longitude"], "start_date": a,
                                "end_date": b, "hourly": "temperature_2m,relative_humidity_2m,precipitation", "timezone": "GMT"})
    try:
        get("https://archive-api.open-meteo.com/v1/archive?" + q, f)
    except Exception as e:
        print("weather miss", sid, e)
    time.sleep(0.5)
print("weather done")

# 5. North American leagues
get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", EXT / "nfl_games.csv")
for y in range(2013, 2026):
    get(f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={y}&standingsTypes=regularSeason", EXT / f"mlb_standings_{y}.json")
    get(f"https://site.api.espn.com/apis/v2/sports/basketball/nba/standings?season={y + 1}", EXT / f"nba_standings_{y + 1}.json")
print("NA done")
