"""Stage 1: fetch public ASA data for the readiness passport (cached; reruns are free).
Data: American Soccer Analysis public API. ASA caps responses at 10,000 rows, so match-level
data is fetched one team-season at a time.
Usage: python -I fetch.py <raw_dir>
"""
import sys, os, json, time, urllib.request, urllib.parse

RAW = sys.argv[1]
BASE = "https://app.americansocceranalysis.com/api/v1"
NP_SEASONS = [2022, 2023, 2024, 2025, 2026]
MLS_MIN_SEASONS = list(range(2013, 2027))
CAP = 10000


def get(path, **q):
    os.makedirs(RAW, exist_ok=True)
    key = path.replace("/", "_") + ("_" + "_".join(f"{k}-{v}" for k, v in sorted(q.items())) if q else "")
    fn = os.path.join(RAW, key + ".json")
    if os.path.exists(fn):
        return json.load(open(fn))
    url = f"{BASE}/{path}" + ("?" + urllib.parse.urlencode(q) if q else "")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "inverted-passport"})
            d = json.load(urllib.request.urlopen(req, timeout=180))
            break
        except Exception:
            time.sleep(4 * (attempt + 1))
    else:
        raise RuntimeError(f"failed: {url}")
    if isinstance(d, list) and len(d) >= CAP:
        raise RuntimeError(f"response hit the {CAP}-row cap, split it further: {url}")
    json.dump(d, open(fn, "w"))
    time.sleep(0.4)
    return d


def main():
    get("mlsnp/players"); get("mls/players"); get("mlsnp/teams"); get("mls/teams")
    for s in NP_SEASONS:
        games = get("mlsnp/games", season_name=s)
        teams = sorted({g["home_team_id"] for g in games} | {g["away_team_id"] for g in games})
        for t in teams:
            get("mlsnp/players/xgoals", season_name=s, team_id=t, split_by_games="true")
            get("mlsnp/players/goals-added", season_name=s, team_id=t, split_by_games="true")
    for s in MLS_MIN_SEASONS:
        get("mls/players/xgoals", season_name=s, split_by_teams="true")
    for s in NP_SEASONS:
        get("mls/players/goals-added", season_name=s, split_by_teams="true")
    print("fetch done:", len(os.listdir(RAW)), "files")


if __name__ == "__main__":
    main()
