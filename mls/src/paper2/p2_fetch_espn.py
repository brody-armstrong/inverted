"""Paper 2: download ESPN match summaries (lineups, substitution minutes, cards, shot commentary).

Usage: python3 src/paper2/p2_fetch_espn.py <league> <first_year> <last_year> [max_minutes]
  league = usa.1 (MLS, calendar seasons) or bel.1 / ned.1 / por.1 / sco.1 / tur.1 (Aug-May seasons, labelled by end year)
Resumable: events already saved are skipped. Saves only the fields the analysis uses, one gzipped JSON per match:
  data/espn/<league>/<season>/<event_id>.json.gz
"""
import gzip, json, pathlib, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
OUT = pathlib.Path("data/espn")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def season_events(lg, season):
    rng = f"{season}0101-{season}1231" if lg == "usa.1" else f"{season - 1}0701-{season}0630"
    sb = get(f"{BASE}/{lg}/scoreboard?dates={rng}&limit=1000")
    evs = []
    for e in (sb or {}).get("events", []):
        comp = e["competitions"][0]
        if comp.get("status", {}).get("type", {}).get("completed"):
            evs.append({"id": e["id"], "date": e["date"], "season_slug": e.get("season", {}).get("slug")})
    return evs


def slim(s, ev):
    comp = s.get("header", {}).get("competitions", [{}])[0]
    teams = [{"id": c["team"]["id"], "name": c["team"]["displayName"], "homeAway": c["homeAway"], "score": c.get("score")}
             for c in comp.get("competitors", [])]
    rosters = []
    for t in s.get("rosters", []):
        pl = []
        for p in t.get("roster", []):
            pl.append({"id": p["athlete"]["id"], "name": p["athlete"]["displayName"], "pos": p.get("position", {}).get("abbreviation"),
                       "starter": p.get("starter"), "subbedIn": p.get("subbedIn"), "subbedOut": p.get("subbedOut"),
                       "plays": [{"clock": pp.get("clock", {}).get("displayValue"), "sub": pp.get("substitution"),
                                  "yc": pp.get("yellowCard"), "rc": pp.get("redCard")} for pp in p.get("plays", [])]})
        rosters.append({"team_id": t["team"]["id"], "homeAway": t.get("homeAway"), "players": pl})
    key = [{"type": k.get("type", {}).get("text"), "clock": k.get("clock", {}).get("displayValue"), "team": k.get("team", {}).get("id"),
            "players": [x.get("athlete", {}).get("id") for x in k.get("participants", [])], "text": k.get("text")}
           for k in s.get("keyEvents", [])]
    com = [{"t": c.get("time", {}).get("displayValue"), "text": c.get("text")} for c in s.get("commentary", [])]
    return {"id": ev["id"], "date": ev["date"], "season_slug": ev["season_slug"], "teams": teams, "rosters": rosters,
            "key": key, "commentary": com}


def run(lg, y0, y1, max_min=9.0):
    t0 = time.time()
    for season in range(y0, y1 + 1):
        d = OUT / lg / str(season); d.mkdir(parents=True, exist_ok=True)
        idx = d / "_events.json"
        evs = json.load(open(idx)) if idx.exists() else season_events(lg, season)
        json.dump(evs, open(idx, "w"))
        todo = [e for e in evs if not (d / f"{e['id']}.json.gz").exists()]

        def one(e):
            s = get(f"{BASE}/{lg}/summary?event={e['id']}")
            if s:
                with gzip.open(d / f"{e['id']}.json.gz", "wt") as f:
                    json.dump(slim(s, e), f)
            return bool(s)

        done = 0
        with ThreadPoolExecutor(8) as ex:
            for ok in ex.map(one, todo):
                done += ok
                if (time.time() - t0) / 60 > max_min:
                    break
        have = len(list(d.glob("*.json.gz")))
        print(f"{lg} {season}: {len(evs)} completed events, {have} saved", flush=True)
        if (time.time() - t0) / 60 > max_min:
            print("time budget reached; rerun to resume"); return


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 9.0)
