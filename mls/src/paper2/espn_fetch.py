"""Download ESPN soccer match summaries (lineups, substitutions, cards, shot commentary).

Usage: python src/paper2/espn_fetch.py <league> <year> [<year> ...]     e.g. usa.1 2013 2014
Writes data/espn/<league>/<year>.jsonl.gz, one compact record per match. Resumable: matches already saved are skipped.
Leagues: usa.1 (MLS), bel.1, ned.1, por.1, sco.1, tur.1, eng.1; international: fifa.worldq.concacaf, concacaf.gold, fifa.friendly ...
"""
import gzip, json, sys, time, pathlib, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def compact(s, ev):
    comp = ev["competitions"][0]
    teams = {c["team"]["id"]: {"name": c["team"]["displayName"], "home": c["homeAway"] == "home", "score": c.get("score")}
             for c in comp["competitors"]}
    rosters = []
    for t in s.get("rosters", []):
        for p in t.get("roster", []):
            rosters.append({"team": t["team"]["id"], "pid": p["athlete"]["id"], "name": p["athlete"]["displayName"],
                            "pos": (p.get("position") or {}).get("abbreviation"), "starter": p.get("starter", False),
                            "subIn": p.get("subbedIn", False), "subOut": p.get("subbedOut", False),
                            "plays": [{"t": (pl.get("clock") or {}).get("displayValue"), "sub": pl.get("substitution", False),
                                       "yc": pl.get("yellowCard", False), "rc": pl.get("redCard", False),
                                       "goal": pl.get("scoringPlay", False)} for pl in p.get("plays", [])]})
    keys = [{"type": (k.get("type") or {}).get("text"), "t": (k.get("clock") or {}).get("displayValue"),
             "team": (k.get("team") or {}).get("id"), "text": k.get("text"),
             "who": [(x.get("athlete") or {}).get("id") for x in k.get("participants", [])]} for k in s.get("keyEvents", [])]
    com = [{"t": (c.get("time") or {}).get("displayValue"), "text": c.get("text")} for c in s.get("commentary", [])]
    return {"id": ev["id"], "date": ev["date"], "name": ev.get("name"), "status": comp["status"]["type"]["name"],
            "season_type": (ev.get("season") or {}).get("slug"), "venue": (comp.get("venue") or {}).get("fullName"),
            "teams": teams, "rosters": rosters, "keyEvents": keys, "commentary": com}


def run(league, year):
    out = pathlib.Path(f"data/espn/{league}/{year}.jsonl.gz"); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with gzip.open(out, "rt") as f:
            done = {json.loads(l)["id"] for l in f}
    evs = [e for e in get(f"{BASE}/{league}/scoreboard?dates={year}&limit=1000").get("events", []) if "id" in e and e.get("competitions")]
    todo = [e for e in evs if e["id"] not in done and e["competitions"][0]["status"]["type"].get("completed")]
    print(f"{league} {year}: {len(evs)} events, {len(done)} saved, {len(todo)} to fetch", flush=True)
    t0 = time.time(); n = 0
    with gzip.open(out, "at") as f, ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(get, f"{BASE}/{league}/summary?event={e['id']}"): e for e in todo}
        for fu in as_completed(futs):
            e = futs[fu]
            try:
                f.write(json.dumps(compact(fu.result(), e)) + "\n"); n += 1
            except Exception as err:
                print("  failed", e["id"], err, flush=True)
    print(f"  saved {n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for y in sys.argv[2:]:
        run(sys.argv[1], int(y))
