"""Inputs for the simulator path that earlier had no fetch script.

- data/raw/mls_stadia.json          ASA stadia (needed by src/paper1/random_matches.py)
- data/raw/mls_games_xg_<s>.json    ASA match xG, 2013-2026 (same request as src/paper1/random_fetch.py step 1)
- data/ext/odds_USA.csv             football-data.co.uk MLS odds (same request as src/paper1/random_fetch.py step 2)

Runs only these requests; random_fetch.py also downloads weather, other sports and European odds, which the simulator
does not need. Resumable: existing files are kept.
Usage: python3 src/simulator/fetch_inputs.py
"""
import pathlib, time, urllib.request

RAW, EXT = pathlib.Path("data/raw"), pathlib.Path("data/ext")
RAW.mkdir(parents=True, exist_ok=True); EXT.mkdir(parents=True, exist_ok=True)
ASA = "https://app.americansocceranalysis.com/api/v1/mls"


def get(url, path, tries=4):
    p = pathlib.Path(path)
    if p.exists():
        return "cached"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
            return "ok"
        except Exception as e:
            err = e; time.sleep(3 + 3 * i)
    raise RuntimeError(f"{url}: {err}")


print("stadia", get(f"{ASA}/stadia", RAW / "mls_stadia.json"))
for s in range(2013, 2027):
    get(f"{ASA}/games/xgoals?season_name={s}", RAW / f"mls_games_xg_{s}.json")
print("match xG done")
print("odds", get("https://www.football-data.co.uk/new/USA.csv", EXT / "odds_USA.csv"))
