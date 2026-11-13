import json, time, urllib.request, pathlib
from concurrent.futures import ThreadPoolExecutor
RAW=pathlib.Path("data/raw")
def get(path,name):
    f=RAW/f"{name}.json"
    if f.exists(): return name,"cached"
    for i in range(4):
        try:
            with urllib.request.urlopen(f"https://app.americansocceranalysis.com/api/v1/mls/{path}",timeout=120) as r:
                d=json.load(r); f.write_text(json.dumps(d)); return name,len(d)
        except Exception as e: err=e; time.sleep(4)
    return name,f"FAILED {err}"
jobs=[(f"games?season_name={s}",f"mls_games_{s}") for s in range(2013,2027)]
jobs+=[(f"players/salaries?season_name={s}",f"mls_sal_{s}") for s in range(2013,2027)]
jobs+=[(f"players/xgoals?season_name={s}",f"mls_xgoals_{s}") for s in range(2013,2017)]
with ThreadPoolExecutor(4) as ex: print(" ".join(f"{a}:{b}" for a,b in ex.map(lambda j:get(*j),jobs)))
