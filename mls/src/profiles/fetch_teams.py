import json, time, urllib.request, pathlib
from concurrent.futures import ThreadPoolExecutor
RAW=pathlib.Path("data/raw")
def get(ep,s):
    f=RAW/f"mls_team_{ep}_{s}.json"
    if f.exists(): return f.name,"cached"
    for i in range(4):
        try:
            with urllib.request.urlopen(f"https://app.americansocceranalysis.com/api/v1/mls/teams/{ep}?season_name={s}&stage_name=Regular%20Season",timeout=120) as r:
                d=json.load(r); f.write_text(json.dumps(d)); return f.name,len(d)
        except Exception as e: err=e; time.sleep(4)
    return f.name,f"FAILED {err}"
jobs=[(ep,s) for ep in ["xgoals","xpass","goals-added"] for s in range(2017,2027)]
with ThreadPoolExecutor(4) as ex:
    print(" ".join(f"{a}:{b}" for a,b in ex.map(lambda j:get(*j),jobs)))
