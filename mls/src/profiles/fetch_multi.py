import json, time, urllib.request, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
RAW=pathlib.Path("data/raw")
def get(lg, path, name):
    f=RAW/f"{name}.json"
    if f.exists(): return name, "cached"
    for i in range(4):
        try:
            with urllib.request.urlopen(f"https://app.americansocceranalysis.com/api/v1/{lg}/{path}", timeout=120) as r:
                d=json.load(r); f.write_text(json.dumps(d)); return name, len(d)
        except Exception as e:
            err=e; time.sleep(4)
    return name, f"FAILED {err}"
jobs=[]
for lg in ["mls","uslc","usl1","mlsnp"]:
    jobs.append((lg,"players",f"{lg}_players"))
    for s in range(2017,2027):
        if lg=="mlsnp" and s<2022: continue
        for ep,k in [("players/xgoals","xgoals"),("players/xpass","xpass"),("players/goals-added","ga")]:
            jobs.append((lg,f"{ep}?season_name={s}",f"{lg}_{k}_{s}"))
with ThreadPoolExecutor(4) as ex:
    for name,res in ex.map(lambda j:get(*j), jobs): print(name,res)
