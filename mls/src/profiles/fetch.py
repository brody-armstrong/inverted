import json, sys, time, urllib.request, pathlib
BASE="https://app.americansocceranalysis.com/api/v1/mls"
RAW=pathlib.Path("data/raw")
def get(path, name):
    for i in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/{path}", timeout=90) as r:
                d=json.load(r); (RAW/f"{name}.json").write_text(json.dumps(d)); print(name, len(d)); return d
        except Exception as e:
            print("retry", name, e); time.sleep(3)
get("players","players"); get("teams","teams")
for s in sys.argv[1:]:
    q=f"season_name={s}"
    get(f"players/xgoals?{q}", f"xgoals_{s}")
    get(f"players/xpass?{q}", f"xpass_{s}")
    get(f"players/goals-added?{q}", f"ga_{s}")
    get(f"players/salaries?season_name={s}", f"sal_{s}")
