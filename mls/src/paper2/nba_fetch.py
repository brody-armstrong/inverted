"""NBA: player advanced stats (BPM, minutes) and each team's top 5-man lineups, from Basketball-Reference.
Polite: one request every 3.5 s. Resumable. Usage: python src/paper2/nba_fetch.py 2014 2015 ...
Writes data/nba/advanced_<y>.csv and data/nba/lineups_<y>.csv.
"""
import io, sys, time, pathlib, urllib.request
import pandas as pd

H = {"User-Agent": "Mozilla/5.0 (research; low rate)"}


def page(url):
    time.sleep(3.5)
    req = urllib.request.Request(url, headers=H)
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore").replace("<!--", "").replace("-->", "")


for y in map(int, sys.argv[1:]):
    adv = pathlib.Path(f"data/nba/advanced_{y}.csv")
    if not adv.exists():
        t = pd.read_html(io.StringIO(page(f"https://www.basketball-reference.com/leagues/NBA_{y}_advanced.html")),
                         attrs={"id": "advanced"})[0]
        t = t[t.Player != "Player"]; t.to_csv(adv, index=False)
    a = pd.read_csv(adv)
    tcol = "Team" if "Team" in a.columns else "Tm"
    teams = sorted(set(a[tcol].dropna()) - {"TOT", "2TM", "3TM", "4TM", "5TM"})
    out = pathlib.Path(f"data/nba/lineups_{y}.csv")
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["team"])
    rows = [have] if len(have) else []
    for tm in teams:
        if tm in set(have.team):
            continue
        try:
            h = page(f"https://www.basketball-reference.com/teams/{tm}/{y}/lineups/")
            L = pd.read_html(io.StringIO(h), attrs={"id": "lineups_5-man_"})[0]
            L.columns = [b if a.startswith("Unnamed") else f"{a}|{b}" for a, b in L.columns]
            L = L[L.Lineup.notna() & (L.Lineup != "Team Average")]
            L["team"] = tm; L["season"] = y
            rows.append(L)
            pd.concat(rows).to_csv(out, index=False)
        except Exception as e:
            print(y, tm, "failed", e, flush=True)
    print(y, "teams", len(teams), "lineups", len(pd.read_csv(out)) if out.exists() else 0, flush=True)
