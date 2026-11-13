"""Players with 2026 MLS minutes but no row in the April 16, 2026 salary release (signed later)."""
import json, csv

P = {p["player_id"]: p for p in json.load(open("data/raw/players.json"))}
tm = {t["team_id"]: t["team_short_name"] for t in json.load(open("data/raw/teams.json"))}
salids = {x["player_id"] for x in json.load(open("data/raw/mls_sal_2026.json"))}
rows = []
for x in json.load(open("data/raw/mls_xgoals_2026.json")):
    if x["player_id"] in salids:
        continue
    t = x["team_id"] if isinstance(x["team_id"], list) else [x["team_id"]]
    p = P.get(x["player_id"], {})
    bd = p.get("birth_date")
    m = x["minutes_played"]
    rows.append(["1 Must check" if m >= 500 else "2 Worth a look",
                 "Played in 2026 but not on the April 16 salary release: signed later. DP, TAM or neither?",
                 2026, tm.get(t[-1], t[-1]), p.get("player_name", ""), "Joined after April 16",
                 "" if not bd else ("Yes" if 2026 - int(bd[:4]) <= 23 else "No"), "", "", m,
                 " → ".join(tm.get(i, i) for i in t), "", "", "Match data: 2026 minutes, no 2026 salary row",
                 p.get("player_name", ""), x["player_id"], "", ""])
rows.sort(key=lambda r: (r[0], -r[9]))
h = ["Priority", "What to check", "Season", "Club", "Player", "Status (auto-detected)", "Young DP (by birth year)",
     "Spring guaranteed comp ($)", "Fall guaranteed comp ($)", "MLS minutes", "Clubs played for", "Years as DP (Wikipedia)",
     "Previous club", "Source", "Name in ASA data", "ASA player ID", "Verified (Y/N)", "Notes"]
with open("out/dp_2026_late_arrivals.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(h); w.writerows(rows)
print(len(rows), "late arrivals,", sum(r[0].startswith("1") for r in rows), "with 500+ minutes")
