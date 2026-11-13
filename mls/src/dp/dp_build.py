"""Draft MLS Designated Player list 2013-2026.

Start: Wikipedia club DP histories (as of 2025-12-19, transcribed).
Then: expand to player-seasons, match to ASA players, join spring/fall salary
releases and minutes by club, detect mid-season moves, flag rows to verify,
and add likely-missing DPs (high earners not on the list).
"""
import json, re, pathlib, unicodedata, difflib
import pandas as pd

RAW = pathlib.Path("data/raw")
load = lambda n: json.loads((RAW / f"{n}.json").read_text())
FIRST, LAST = 2013, 2026


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


def sim(a, b):
    ta, tb = norm(a), norm(b)
    if not ta or not tb:
        return 0
    if " ".join(ta) == " ".join(tb):
        return 1.0
    sa, sb = set(ta), set(tb)
    overlap = len(sa & sb) / min(len(sa), len(sb))
    return max(overlap * 0.95, difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio())


def seasons(spec):
    out = []
    for part in spec.replace("-", "–").split(","):
        part = part.strip()
        m = re.match(r"(\d{4})(–(\d{4})?)?$", part)
        if not m:
            raise ValueError(spec)
        a = int(m.group(1))
        b = a if not m.group(2) else (int(m.group(3)) if m.group(3) else LAST)
        out += list(range(a, b + 1))
    return out


# ---------- ASA reference data ----------
teams = pd.DataFrame(load("teams"))
players = {p["player_id"]: p for p in load("players")}
CLUB_ALIASES = {"Chivas USA (defunct)": "Chivas USA", "Inter Miami CF": "Inter Miami", "CF Montréal": "Montréal",
                "CF Montreal": "Montréal", "NY Red Bulls": "New York Red Bulls", "Sporting KC": "Sporting Kansas City",
                "DC United": "D.C. United", "Vancouver Whitecaps": "Vancouver Whitecaps FC"}


def team_id_for(club):
    c = CLUB_ALIASES.get(club, club)
    best = max(teams.itertuples(), key=lambda t: max(sim(c, t.team_name), sim(c, t.team_short_name)))
    return best.team_id, best.team_short_name

sal_rows = []
for s in range(FIRST, LAST + 1):
    for x in load(f"mls_sal_{s}"):
        tid = x.get("team_id")
        if tid is None:
            continue
        tid = tid[-1] if isinstance(tid, list) else tid
        sal_rows.append({"season": s, "player_id": x["player_id"], "team_id": tid, "release": x["mlspa_release"],
                         "comp": x.get("guaranteed_compensation"), "base": x.get("base_salary")})
SAL = pd.DataFrame(sal_rows)
rel = SAL.groupby("season").release.agg(["min", "max", "nunique"])
SAL["which"] = SAL.apply(lambda r: "spring" if r.release == rel.loc[r.season, "min"] else
                         ("fall" if r.release == rel.loc[r.season, "max"] else "mid"), axis=1)
SAL["name"] = SAL.player_id.map(lambda i: players.get(i, {}).get("player_name"))
SAL["src"] = "MLSPA (via ASA)"

# 2026 later-season salaries: uploaded third-party list, one snapshot per club (Mar-Sep 2026)
U = pd.read_csv("data/salaries_2026-08.csv")
xg26 = {}
for x in load("mls_xgoals_2026"):
    for t in (x["team_id"] if isinstance(x["team_id"], list) else [x["team_id"]]):
        xg26.setdefault(t, set()).add(x["player_id"])
upl = []
for r in U.itertuples():
    tid, _ = team_id_for(r.club)
    pool = xg26.get(tid, set()) | set(SAL[(SAL.season == LAST) & (SAL.team_id == tid)].player_id)
    best = max(pool, key=lambda p: sim(r.player, players.get(p, {}).get("player_name", "")), default=None)
    sc = sim(r.player, players[best]["player_name"]) if best else 0
    if sc < 0.8:
        best = max(players, key=lambda p: sim(r.player, players[p]["player_name"]))
        sc = sim(r.player, players[best]["player_name"])
        if sc < 0.9:
            continue
    upl.append({"season": LAST, "player_id": best, "team_id": tid, "release": pd.Timestamp(r.updated).strftime("%Y-%m-%d"),
                "comp": r.annual, "base": None, "which": "fall", "name": players[best]["player_name"],
                "src": f"Uploaded salary list ({r.updated})"})
UPL = pd.DataFrame(upl)
UPL = UPL[UPL.release > "2026-04-16"]
print("uploaded 2026 salaries matched:", len(UPL), "of", len(U))
SAL = pd.concat([SAL, UPL], ignore_index=True)
FALL_TEAMS = set(zip(SAL[SAL.which == "fall"].season, SAL[SAL.which == "fall"].team_id))
LIST_DATE = UPL.groupby("team_id").release.first().to_dict()

mins = {}
for s in range(FIRST, LAST + 1):
    for x in load(f"mls_xgoals_{s}"):
        t = x["team_id"] if isinstance(x["team_id"], list) else [x["team_id"]]
        mins[(s, x["player_id"])] = (x["minutes_played"], t)

# ---------- expand Wikipedia list ----------
W = pd.read_csv("data/dp_wikipedia_2026-05-12.psv", sep="|")
rows = []
for r in W.itertuples():
    tid, tname = team_id_for(r.club)
    for s in seasons(r._4):
        if FIRST <= s <= LAST:
            rows.append({"season": s, "club": tname, "team_id": tid, "player": r.player, "previous_club": r._3,
                         "years_as_dp": r._4, "source": "Wikipedia club DP history (saved Nov 2026)"})
D = pd.DataFrame(rows)


# ---------- match players ----------
ALIASES = {"Alexander Ring": "Alex Ring", "Javier Hernández": "Chicharito", "Maximiliano Moralez": "Maxi Moralez",
           "Kaku": "Alejandro Romero Gamarra"}


def match(row):
    row = row.copy()
    row["player"] = ALIASES.get(row.player, row.player)
    cand = SAL[(SAL.season == row.season) & (SAL.team_id == row.team_id)].drop_duplicates("player_id")
    if len(cand):
        sc = cand.name.map(lambda n: sim(row.player, n))
        if sc.max() >= 0.8:
            i = sc.idxmax()
            return cand.player_id[i], cand.name[i], float(sc.max()), "club roster"
    allp = [(pid, p["player_name"]) for pid, p in players.items()]
    best = max(allp, key=lambda x: sim(row.player, x[1]))
    s = sim(row.player, best[1])
    return (best[0], best[1], s, "all players") if s >= 0.85 else (None, None, s, "unmatched")

m = D.apply(match, axis=1, result_type="expand")
D[["player_id", "asa_name", "match_score", "match_how"]] = m

# ---------- join salary + minutes ----------
def info(row):
    out = {}
    p = players.get(row.player_id, {})
    bd = p.get("birth_date")
    by = int(bd[:4]) if bd else None
    out["birth_year"] = by
    out["young_dp"] = (row.season - by <= 23) if by else None
    s = SAL[(SAL.season == row.season) & (SAL.player_id == row.player_id)]
    sc = s[s.team_id == row.team_id]
    sp = sc[sc.which == "spring"].comp
    fa = sc[sc.which == "fall"].comp
    out["spring_comp"] = float(sp.iloc[0]) if len(sp) else None
    out["fall_comp"] = float(fa.iloc[0]) if len(fa) else None
    out["other_club_releases"] = ", ".join(sorted(set(s[s.team_id != row.team_id].team_id.map(
        dict(zip(teams.team_id, teams.team_short_name))).dropna()))) or ""
    mm = mins.get((row.season, row.player_id))
    out["mls_minutes"] = mm[0] if mm else 0
    clubs = [dict(zip(teams.team_id, teams.team_short_name)).get(t, t) for t in (mm[1] if mm else [])]
    out["clubs_played_for"] = " → ".join(clubs)
    two = (row.season, row.team_id) in FALL_TEAMS
    in_sp, in_fa = out["spring_comp"] is not None, out["fall_comp"] is not None
    tids = mm[1] if mm else []
    if not mm or out["mls_minutes"] == 0:
        status = "No MLS minutes"
    elif row.team_id not in tids:
        status = "Played only for another club"
    elif len(tids) > 1:
        sp_club = set(s[s.which == "spring"].team_id)
        fa_club = set(s[s.which == "fall"].team_id)
        status = "Moved mid-season" + (" (left)" if row.team_id in sp_club and row.team_id not in fa_club else
                                       " (arrived)" if row.team_id in fa_club and row.team_id not in sp_club else "")
    elif two and in_fa and not in_sp:
        status = "Joined mid-season"
    elif two and in_sp and not in_fa:
        status = "Left mid-season"
    else:
        status = "Full season"
    out["status"] = status
    flags = []
    if row.match_how == "unmatched":
        flags.append("no ASA match")
    elif row.match_score < 0.9:
        flags.append(f"check name match ({row.asa_name})")
    if status != "Full season":
        flags.append(status.lower())
    if not in_sp and not in_fa and row.player_id is not None:
        if row.season == LAST:
            flags.append(f"no 2026 salary yet: signed after the April 16 release"
                         + (f" and the club's {LIST_DATE[row.team_id]} list" if row.team_id in LIST_DATE else ""))
        else:
            flags.append("not on club's salary release")
    comp = out["fall_comp"] or out["spring_comp"]
    if comp is not None and comp < 400_000:
        flags.append("low salary for a DP: bought down or not a DP that year?")
    out["check"] = "; ".join(flags)
    return out

D = pd.concat([D, D.apply(info, axis=1, result_type="expand")], axis=1)
D["share_of_season"] = D.status.map({"Full season": 1.0, "Joined mid-season": 0.5, "Left mid-season": 0.5,
                                     "Moved mid-season (left)": 0.5, "Moved mid-season (arrived)": 0.5,
                                     "Played only for another club": 0.0, "No MLS minutes": None})

# ---------- likely missing DPs: high earners not on the list ----------
dp_keys = set(zip(D.season, D.player_id))
hi = (SAL[SAL.comp >= 1_500_000].sort_values("release").groupby(["season", "player_id"]).last().reset_index())
hi = hi[[(s, p) not in dp_keys for s, p in zip(hi.season, hi.player_id)]]
tn = dict(zip(teams.team_id, teams.team_short_name))
miss = pd.DataFrame({"season": hi.season, "club": hi.team_id.map(tn), "team_id": hi.team_id, "player": hi.name,
                     "previous_club": "", "years_as_dp": "", "source": "Salary data: earns $1.5M+ but not on list",
                     "player_id": hi.player_id, "asa_name": hi.name, "match_score": 1.0, "match_how": "salary",
                     "spring_comp": None, "fall_comp": hi.comp, "status": "", "share_of_season": None,
                     "check": ["likely missing DP: earns $2.5M+ but not on the list" if c >= 2_500_000 else
                               "earns $1.5M+ but not on the list: probably TAM or bought down, could be a missing DP"
                               for c in hi.comp]})
# players who are DPs next season and were already at that club in this season's fall release
nxt = D[D.season > FIRST][["season", "player_id", "team_id", "player", "club"]].assign(season=lambda x: x.season - 1)
fall = SAL[SAL.which == "fall"][["season", "player_id", "team_id", "comp"]]
early = nxt.merge(fall, on=["season", "player_id", "team_id"])
early = early[[(s, p) not in dp_keys for s, p in zip(early.season, early.player_id)]]
early = early[~early.player_id.isin(miss[miss.season.isin(early.season)].player_id)]
early_rows = pd.DataFrame({"season": early.season, "club": early.club, "team_id": early.team_id, "player": early.player,
                           "previous_club": "", "years_as_dp": "", "source": "DP next season, already at club in fall release",
                           "player_id": early.player_id, "asa_name": early.player, "match_score": 1.0, "match_how": "salary",
                           "spring_comp": None, "fall_comp": early.comp, "status": "", "share_of_season": None,
                           "check": "DP the next season and already at the club this fall: was he a DP from his arrival?"})
miss = pd.concat([miss, early_rows], ignore_index=True)
for c in D.columns:
    if c not in miss.columns:
        miss[c] = None
sal26 = set(SAL[SAL.season == LAST].player_id)
dp26 = set(D[D.season == LAST].player_id) | set(miss[miss.season == LAST].player_id)
late = []
for x in load(f"mls_xgoals_{LAST}"):
    pid = x["player_id"]
    if pid in sal26 or pid in dp26:
        continue
    t = x["team_id"] if isinstance(x["team_id"], list) else [x["team_id"]]
    m = x["minutes_played"]
    late.append({"season": LAST, "club": tn.get(t[-1], t[-1]), "team_id": t[-1], "player": players[pid]["player_name"],
                 "previous_club": "", "years_as_dp": "", "source": "Match data: 2026 minutes, no 2026 salary row",
                 "player_id": pid, "asa_name": players[pid]["player_name"], "match_score": 1.0, "match_how": "minutes",
                 "spring_comp": None, "fall_comp": None, "status": "Joined after April 16", "share_of_season": None,
                 "mls_minutes": m, "clubs_played_for": " → ".join(tn.get(i, i) for i in t),
                 "check": ("late arrival with 500+ minutes, not on the DP list: DP, TAM or neither?" if m >= 500
                           else "late arrival, not on the DP list: DP, TAM or neither?")})
LATE = pd.DataFrame(late)
for c in D.columns:
    if c not in LATE.columns:
        LATE[c] = None
print("late arrivals not on DP list:", len(LATE), "| 500+ min:", (LATE.mls_minutes >= 500).sum())
OUT = pd.concat([D, miss[D.columns], LATE[D.columns]], ignore_index=True).sort_values(["season", "club", "player"])
OUT["verified"] = ""
OUT["notes"] = ""
# confirmed answers from J override the sources; "verified" = Y confirmed DP, N confirmed not a DP
OUT["is_dp_final"] = OUT.source.str.startswith("Wikipedia").map({True: "Y", False: ""})
OV = pd.read_csv("data/dp_overrides.csv")
for o in OV.itertuples():
    k = (OUT.season == o.season) & (OUT.club == o.club) & (OUT.player == o.player)
    if not k.any():
        print("override matched no row:", o.season, o.club, o.player)
    OUT.loc[k, ["verified", "notes", "is_dp_final"]] = [o.is_dp, o.note, o.is_dp]
    if isinstance(o.joined, str) or isinstance(o.left, str):
        g = pd.Series([x["date_time_utc"][:10] for x in load(f"mls_games_{o.season}") if not x.get("knockout_game")])
        start, end = pd.Timestamp(g.min()), pd.Timestamp(g.max())
        a = pd.Timestamp(o.joined) if isinstance(o.joined, str) else start
        b = pd.Timestamp(o.left) if isinstance(o.left, str) else end
        OUT.loc[k, "share_of_season"] = round(max((b - a).days, 0) / (end - start).days, 2)
        OUT.loc[k, "status"] = (("Joined " + a.strftime("%b %-d")) if isinstance(o.joined, str) else
                                ("Left before season (" + b.strftime("%b %-d, %Y") + ")") if b < start else ("Left " + b.strftime("%b %-d")))
OUT.to_csv("out/dp_draft_2013_2026.csv", index=False)

print("DP player-seasons:", len(D), "| players:", D.player.nunique(), "| unmatched:", (D.match_how == "unmatched").sum())
print(D.groupby("season").size().to_dict())
print("status:", D.status.value_counts().to_dict())
print("rows flagged:", (D["check"] != "").sum(), "| likely-missing rows:", len(miss))
print(D[D.match_how == "unmatched"][["season", "club", "player"]].drop_duplicates("player").to_string())
print(D[(D.match_score < 0.9) & (D.match_how != "unmatched")][["player", "asa_name", "match_score"]].drop_duplicates().to_string())
print(miss[["season", "club", "player", "fall_comp"]].to_string())

# ---------- priorities for the person checking the list ----------
import numpy as np
c = OUT["check"].fillna("")
must = c.str.contains("no ASA match|played only for another club|no mls minutes|likely missing DP") | \
       (OUT.source.str.startswith("DP next") & (OUT.fall_comp >= 1_000_000))
look = c.str.contains("low salary|not on club|check name|TAM|late arrival") | OUT.source.str.startswith("DP next")
OUT["priority"] = np.where(OUT.verified != "", "0 Confirmed", np.where(must, "1 Must check", np.where(look, "2 Worth a look", np.where(c != "", "3 Auto-detected, info only", ""))))
OUT.to_csv("out/dp_draft_2013_2026.csv", index=False)
print(OUT.priority.value_counts().to_dict())
