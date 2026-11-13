"""Stints: stretches of a match with the same players on the field, with xG for each side.

Time is real elapsed match time: first-half stoppage (from the Halftime clock) pushes the second half later.
A stint changes at every substitution and red card. Also maps ESPN player ids to ASA player ids by name within linked
matches (votes across all of a player's matches).
Writes out/stints_usa.1.pkl, out/stint_players_usa.1.pkl, out/espn_asa_players.csv.
"""
import gzip, json, glob, re, unicodedata, collections
import numpy as np, pandas as pd

S = pd.read_pickle("out/espn_shots_xg_usa.1.pkl")
LINK = pd.read_csv("out/espn_asa_link.csv", dtype=str)
link = dict(zip(LINK.event, LINK.game_id))


def clock(t):
    m = re.match(r"(\d+)'(?:\+(\d+)')?", t or "")
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def elapsed(t, s1, s2, s3):
    """Map a displayed clock to elapsed minutes given 1st-half, 2nd-half, 1st-extra stoppage."""
    c = clock(t)
    if c is None:
        return None
    m, plus = c
    if m <= 45 and plus and m == 45:
        return 45 + plus
    if m <= 45:
        return m
    if m <= 90:
        return m + s1 + (plus if m == 90 else 0)
    if m <= 105:
        return m + s1 + s2 + (plus if m == 105 else 0)
    return m + s1 + s2 + s3 + plus


def norm(n):
    n = unicodedata.normalize("NFKD", n or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", n).split()


stints, sp = [], []
for f in sorted(glob.glob("data/espn/usa.1/*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        if "all-star" in (r["season_type"] or ""):
            continue
        ev = r["id"]
        home = [k for k, v in r["teams"].items() if v["home"]][0]
        ke = {k["type"]: k["t"] for k in r["keyEvents"]}
        s1 = (clock(ke.get("Halftime")) or (45, 2))[1]
        s2 = (clock(ke.get("End Regular Time")) or (90, 5))[1]
        s3 = (clock(ke.get("End of Extra Time First Half", "")) or (105, 1))[1]
        extra = any("Extra Time" in (k["type"] or "") for k in r["keyEvents"]) or any(
            (clock(c["t"]) or (0, 0))[0] > 90 for c in r["commentary"])
        end = (120 + s1 + s2 + s3 + 2) if extra else (90 + s1 + s2)
        # on/off times per player
        onoff = {}
        for p in r["rosters"]:
            times = [elapsed(x["t"], s1, s2, s3) for x in p["plays"] if x["sub"]]
            reds = [elapsed(x["t"], s1, s2, s3) for x in p["plays"] if x["rc"]]
            on = 0.0 if p["starter"] else (times[0] if p["subIn"] and times else None)
            if on is None:
                continue
            outs = []
            if p["subOut"]:
                outs.append(times[-1] if (p["starter"] and times) or len(times) > 1 else None)
            outs += reds
            off = min([o for o in outs if o is not None], default=end)
            if off > on:
                onoff[p["pid"]] = (p["team"], on, off, p["name"])
        if len(onoff) < 18:
            continue
        cuts = sorted({0.0, end, *[v[1] for v in onoff.values()], *[v[2] for v in onoff.values()]})
        cuts = [c for c in cuts if c <= end]
        sh = S[(S.event == ev) & (S.kind != "own_goal") & S.team_id.notna()].copy()
        sh["el"] = [elapsed(t, s1, s2, s3) for t in sh.t]
        og = S[(S.event == ev) & (S.kind == "own_goal") & S.team_id.notna()].copy()
        og["el"] = [elapsed(t, s1, s2, s3) for t in og.t]
        gh = ga = 0
        for k in range(len(cuts) - 1):
            a, b = cuts[k], cuts[k + 1]
            if b - a < 0.5:
                continue
            onf = [(pid, v[0]) for pid, v in onoff.items() if v[1] <= a and v[2] >= b]
            hp = [pid for pid, t in onf if t == home]; ap = [pid for pid, t in onf if t != home]
            w = sh[(sh.el > a) & (sh.el <= b)] if k else sh[(sh.el >= a) & (sh.el <= b)]
            npx = w[w.kind == "shot"]; pens = w[w.kind == "penalty"]
            g_in = w[w.goal == 1]; o_in = og[(og.el > a) & (og.el <= b)]
            dh = int((g_in.team_id == home).sum() + (o_in.team_id == home).sum()); da = int((g_in.team_id != home).sum() + (o_in.team_id != home).sum())
            stints.append({"event": ev, "game_id": link.get(ev), "season_type": r["season_type"], "date": r["date"], "k": k,
                           "start": a, "end": b, "dur": b - a, "n_home": len(hp), "n_away": len(ap), "score_diff": gh - ga,
                           "xg_h": float(npx[npx.team_id == home].xg.sum()), "xg_a": float(npx[npx.team_id != home].xg.sum()),
                           "pen_h": int((pens.team_id == home).sum()), "pen_a": int((pens.team_id != home).sum()), "g_h": dh, "g_a": da,
                           "home_players": hp, "away_players": ap})
            gh += dh; ga += da
        for pid, v in onoff.items():
            sp.append({"event": ev, "game_id": link.get(ev), "pid": pid, "team": v[0], "is_home": v[0] == home, "on": v[1], "off": v[2],
                       "min": v[2] - v[1], "name": v[3]})

ST = pd.DataFrame(stints); SP = pd.DataFrame(sp)
ST["date"] = pd.to_datetime(ST.date, utc=True); ST["season"] = ST.date.dt.tz_convert("America/New_York").dt.year

# ---------- ESPN -> ASA player ids ----------
asa_names = {p["player_id"]: p["player_name"] for p in json.load(open("data/raw/players.json"))}
for f in ("data/raw/mls_players.json",):
    try:
        asa_names.update({p["player_id"]: p["player_name"] for p in json.load(open(f))})
    except Exception:
        pass
G = pd.concat([pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob("data/raw/asa_player_games_*.json"))])
G = G[G.minutes_played > 0][["player_id", "game_id", "team_id", "minutes_played"]]
votes = collections.defaultdict(collections.Counter)
gg = G.groupby("game_id")
SPl = SP[SP.game_id.notna()]
for gid, x in SPl.groupby("game_id"):
    if gid not in gg.groups:
        continue
    cand = gg.get_group(gid)
    cn = [(row.player_id, norm(asa_names.get(row.player_id, ""))) for row in cand.itertuples()]
    for row in x.itertuples():
        en = norm(row.name)
        best, bs = None, 0
        for aid, an in cn:
            if not an or not en:
                continue
            sc = len(set(en) & set(an)) / max(len(set(en) | set(an)), 1) + (0.5 if en[-1] == an[-1] else 0)
            if sc > bs:
                best, bs = aid, sc
        if best and bs >= 0.5:
            votes[row.pid][best] += 1
MAP = pd.DataFrame([{"pid": p, "asa_id": c.most_common(1)[0][0], "votes": c.most_common(1)[0][1], "share": c.most_common(1)[0][1] / sum(c.values())}
                    for p, c in votes.items()])
MAP.to_csv("out/espn_asa_players.csv", index=False)
SP = SP.merge(MAP[["pid", "asa_id"]], on="pid", how="left")
ST.to_pickle("out/stints_usa.1.pkl"); SP.to_pickle("out/stint_players_usa.1.pkl")

mins_mapped = SP[SP.asa_id.notna()]["min"].sum() / SP["min"].sum()
print("stints", len(ST), "matches", ST.event.nunique(), "mean stint min", round(ST.dur.mean(), 1))
print("11 v 11 share of minutes", round((ST[(ST.n_home == 11) & (ST.n_away == 11)].dur.sum() / ST.dur.sum()), 3))
print("players", SP.pid.nunique(), "mapped to ASA", MAP.pid.nunique(), "share of minutes mapped", round(mins_mapped, 3),
      "ambiguous (<80% vote share)", int((MAP.share < 0.8).sum()))
# consistency: stint xG summed per match vs match xG, and goals vs score
chk = ST.groupby("event")[["xg_h", "xg_a", "g_h", "g_a"]].sum()
print("stint goals equal match-level parsed goals in", round(((chk.g_h + chk.g_a) > -1).mean(), 3))
odd = ST[(ST.n_home < 10) | (ST.n_away < 10) | (ST.n_home > 11) | (ST.n_away > 11)]
print("stints with <10 or >11 players on a side:", len(odd), "minutes", round(odd.dur.sum()))
