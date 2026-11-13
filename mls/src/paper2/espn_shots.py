"""Parse ESPN shot commentary into a shot table, and match metadata into a match table.

Shot features are pre-shot only (location, body part, assist type, situation); the outcome is the label.
Writes out/espn_shots_<league>.pkl and out/espn_matches_<league>.pkl.
Usage: python src/paper2/espn_shots.py usa.1
"""
import gzip, json, re, sys, glob
import pandas as pd

LOCS = [("very close range", "very_close"), ("the left side of the six yard box", "six_yard"), ("the right side of the six yard box", "six_yard"),
        ("the centre of the box", "centre_box"), ("the left side of the box", "side_box"), ("the right side of the box", "side_box"),
        ("a difficult angle and long range", "angle_long"), ("a difficult angle", "tight_angle"), ("more than 35 yards", "very_long"),
        ("more than 40 yards", "very_long"), ("long range", "long"), ("outside the box", "outside_box")]


def minute(t):
    if not t:
        return None
    m = re.match(r"(\d+)'(?:\+(\d+)')?", t)
    return None if not m else int(m.group(1)) + (int(m.group(2)) / 10 if m.group(2) else 0)


def parse(text):
    """Return a dict for a shot line, or None. Own goals are returned with kind='own_goal'."""
    if text.startswith("Own Goal"):
        m = re.match(r"Own Goal by (.+?), (.+?)\.", text)
        return {"kind": "own_goal", "player": m.group(1) if m else None, "team_txt": m.group(2) if m else None}
    if not re.match(r"(Attempt (missed|blocked|saved)|Goal!|Penalty (saved|missed)!?)", text):
        return None
    goal = text.startswith("Goal!")
    m = re.search(r"(?:Goal! .*?\d+\. |Attempt \w+\. |Penalty \w+! )([^()]+?) \(([^()]+)\)", text)
    player, team = (m.group(1).strip(), m.group(2).strip()) if m else (None, None)
    pen = ("converts the penalty" in text) or text.startswith("Penalty saved") or text.startswith("Penalty missed")
    body = "header" if re.search(r"\bheader\b", text) else ("foot" if re.search(r"(right|left) footed", text) else "other")
    loc = "unknown"
    for phrase, code in LOCS:
        if f"from {phrase}" in text:
            loc = code; break
    assist = "none"
    if "Assisted by" in text:
        assist = "assisted"
        for k, c in (("with a cross", "cross"), ("with a through ball", "through_ball"), ("with a headed pass", "headed_pass")):
            if k in text:
                assist = c; break
    situ = "open"
    if "direct free kick" in text:
        situ = "free_kick"
    elif "following a corner" in text:
        situ = "corner"
    elif "set piece situation" in text:
        situ = "set_piece"
    elif "fast break" in text:
        situ = "fast_break"
    return {"kind": "penalty" if pen else "shot", "goal": int(goal), "player": player, "team_txt": team, "body": body, "loc": loc,
            "assist": assist, "situ": situ, "blocked": int(text.startswith("Attempt blocked"))}


def main(league):
    shots, matches = [], []
    for f in sorted(glob.glob(f"data/espn/{league}/*.jsonl.gz")):
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            home = [k for k, v in r["teams"].items() if v["home"]][0]; away = [k for k, v in r["teams"].items() if not v["home"]][0]
            matches.append({"event": r["id"], "date": r["date"], "season_type": r["season_type"], "home_id": home, "away_id": away,
                            "home": r["teams"][home]["name"], "away": r["teams"][away]["name"],
                            "hg": int(r["teams"][home]["score"] or 0), "ag": int(r["teams"][away]["score"] or 0),
                            "n_comm": len(r["commentary"]), "n_roster": len(r["rosters"])})
            # player name -> team id within the match, for assigning shots
            who = {}
            for p in r["rosters"]:
                who.setdefault(p["name"], p["team"])
            for c in r["commentary"]:
                if not c.get("text"):
                    continue
                p = parse(c["text"])
                if not p:
                    continue
                tid = who.get(p.get("player"))
                if tid is None and p.get("team_txt"):  # fall back to team text
                    for k, v in r["teams"].items():
                        a, b = v["name"].lower(), p["team_txt"].lower()
                        if a.startswith(b[:8]) or b.startswith(a[:8]):
                            tid = k
                if p["kind"] == "own_goal" and tid is not None:   # own goal credits the other side
                    tid = away if tid == home else home
                shots.append({"event": r["id"], "t": c.get("t"), "minute": minute(c.get("t")), "team_id": tid, "is_home": tid == home, **p})
    S = pd.DataFrame(shots); M = pd.DataFrame(matches)
    M["date"] = pd.to_datetime(M.date, utc=True)
    S.to_pickle(f"out/espn_shots_{league}.pkl"); M.to_pickle(f"out/espn_matches_{league}.pkl")
    print(league, "matches", len(M), "with commentary", (M.n_comm > 20).sum(), "| shots", (S.kind == "shot").sum(),
          "penalties", (S.kind == "penalty").sum(), "own goals", (S.kind == "own_goal").sum(), "| team unassigned", S.team_id.isna().sum())
    print(S[S.kind == "shot"].loc[:, ["loc", "body", "assist", "situ"]].apply(lambda c: c.value_counts().head(6).to_dict()).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
