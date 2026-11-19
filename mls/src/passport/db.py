"""Stage 2: load the cached ASA responses into a DuckDB database.
Tables: players, teams, matches, player_match (MLS NEXT Pro, one row per player per match),
mls_season (MLS minutes and g+ per player-team-season), contracts (schema only: not collected).
Usage: python -I db.py <raw_dir> <db_path>
"""
import sys, os, json, glob
import pandas as pd
import duckdb

RAW, DB = sys.argv[1], sys.argv[2]
ACTIONS = ["Dribbling", "Fouling", "Interrupting", "Passing", "Receiving", "Shooting"]


def load(name):
    return json.load(open(os.path.join(RAW, name)))


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = duckdb.connect(DB)

    pl = {}
    for lg in ("mls", "mlsnp"):
        for p in load(f"{lg}_players.json"):
            cur = pl.setdefault(p["player_id"], {"player_id": p["player_id"], "name": p["player_name"], "birth_date": None})
            if p.get("birth_date") and not cur["birth_date"]:
                cur["birth_date"] = p["birth_date"]
    players = pd.DataFrame(pl.values())
    players["birth_date"] = pd.to_datetime(players.birth_date, errors="coerce")

    teams = pd.DataFrame([dict(team_id=t["team_id"], league=lg, name=t["team_name"], short=t.get("team_short_name"))
                          for lg in ("mls", "mlsnp") for t in load(f"{lg}_teams.json")])

    games = []
    for f in sorted(glob.glob(f"{RAW}/mlsnp_games_season_name-*.json")):
        for g in json.load(open(f)):
            games.append(dict(game_id=g["game_id"], season=int(g["season_name"]), date=g["date_time_utc"][:10],
                              home_team_id=g["home_team_id"], away_team_id=g["away_team_id"],
                              knockout=bool(g.get("knockout_game"))))
    matches = pd.DataFrame(games).drop_duplicates("game_id")
    matches["date"] = pd.to_datetime(matches.date)

    xg_rows, ga_rows = [], []
    for f in glob.glob(f"{RAW}/mlsnp_players_xgoals_season_name-*_split_by_games-true_team_id-*.json"):
        fs = int(f.split("season_name-")[1].split("_")[0])
        for r in json.load(open(f)):
            xg_rows.append(dict(player_id=r["player_id"], game_id=r["game_id"], team_id=r["team_id"], file_season=fs,
                                position=r["general_position"], minutes=r["minutes_played"],
                                xg=r["xgoals"], xa=r["xassists"]))
    for f in glob.glob(f"{RAW}/mlsnp_players_goals-added_season_name-*_split_by_games-true_team_id-*.json"):
        for r in json.load(open(f)):
            d = dict(player_id=r["player_id"], game_id=r["game_id"], team_id=r["team_id"], minutes_ga=r["minutes_played"])
            for a in r["data"]:
                d["ga_" + a["action_type"].lower()] = a["goals_added_above_avg"]
            ga_rows.append(d)
    xg = pd.DataFrame(xg_rows).drop_duplicates(["player_id", "game_id"])
    ga = pd.DataFrame(ga_rows).drop_duplicates(["player_id", "game_id"])
    pm = xg.merge(ga, on=["player_id", "game_id", "team_id"], how="outer")
    pm["minutes"] = pm.minutes.fillna(pm.minutes_ga)
    pm = pm.drop(columns=["minutes_ga"]).merge(matches[["game_id", "season", "date", "knockout"]], on="game_id", how="left")
    # a few games are missing from the games endpoint: take the season from the request, mark as regular season
    pm["season"] = pm.season.fillna(pm.file_season)
    pm["knockout"] = pm.knockout.fillna(False).astype(bool)
    pm = pm.drop(columns=["file_season"])
    for a in ACTIONS:
        pm["ga_" + a.lower()] = pm.get("ga_" + a.lower(), 0.0)
        pm["ga_" + a.lower()] = pm["ga_" + a.lower()].fillna(0.0)
    pm[["xg", "xa"]] = pm[["xg", "xa"]].fillna(0.0)

    ms = []
    for f in glob.glob(f"{RAW}/mls_players_xgoals_season_name-*_split_by_teams-true.json"):
        s = int(f.split("season_name-")[1].split("_")[0])
        for r in json.load(open(f)):
            ms.append(dict(player_id=r["player_id"], season=s, team_id=r["team_id"], minutes=r["minutes_played"]))
    mls = pd.DataFrame(ms)
    gm = []
    for f in glob.glob(f"{RAW}/mls_players_goals-added_season_name-*_split_by_teams-true.json"):
        s = int(f.split("season_name-")[1].split("_")[0])
        for r in json.load(open(f)):
            gm.append(dict(player_id=r["player_id"], season=s, team_id=r["team_id"],
                           ga_total=sum(a["goals_added_above_avg"] for a in r["data"])))
    mls = mls.merge(pd.DataFrame(gm), on=["player_id", "season", "team_id"], how="left")

    for name, df in (("players", players), ("teams", teams), ("matches", matches), ("player_match", pm), ("mls_season", mls)):
        con.register("tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    # contracts: league-wide collection from club announcements isn't feasible; only the Timbers2 regulars'
    # end-of-2026 roster status is recorded, by hand, with its source. Context only: no model uses it.
    con.execute("""CREATE TABLE contracts (player_id VARCHAR, player VARCHAR, date DATE, club VARCHAR, event VARCHAR,
                   source_url VARCHAR)""")
    cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contracts_timbers2.csv")
    if os.path.exists(cf):
        c = pd.read_csv(cf)   # joined by ASA player_id; the name column is the club's spelling, for display only
        t2_ids = set(pm[(pm.team_id == "zeQZe4DqKw") & (pm.season == 2026)].player_id)
        assert set(c.player_id) <= t2_ids, set(c.player_id) - t2_ids
        con.register("tmp", c)
        con.execute("INSERT INTO contracts SELECT player_id, player, CAST(date AS DATE), club, event, source_url FROM tmp")
        con.unregister("tmp")
    for t in ("players", "teams", "matches", "player_match", "mls_season", "contracts"):
        print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main()
