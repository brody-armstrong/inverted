"""Profile lens: match xG from team and player profiles (no betting lines), and the matchup-hypothesis test.

Plan, written before analysis: project doc claude/playoffs-profile-lens-plan.md (amended: 7 of 8 held-out seasons).
Data: src/playoffs/fetch_profiles.py (ASA). Writes results/playoffs_profile_test.json and out/playoffs/profile_model.json.
Usage: python3 src/playoffs/profile_model.py
"""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
RAW, OUT = pathlib.Path("data/raw"), pathlib.Path("out/playoffs")
OUT.mkdir(parents=True, exist_ok=True)
load = lambda n: json.loads((RAW / f"{n}.json").read_text())
ACTS = ["Shooting", "Receiving", "Passing", "Dribbling"]
ATT = ["gpf_Receiving", "gpf_Dribbling", "gpf_Passing", "gpf_Shooting", "possession", "wide_att", "central_att"]
DEF = ["gpa_Receiving", "gpa_Dribbling", "gpa_Passing", "gpa_Shooting", "press", "fb_unit", "cb_unit"]
HYP = [("H1", "gpf_Receiving", "gpa_Receiving", "Receiving: A's runners v B's leaks to receivers"),
       ("H2", "gpf_Dribbling", "gpa_Dribbling", "Dribbling: A's carriers v B's leaks to dribblers"),
       ("H3", "gpf_Passing", "gpa_Passing", "Passing: A's passers v B's leaks to passing"),
       ("H4", "gpf_Shooting", "gpa_Shooting", "Shooting: A's finishing v B's leaks to shooting"),
       ("H5", "possession", "press", "Build-up v press: A's possession v B's pressing"),
       ("H6", "wide_att", "fb_unit", "Flanks: A's wide attackers v B's fullbacks"),
       ("H7", "central_att", "cb_unit", "Centre: A's strikers and 10s v B's centre-backs")]
PRIOR = 900
GROUP = {"CB": "CB", "FB": "FB", "DM": "MF", "CM": "MF", "AM": "AM", "W": "W", "ST": "ST"}


def players(season, window=False):
    """Player rows with shrunk attack and total g+/96 (quality from the full season); weights from season or window minutes."""
    P = []
    for r in load(f"ga_teams_{season}"):
        a = {d["action_type"]: d["goals_added_above_avg"] for d in r["data"]}
        P.append({"player_id": r["player_id"], "team_id": r["team_id"], "pos": r["general_position"], "min": r["minutes_played"],
                  "att": sum(a.get(k, 0) for k in ACTS), "tot": sum(a.values())})
    P = pd.DataFrame(P)
    P = P[P.pos.isin(GROUP)]
    P["grp"] = P.pos.map(GROUP)
    for c in ("att", "tot"):
        rate = P[c] / P["min"].clip(lower=1) * 96
        gm = P.groupby("grp").apply(lambda g: g[c].sum() / g["min"].sum() * 96)
        P[c + "96"] = (P["min"] * rate + PRIOR * P.grp.map(gm)) / (P["min"] + PRIOR)
    if window:
        W = pd.DataFrame([{"player_id": r["player_id"], "team_id": r["team_id"], "wmin": r["minutes_played"]}
                          for r in load(f"ga_window_{season}")])
        P = P.merge(W, on=["player_id", "team_id"], how="left").fillna({"wmin": 0})
        P["w"] = P.wmin
    else:
        P["w"] = P["min"]
    return P


def team_features(season, window=False):
    xp = {t["team_id"]: t for t in load(f"mls_team_xpass_{season}")}
    ga = {t["team_id"]: {a["action_type"]: a for a in t["data"]} for t in load(f"mls_team_goals-added_{season}")}
    xg = {t["team_id"]: t for t in load(f"mls_team_xgoals_{season}")}
    rows = []
    for tid, t in xg.items():
        g = t["count_games"]; p, a = xp[tid], ga[tid]
        r = {"team_id": tid, "possession": p["attempted_passes_for"] / (p["attempted_passes_for"] + p["attempted_passes_against"]),
             "press": -p["passes_completed_over_expected_p100_against"]}
        for act in ACTS:
            r[f"gpf_{act}"] = a[act]["goals_added_for"] / g
            r[f"gpa_{act}"] = a[act]["goals_added_against"] / g
        rows.append(r)
    F = pd.DataFrame(rows).set_index("team_id")
    P = players(season, window)
    wmean = lambda d, c: np.average(d[c], weights=d.w) if d.w.sum() > 0 else np.nan
    for name, pos, col in (("wide_att", ("W", "FB"), "att96"), ("central_att", ("ST", "AM"), "att96"),
                           ("fb_unit", ("FB",), "tot96"), ("cb_unit", ("CB",), "tot96")):
        F[name] = P[P.pos.isin(pos)].groupby("team_id").apply(lambda d: wmean(d, col))
    Z = (F - F.mean()) / F.std()
    return Z, F


def side_rows(season, Z, games):
    rows = []
    for g in games:
        for att, dfn, xg, home in ((g["home_team_id"], g["away_team_id"], g["home_team_xgoals"], 1),
                                   (g["away_team_id"], g["home_team_id"], g["away_team_xgoals"], 0)):
            r = {"season": season, "game_id": g["game_id"], "y": xg, "home": home}
            for c in ATT: r["A_" + c] = Z.at[att, c]
            for c in DEF: r["B_" + c] = Z.at[dfn, c]
            rows.append(r)
    return rows


def design(D, inter):
    X = D[["home"] + ["A_" + c for c in ATT] + ["B_" + c for c in DEF]].copy()
    if inter:
        for h, a, b, _ in HYP:
            X[h] = D["A_" + a] * D["B_" + b]
    return sm.add_constant(X, has_constant="add")


def dev(y, mu):
    y = np.asarray(y); mu = np.asarray(mu)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0)
    return float(np.mean(2 * (t - (y - mu))))


SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def training(seasons=SEASONS):
    rows = []
    for s in seasons:
        Z, _ = team_features(s)
        ko = {g["game_id"] for g in load(f"mls_games_{s}") if g.get("knockout_game")}
        games = [g for g in load(f"mls_games_xg_{s}") if g["game_id"] not in ko]
        rows += side_rows(s, Z, games)
    return pd.DataFrame(rows).dropna()


def fit_main(D):
    return sm.GLM(D.y, design(D, False), family=sm.families.Poisson()).fit(scale="X2")


def ratings_from_fit(params, Z):
    """Map the main-effects model onto the engine's form: log rate = c + home + att[A] - def[B]."""
    att = sum(params["A_" + c] * Z[c] for c in ATT)
    dfn = -sum(params["B_" + c] * Z[c] for c in DEF)
    return {"c": float(params["const"]), "home": float(params["home"]), "att": att, "def": dfn}


if __name__ == "__main__":
    D = training()
    print("sides", len(D), "matches", D.game_id.nunique())

    cv = []
    for s in SEASONS:
        tr, te = D[D.season != s], D[D.season == s]
        res = {}
        for inter in (False, True):
            m = sm.GLM(tr.y, design(tr, inter), family=sm.families.Poisson()).fit()
            res[inter] = dev(te.y, m.predict(design(te, inter)))
        cv.append({"season": s, "dev_main": res[False], "dev_inter": res[True], "improved": res[True] < res[False]})
        print(s, round(res[False], 5), round(res[True], 5), res[True] < res[False])
    CV = pd.DataFrame(cv)
    n_imp = int(CV.improved.sum())
    pooled = float((CV.dev_main - CV.dev_inter).mean())
    supported = n_imp >= 7 and pooled > 0

    full = {}
    for inter in (False, True):
        m = sm.GLM(D.y, design(D, inter), family=sm.families.Poisson()).fit(scale="X2")
        full[inter] = m
    m = full[True]
    hyp = []
    for h, a, b, label in HYP:
        lo, hi = m.conf_int().loc[h]
        hyp.append({"id": h, "label": label, "coef": float(m.params[h]), "lo": float(lo), "hi": float(hi)})
        print(h, label, round(m.params[h], 4), (round(lo, 4), round(hi, 4)))
    mm = full[False]
    mains = [{"term": k, "coef": float(v), "lo": float(mm.conf_int().loc[k][0]), "hi": float(mm.conf_int().loc[k][1])}
             for k, v in mm.params.items()]
    out = {"seasons": SEASONS, "sides": len(D), "matches": int(D.game_id.nunique()), "cv": cv, "seasons_improved": n_imp,
           "pooled_dev_gain": pooled, "rule": "supported if interactions lower held-out deviance in >= 7 of 8 seasons and pooled gain > 0",
           "supported": bool(supported), "hypotheses": hyp, "main_effects": mains}
    print("improved in", n_imp, "of", len(SEASONS), "| pooled gain", pooled, "| SUPPORTED" if supported else "| NOT SUPPORTED")
    json.dump(out, open("results/playoffs_profile_test.json", "w"), indent=1)
    use = full[supported]
    json.dump({"interactions": bool(supported), "params": {k: float(v) for k, v in use.params.items()}},
              open(OUT / "profile_model.json", "w"), indent=1)
