"""Domestic pathway translation: USLC / USL1 / MLS NEXT Pro -> MLS.

Pairs: a lower-league season (>= LOW_MIN minutes) followed by an MLS season in the same
or next calendar year (>= MLS_MIN minutes). For each stat and league we estimate
  1. a translation factor  = minute-weighted mean(MLS) / mean(lower)   (rate stats only)
  2. a projection model    = MLS ~ a + b * lower + position effects (weighted least squares)
and validate the projection out of sample, leaving one MLS season out at a time.
"""
import json
import numpy as np, pandas as pd

LOW_MIN, MLS_MIN = 900, 450
GROUPS = {"CB": "Centre-backs", "FB": "Fullbacks", "DM": "Midfield", "CM": "Midfield",
          "AM": "Midfield", "W": "Wingers", "ST": "Strikers"}
LEAGUE_NAMES = {"uslc": "USL Championship", "usl1": "USL League One", "mlsnp": "MLS NEXT Pro"}
RATE = ["xg_96", "shots_96", "xa_96", "kp_96", "passes_96", "recv_96", "drib_96", "int_96"]
OTHER = ["share_team_touches", "passes_completed_over_expected_p100", "gplus_96"]
STATS = RATE + OTHER
LABEL = {"xg_96": "xG", "shots_96": "Shots", "xa_96": "xA", "kp_96": "Key passes", "passes_96": "Passes",
         "recv_96": "Receptions", "drib_96": "Dribble actions", "int_96": "Interruptions",
         "share_team_touches": "Touch share", "passes_completed_over_expected_p100": "Pass acc. vs xPass",
         "gplus_96": "g+ above avg"}

d = pd.read_pickle("out/all_seasons.pkl")
d["group"] = d.general_position.map(GROUPS)
mls = d[(d.league == "mls") & (d.minutes_played >= MLS_MIN)]
low = d[(d.league != "mls") & (d.minutes_played >= LOW_MIN)]

pairs = []
for lag in (0, 1):
    m = mls.copy()
    m["season"] = m.season - lag
    p = low.merge(m, on=["player_id", "season"], suffixes=("_lo", "_mls"))
    p["lag"] = lag
    pairs.append(p)
P = pd.concat(pairs, ignore_index=True)
# keep the earliest MLS follow-up for each lower-league season
P = P.sort_values("lag").drop_duplicates(["player_id", "season", "league_lo"])
P["w"] = 2 / (1 / P.minutes_played_lo + 1 / P.minutes_played_mls)  # harmonic mean of minutes
P["group"] = P.group_mls.fillna(P.group_lo)
P = P[P.group.notna()]
print("pairs by league:", P.league_lo.value_counts().to_dict(), "| same-season:", int((P.lag == 0).sum()))

# Survival: share of strong lower-league seasons that led to real MLS minutes next year
nxt = d[d.league == "mls"].assign(season=lambda x: x.season - 1)[["player_id", "season", "minutes_played"]]
surv = low[low.season <= 2025].merge(nxt, on=["player_id", "season"], how="left", suffixes=("", "_next"))
surv["made_it"] = surv.minutes_played_next.fillna(0) >= MLS_MIN


def wls(X, y, w):
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, y * W.ravel(), rcond=None)
    return beta


def design(df, stat, groups):
    X = [np.ones(len(df)), df[f"{stat}_lo"].values]
    X += [(df.group == g).astype(float).values for g in groups[1:]]
    return np.column_stack(X)


results, models, cv_rows = {}, {}, []
for lg in LEAGUE_NAMES:
    sub = P[P.league_lo == lg].dropna(subset=[f"{s}_lo" for s in STATS] + [f"{s}_mls" for s in STATS])
    groups = sorted(sub.group.unique())
    res = {"n": int(len(sub)), "players": int(sub.player_id.nunique()), "stats": {}}
    models[lg] = {"groups": groups, "coef": {}}
    for s in STATS:
        x, y, w = sub[f"{s}_lo"].values, sub[f"{s}_mls"].values, sub.w.values
        r = {}
        if s in RATE:
            r["factor"] = float(np.average(y, weights=w) / np.average(x, weights=w))
            boots = []
            ids = sub.player_id.unique()
            rng = np.random.default_rng(0)
            for _ in range(500):
                pick = rng.choice(ids, len(ids))
                b = pd.concat([sub[sub.player_id == i] for i in pick[:0]]) if False else None
                idx = np.concatenate([np.flatnonzero(sub.player_id.values == i) for i in pick])
                boots.append(np.average(y[idx], weights=w[idx]) / np.average(x[idx], weights=w[idx]))
            r["ci"] = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
        beta = wls(design(sub, s, groups), y, w)
        r["slope"] = float(beta[1])
        r["corr"] = float(np.corrcoef(x, y)[0, 1])
        models[lg]["coef"][s] = beta.tolist()
        # leave-one-MLS-season-out validation
        err_model, err_mean, err_raw, err_factor = [], [], [], []
        mls_season = sub.season + sub.lag
        for hold in sorted(mls_season.unique()):
            tr, te = sub[mls_season != hold], sub[mls_season == hold]
            if len(te) < 5 or len(tr) < 30:
                continue
            g_tr = sorted(tr.group.unique())
            te = te[te.group.isin(g_tr)]
            b = wls(design(tr, s, g_tr), tr[f"{s}_mls"].values, tr.w.values)
            pred = design(te, s, g_tr) @ b
            pos_mean = tr.groupby("group").apply(lambda g: np.average(g[f"{s}_mls"], weights=g.w), include_groups=False)
            yt = te[f"{s}_mls"].values
            err_model += list(np.abs(pred - yt))
            err_mean += list(np.abs(te.group.map(pos_mean).values - yt))
            err_raw += list(np.abs(te[f"{s}_lo"].values - yt))
            if s in RATE:
                f = np.average(tr[f"{s}_mls"], weights=tr.w) / np.average(tr[f"{s}_lo"], weights=tr.w)
                err_factor += list(np.abs(te[f"{s}_lo"].values * f - yt))
        r["mae"] = {"model": float(np.mean(err_model)), "position_mean": float(np.mean(err_mean)),
                    "raw_lower": float(np.mean(err_raw))}
        if err_factor:
            r["mae"]["factor"] = float(np.mean(err_factor))
        r["skill"] = 1 - r["mae"]["model"] / r["mae"]["position_mean"]
        res["stats"][s] = r
    s_l = surv[surv.league == lg]
    res["survival_rate"] = float(s_l.made_it.mean())
    s_l = s_l.dropna(subset=["gplus_96"])
    s_l = s_l.assign(q=pd.qcut(s_l.gplus_96.rank(method="first"), 5, labels=False))
    res["survival_by_gplus_quintile"] = s_l.groupby("q").made_it.mean().round(3).tolist()
    results[lg] = res

json.dump({"results": results, "models": models, "labels": LABEL, "league_names": LEAGUE_NAMES,
           "thresholds": [LOW_MIN, MLS_MIN]}, open("out/pathway_model.json", "w"), indent=1)
P.to_pickle("out/pathway_pairs.pkl")

for lg, r in results.items():
    print(f"\n== {LEAGUE_NAMES[lg]}: {r['n']} pairs, {r['players']} players | MLS-minutes rate {r['survival_rate']:.1%} "
          f"| by g+ quintile {r['survival_by_gplus_quintile']}")
    for s, v in r["stats"].items():
        fac = f"factor {v['factor']:.2f} [{v['ci'][0]:.2f}-{v['ci'][1]:.2f}]" if "factor" in v else " " * 24
        m = v["mae"]
        print(f"  {LABEL[s]:20s} {fac}  slope {v['slope']:.2f}  r {v['corr']:.2f}  skill {v['skill']:+.0%}  "
              f"MAE model {m['model']:.3f} mean {m['position_mean']:.3f} raw {m['raw_lower']:.3f}"
              + (f" factor {m['factor']:.3f}" if "factor" in m else ""))
