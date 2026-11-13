"""Task 18a: out-of-sample test on the 2026 regular season against bookmakers' closing odds.

Every Monday of the 2026 season the Dixon-Coles model is refitted (NUTS, 2 x 300 draws; probabilities averaged over draws) on all regular-season
matches from 2023 up to that day, weighted by recency (half-life 180 days), then predicts that week's matches.
Benchmarks: Pinnacle closing odds with the margin removed (market average closing if Pinnacle missing), and base rates.
Criteria fixed before running:
  PASS 1  model log loss is lower than the base-rate log loss
  PASS 2  model log loss is within 0.015 of the market's (public models rarely beat closing lines; 0.015 = competitive)
Writes out/holdout_preds.csv, out/holdout_summary.json.
"""
import json
import numpy as np, pandas as pd
import jax, numpyro
from numpyro.infer import Predictive
from dc_model import fit_nuts, outcome_probs, model

m = pd.read_pickle("out/mls_matches.pkl")
m = m[~m.knockout.astype(bool)].copy()
m["dt"] = pd.to_datetime(m.kickoff_utc, utc=True)
hist = m[m.season.between(2023, 2026)].sort_values("dt")
test = hist[hist.season == 2026].copy()
teams = sorted(set(hist.home) | set(hist.away)); ids = {t: i for i, t in enumerate(teams)}
weeks = pd.date_range(test.dt.min().normalize() - pd.Timedelta(days=test.dt.min().weekday()), test.dt.max() + pd.Timedelta(days=7),
                      freq="W-MON", tz="UTC")
out = []
for a, b in zip(weeks[:-1], weeks[1:]):
    wk = test[(test.dt >= a) & (test.dt < b)]
    if wk.empty:
        continue
    tr = hist[hist.dt < a]
    age = (a - tr.dt).dt.days.values
    w = 0.5 ** (age / 180)
    mc = fit_nuts(tr.home.map(ids).values, tr.away.map(ids).values, len(teams), tr.hg.values, tr.ag.values,
                  w=w, neutral=tr.neutral.astype(float).values, warmup=300, samples=300, chains=2, seed=len(out))
    sm = {k: np.asarray(v) for k, v in mc.get_samples().items()}
    att = sm["att"] - sm["att"].mean(1, keepdims=True); dfn = sm["def"] - sm["def"].mean(1, keepdims=True); hc = sm["home_club"]
    h, aw = wk.home.map(ids).values, wk.away.map(ids).values
    neu = wk.neutral.astype(float).values
    lh = np.exp(sm["c"][:, None] + (sm["home"][:, None] + hc[:, h]) * (1 - neu)[None] + att[:, h] - dfn[:, aw])
    la = np.exp(sm["c"][:, None] + att[:, aw] - dfn[:, h])
    pH, pD, pA = outcome_probs(lh, la, np.broadcast_to(sm["rho"][:, None], lh.shape))
    pH, pD, pA = [p.reshape(lh.shape).mean(0) for p in (pH, pD, pA)]
    base = np.array([(tr.hg > tr.ag).mean(), (tr.hg == tr.ag).mean(), (tr.hg < tr.ag).mean()])
    for k, r in enumerate(wk.itertuples()):
        out.append({"game_id": r.game_id, "date": r.dt, "home": r.home, "away": r.away, "hg": r.hg, "ag": r.ag,
                    "m_H": pH[k], "m_D": pD[k], "m_A": pA[k], "k_H": r.mkt_H, "k_D": r.mkt_D, "k_A": r.mkt_A, "src": r.odds_src,
                    "b_H": base[0], "b_D": base[1], "b_A": base[2]})
    print(a.date(), len(wk), "fitted on", len(tr), flush=True)

O = pd.DataFrame(out).dropna(subset=["k_H"])
y = np.select([O.hg > O.ag, O.hg == O.ag], [0, 1], 2)
Y = np.eye(3)[y]


def scores(pref):
    p = O[[f"{pref}_H", f"{pref}_D", f"{pref}_A"]].values.astype(float)
    p = p / p.sum(1, keepdims=True)
    ll = -np.log(np.clip(p[np.arange(len(p)), y], 1e-12, 1)); br = ((p - Y) ** 2).sum(1)
    return ll, br


S = {}
for pref, name in (("m", "model"), ("k", "market"), ("b", "base rates")):
    ll, br = scores(pref); S[name] = {"log_loss": float(ll.mean()), "brier": float(br.mean())}
    O[f"ll_{pref}"] = ll; O[f"br_{pref}"] = br
rng = np.random.default_rng(1)
d = (O.ll_m - O.ll_k).values
bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
S["model_minus_market_ll"] = {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
db = (O.ll_m - O.ll_b).values; bb = np.array([rng.choice(db, len(db)).mean() for _ in range(4000)])
S["model_minus_base_ll"] = {"mean": float(db.mean()), "lo": float(np.percentile(bb, 2.5)), "hi": float(np.percentile(bb, 97.5))}
S["n"] = int(len(O)); S["pinnacle_share"] = float((O.src == "Pinnacle closing").mean()) if "src" in O else None
S["pass_beats_base"] = S["model"]["log_loss"] < S["base rates"]["log_loss"]
S["pass_within_0.015_of_market"] = S["model"]["log_loss"] - S["market"]["log_loss"] <= 0.015
# calibration of home-win probability (deciles)
O["bin"] = pd.qcut(O.m_H, 5, labels=False)
S["calibration_home"] = O.assign(hw=(y == 0)).groupby("bin").agg(pred=("m_H", "mean"), obs=("hw", "mean"), n=("hw", "size")).round(3).to_dict("records")
O.to_csv("out/holdout_preds.csv", index=False)
json.dump(S, open("out/holdout_summary.json", "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
