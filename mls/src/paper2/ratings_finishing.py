"""Second rating version: xG rating + finishing skill + shot-stopping.

xG-based plus-minus measures chance creation and prevention but cannot see a player who scores more than his chances
predict (finishing) or a keeper who concedes fewer than the shots he faces predict (shot-stopping). Both are estimated with
empirical-Bayes shrinkage on career-to-date data (that season and earlier), since single-season goals-minus-xG is mostly luck:
    observed per-shot residual = true skill + noise;  noise variance = mean xG(1-xG) per shot / shots
    skill variance tau^2 = weighted variance of observed residuals - mean noise variance
    shrunk skill = observed * n / (n + k),  k = noise variance per shot / tau^2
Contribution per 90 = shrunk skill per shot x shots per 90 (shooters); shrunk goals prevented per shot faced x shots faced per 90 (keepers).
Writes out/ratings_mls_goals.pkl (adds columns finish90, gk90, rating_goals) and out/finishing_meta.json.
"""
import json, glob
import numpy as np, pandas as pd

K = pd.read_pickle("out/ratings_mls.pkl")
G = pd.concat([pd.DataFrame(json.load(open(f))).assign(season=int(f[-9:-5])) for f in sorted(glob.glob("data/raw/asa_player_games_*.json"))])
P = G.groupby(["player_id", "season"]).agg(shots=("shots", "sum"), goals=("goals", "sum"), xg=("xgoals", "sum"), minutes=("minutes_played", "sum")).reset_index()
GK = pd.concat([pd.DataFrame(json.load(open(f))).assign(season=int(f[-9:-5])) for f in sorted(glob.glob("data/raw/asa_gk_*.json"))])
GK = GK.groupby(["player_id", "season"]).agg(faced=("shots_faced", "sum"), conceded=("goals_conceded", "sum"), xgf=("xgoals_gk_faced", "sum"),
                                              gmin=("minutes_played", "sum")).reset_index()


def eb(df, n, resid, noise_per_shot, tau2_fixed=None):
    """Career-to-date empirical Bayes. df sorted by player, season; returns shrunk per-shot skill and k."""
    df = df.sort_values(["player_id", "season"]).copy()
    df["cn"] = df.groupby("player_id")[n].cumsum(); df["cr"] = df.groupby("player_id")[resid].cumsum()
    f = df.cr / df.cn.clip(lower=1)
    ok = df.cn >= 20
    w = df.cn[ok]
    var_obs = np.average((f[ok] - np.average(f[ok], weights=w)) ** 2, weights=w)
    noise = np.average(noise_per_shot / df.cn[ok], weights=w)
    tau2 = tau2_fixed if tau2_fixed is not None else max(var_obs - noise, 1e-6)
    k = noise_per_shot / tau2
    df["skill"] = f * df.cn / (df.cn + k)
    return df, k, tau2


P["resid"] = P.goals - P.xg
noise_shot = (P.xg / P.shots.clip(lower=1) * (1 - P.xg / P.shots.clip(lower=1)))[P.shots > 0].mean()
# Skill variance from year-to-year covariance of per-shot residuals (players with >= 20 shots in both seasons).
# The pooled method-of-moments estimate is ~0; this is the most generous estimate the data allow, deliberately
# favouring stars (a choice that works against the weak-link hypothesis).
P["f"] = P.resid / P.shots.clip(lower=1)
J = P.merge(P.assign(season=P.season - 1), on=["player_id", "season"], suffixes=("", "_n"))
J = J[(J.shots >= 20) & (J.shots_n >= 20)]; wj = np.sqrt(J.shots * J.shots_n)
cov = np.average((J.f - np.average(J.f, weights=wj)) * (J.f_n - np.average(J.f_n, weights=wj)), weights=wj)
P, k_f, tau_f = eb(P, "shots", "resid", noise_shot, tau2_fixed=max(cov, 1e-6))
P["finish90"] = P.skill * P.shots / P.minutes.clip(lower=90) * 90
GK["resid"] = GK.xgf - GK.conceded                     # positive = goals prevented
noise_gk = 0.09
GK, k_g, tau_g = eb(GK.rename(columns={"faced": "n_faced"}), "n_faced", "resid", noise_gk)
GK["gk90"] = GK.skill * GK.n_faced / GK.gmin.clip(lower=90) * 90

K = K.merge(P[["player_id", "season", "finish90"]].rename(columns={"player_id": "asa_id"}), on=["asa_id", "season"], how="left")
K = K.merge(GK[["player_id", "season", "gk90"]].rename(columns={"player_id": "asa_id"}), on=["asa_id", "season"], how="left")
K[["finish90", "gk90"]] = K[["finish90", "gk90"]].fillna(0)
K["rating_goals"] = K.rating + K.finish90 + K.gk90
K.to_pickle("out/ratings_mls_goals.pkl")
meta = {"finishing_k_shots": float(k_f), "finishing_skill_sd_per_shot": float(np.sqrt(tau_f)),
        "gk_k_shots_faced": float(k_g), "gk_skill_sd_per_shot": float(np.sqrt(tau_g))}
json.dump(meta, open("out/finishing_meta.json", "w"), indent=1)
print(json.dumps({k: round(v, 4) for k, v in meta.items()}))
big = K[K.minutes >= 900]
print("SD per 90 among 900+ min: xG rating", round(big.rating.std(), 3), "| finishing", round(big.finish90.std(), 3), "| keepers (nonzero)",
      round(big.gk90[big.gk90 != 0].std(), 3), "| combined", round(big.rating_goals.std(), 3))
names = pd.read_pickle("out/stint_players_usa.1.pkl").drop_duplicates("pid").set_index("pid").name
top = big.assign(name=big.pid.map(names)).sort_values("finish90", ascending=False)
print(top[["name", "season", "rating", "finish90", "rating_goals"]].head(8).round(3).to_string(index=False))
