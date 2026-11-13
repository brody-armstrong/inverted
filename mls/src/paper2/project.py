"""Roster-based preseason projection (production model for the Roster Investment Simulator) and its backtest.

1. Player projection for season S from data before S: market rating in S-1 (shrunk by minutes) and S-2, age, log salary
   in S (first MLSPA release), new-to-MLS flag. Weighted least squares on earlier seasons only (rolling).
2. Expected minutes share by projected rank within the club (from earlier seasons).
3. Team strength Q = sum(share x 11 x projected rating). Match expected goal difference = h + c*(Q_home - Q_away), with h, c
   fitted to the closing market on earlier seasons. Poisson scoring (league total) -> win/draw/loss.
4. Monte Carlo of the real schedule: points, playoff spot (top 9 per conference), MLS Cup (2023+ bracket).
Backtest: 2016-2026 projected vs actual points (Spearman), against a naive baseline (last season's points per game);
2025 and 2026: model Cup probabilities vs preseason futures. Also writes the 2027 outlook from end-of-2026 rosters.
Outputs: out/proj_backtest.json, out/proj_players.pkl, out/proj_teams.pkl, out/proj_model.json
"""
import json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr

rng = np.random.default_rng(2026)
PM = pd.read_pickle("out/peer_pm_usa.1.pkl")                          # season, team (ESPN id), pid, min, rating
MAP = pd.read_csv("out/espn_asa_players.csv", dtype=str)
a2e = dict(zip(MAP.asa_id, MAP.pid))
born = {p["player_id"]: p.get("birth_date") for p in json.load(open("data/raw/players.json"))}
pname = {p["player_id"]: p["player_name"] for p in json.load(open("data/raw/players.json"))}
EM = pd.read_pickle("out/espn_matches_usa.1.pkl")
EM["season"] = EM.date.dt.tz_convert("America/New_York").dt.year
REG = EM[EM.season_type.str.contains("regular", na=False) & ~EM.home.str.contains("All-Star") & ~EM.away.str.contains("All-Star")]
ename = {}
for r in EM.itertuples():
    ename[r.home_id] = r.home; ename[r.away_id] = r.away
# ESPN team id <-> ASA team id via linked matches
link = pd.read_csv("out/espn_asa_link.csv", dtype=str)
A = pd.read_pickle("out/mls_matches.pkl")[["game_id", "home", "away"]]
L = link.merge(A, on="game_id").merge(EM[["event", "home_id", "away_id"]], on="event")
t_a2e = pd.concat([L[["home", "home_id"]].set_axis(["a", "e"], axis=1), L[["away", "away_id"]].set_axis(["a", "e"], axis=1)]).groupby("a").e.agg(
    lambda v: v.value_counts().index[0]).to_dict()

# ---------- preseason rosters (first salary release of each season) ----------
S = pd.concat([pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob("data/raw/mls_sal_*.json"))])
S["rel"] = pd.to_datetime(S.mlspa_release)
first = S.groupby("season_name").rel.transform("min")
PRE = S[S.rel == first].groupby(["season_name", "player_id"]).agg(team=("team_id", "last"), sal=("guaranteed_compensation", "max")).reset_index()
PRE = PRE.rename(columns={"season_name": "season", "player_id": "asa_id"})
PRE["team_e"] = PRE.team.map(t_a2e); PRE["pid"] = PRE.asa_id.map(a2e)
PRE["age"] = [(pd.Timestamp(f"{s}-03-01") - pd.Timestamp(born[a])).days / 365.25 if born.get(a) else np.nan for a, s in zip(PRE.asa_id, PRE.season)]

# player-season outcomes (market rating, minutes) by ESPN pid
PS = PM.groupby(["season", "pid"]).agg(rating=("rating", "first"), min=("min", "sum")).reset_index()
prev = PS.assign(season=PS.season + 1).rename(columns={"rating": "r1", "min": "m1"})
prev2 = PS.assign(season=PS.season + 2).rename(columns={"rating": "r2", "min": "m2"})
D = PRE.merge(PS, on=["season", "pid"], how="left").merge(prev, on=["season", "pid"], how="left").merge(prev2, on=["season", "pid"], how="left")
D[["r1", "m1", "r2", "m2"]] = D[["r1", "m1", "r2", "m2"]].fillna(0)
D["p1"] = D.r1 * D.m1 / (D.m1 + 900); D["p2"] = D.r2 * D.m2 / (D.m2 + 1800)
D["new"] = ((D.m1 == 0) & (D.m2 == 0)).astype(float)
D["ls"] = np.log(D.sal.clip(lower=50000)); D["ag"] = (D.age.fillna(26) - 27) / 5
D = D.dropna(subset=["team_e"]).reset_index(drop=True)
print("preseason roster rows", len(D), "| unmapped to ESPN player id", int(D.pid.isna().sum()))
FEAT = ["p1", "p2", "new", "ls", "ag"]


def Xmat(d):
    return np.c_[np.ones(len(d)), d.p1, d.p2, d.new, d.ls, d.ls * d.new, d.ag, d.ag ** 2]


def fit_player(train):
    t = train[train["min"].fillna(0) > 0]
    w = t["min"].values
    return np.linalg.lstsq(Xmat(t) * np.sqrt(w)[:, None], t.rating.values * np.sqrt(w), rcond=None)[0]


def minute_curve(train, coef):
    t = train.copy(); t["proj"] = Xmat(t) @ coef
    t["rk"] = t.groupby(["season", "team_e"]).proj.rank(ascending=False, method="first")
    t["share"] = t["min"].fillna(0) / t.groupby(["season", "team_e"])["min"].transform(lambda v: v.fillna(0).sum())
    return t.groupby("rk").share.mean().reindex(range(1, 41)).fillna(0).values


def team_Q(d, coef, curve):
    d = d.copy(); d["proj"] = Xmat(d) @ coef
    d["rk"] = d.groupby("team_e").proj.rank(ascending=False, method="first").astype(int)
    d["share"] = curve[(d.rk.clip(upper=40) - 1).values]
    d["share"] = d.share / d.groupby("team_e").share.transform("sum")
    d["contrib"] = d.share * 11 * d.proj
    return d, d.groupby("team_e").contrib.sum()


MK = pd.read_pickle("out/market_gd_mls.pkl").set_index("event").mkt_gd


def fit_link(seasons, coef, curve):
    """h, c: market expected goal difference = h + c*(Q_home - Q_away), on earlier seasons."""
    xs, ys = [], []
    for s in seasons:
        _, Q = team_Q(D[D.season == s], coef, curve)
        m = REG[REG.season == s]
        for r in m.itertuples():
            if r.event in MK.index and r.home_id in Q and r.away_id in Q:
                xs.append(Q[r.home_id] - Q[r.away_id]); ys.append(MK[r.event])
    xs, ys = np.array(xs), np.array(ys)
    c, h = np.polyfit(xs, ys, 1)
    return float(h), float(c)


EAST = {"Atlanta United FC", "Charlotte FC", "Chicago Fire FC", "FC Cincinnati", "Columbus Crew", "D.C. United", "Inter Miami CF",
        "CF Montréal", "Nashville SC", "New England Revolution", "New York City FC", "Red Bull New York", "New York Red Bulls",
        "Orlando City SC", "Philadelphia Union", "Toronto FC"}
TOTAL = 2.95
GG = np.arange(0, 11)
from scipy.stats import poisson


def probs(gd):
    lh = np.clip((TOTAL + gd) / 2, 0.15, None); la = np.clip((TOTAL - gd) / 2, 0.15, None)
    P = np.einsum("ni,nj->nij", poisson.pmf(GG[None, :], lh[:, None]), poisson.pmf(GG[None, :], la[:, None]))
    return np.tril(P, -1).sum((1, 2)), np.trace(P, axis1=1, axis2=2), np.triu(P, 1).sum((1, 2))


def simulate(sched, Q, h, c, n=4000, cup=False):
    teams = sorted(set(sched.home_id) | set(sched.away_id)); ti = {t: i for i, t in enumerate(teams)}
    gd = h + c * (sched.home_id.map(Q).fillna(Q.mean()).values - sched.away_id.map(Q).fillna(Q.mean()).values)
    pH, pD, pA = probs(gd)
    u = rng.random((n, len(sched)))
    hp = np.where(u < pH, 3, np.where(u < pH + pD, 1, 0)); ap = np.where(u < pH, 0, np.where(u < pH + pD, 1, 3))
    pts = np.zeros((n, len(teams)))
    hi = sched.home_id.map(ti).values; ai = sched.away_id.map(ti).values
    for j in range(len(sched)):
        pts[:, hi[j]] += hp[:, j]; pts[:, ai[j]] += ap[:, j]
    pts += rng.random(pts.shape) * 1e-3
    east = np.array([ename.get(t, "") in EAST for t in teams])
    playoff = np.zeros_like(pts); champ = np.zeros(len(teams))
    qv = np.array([Q.get(t, Q.mean()) for t in teams])
    for conf in (east, ~east):
        idx = np.where(conf)[0]
        order = idx[np.argsort(-pts[:, idx], axis=1)]
        for k in range(min(9, len(idx))):
            playoff[np.arange(n), order[:, k]] = 1
        if cup:
            pass
    res = pd.DataFrame({"team": teams, "name": [ename.get(t, t) for t in teams], "Q": qv, "proj_pts": pts.mean(0),
                        "pts_lo": np.percentile(pts, 10, axis=0), "pts_hi": np.percentile(pts, 90, axis=0), "playoff": playoff.mean(0)})
    if cup:
        res["cup"] = cup_prob(teams, qv, east, pts, h, c)
    return res


def ko(qa, qb, h, c, home_a=True, n=1):
    gd = h * (1 if home_a else -1) + c * (qa - qb)
    pH, pD, _ = probs(np.atleast_1d(gd))
    return pH + pD / 2


def cup_prob(teams, qv, east, pts, h, c, nsim=4000):
    """2023+ format: per conference 8v9 wild card, best-of-3 round one (1 v WC, 2v7, 3v6, 4v5), single-game semis,
    conference final, MLS Cup (higher seed hosts). Shootouts = coin flips (folded into the draw half)."""
    wins = np.zeros(len(teams))
    for s in range(nsim):
        p = pts[s % len(pts)]
        champs = []
        for conf in (east, ~east):
            idx = np.where(conf)[0]; seeds = list(idx[np.argsort(-p[idx])][:9])
            beats = lambda a, b, home=True: rng.random() < ko(qv[a], qv[b], h, c, home)[0]
            wc = seeds[7] if beats(seeds[7], seeds[8]) else seeds[8]
            def bo3(a, b):
                w = 0
                for g in range(3):
                    w += beats(a, b, g != 1)
                    if w == 2 or (g + 1 - w) == 2:
                        break
                return a if w >= 2 else b
            r1 = [bo3(seeds[0], wc), bo3(seeds[3], seeds[4]), bo3(seeds[1], seeds[6]), bo3(seeds[2], seeds[5])]
            rank = {t: i for i, t in enumerate(seeds + [wc])}
            def game(a, b):
                a, b = sorted([a, b], key=rank.get)
                return a if beats(a, b) else b
            champs.append(game(game(r1[0], r1[1]), game(r1[2], r1[3])))
        a, b = sorted(champs, key=lambda t: -p[t])
        wins[a if rng.random() < ko(qv[a], qv[b], h, c)[0] else b] += 1
    return wins / nsim


bt, teams_out = [], []
for s in range(2016, 2027):
    train = D[D.season < s]
    coef = fit_player(train); curve = minute_curve(train, coef)
    h, c = fit_link(range(max(2014, s - 4), s), coef, curve)
    players, Q = team_Q(D[D.season == s], coef, curve)
    sched = REG[REG.season == s][["home_id", "away_id"]]
    res = simulate(sched, Q, h, c, cup=s in (2025, 2026))
    act = REG[REG.season == s]
    pts = pd.concat([pd.DataFrame({"team": act.home_id, "p": np.select([act.hg > act.ag, act.hg == act.ag], [3, 1], 0)}),
                     pd.DataFrame({"team": act.away_id, "p": np.select([act.ag > act.hg, act.hg == act.ag], [3, 1], 0)})]).groupby("team").p.sum()
    res["actual_pts"] = res.team.map(pts)
    lastp = REG[REG.season == s - 1]
    lp = pd.concat([pd.DataFrame({"team": lastp.home_id, "p": np.select([lastp.hg > lastp.ag, lastp.hg == lastp.ag], [3, 1], 0)}),
                    pd.DataFrame({"team": lastp.away_id, "p": np.select([lastp.ag > lastp.hg, lastp.hg == lastp.ag], [3, 1], 0)})]).groupby("team").p.mean()
    res["naive"] = res.team.map(lp).fillna(lp.mean() if len(lp) else 1.3)
    rm = spearmanr(res.proj_pts, res.actual_pts).statistic; rn = spearmanr(res.naive, res.actual_pts).statistic
    mae = float(np.mean(np.abs(res.proj_pts - res.actual_pts)))
    bt.append({"season": s, "teams": int(len(res)), "spearman_model": float(rm), "spearman_naive": float(rn), "mae_points": mae, "h": h, "c": c})
    res["season"] = s; teams_out.append(res)
    players["season"] = s; players["name"] = players.asa_id.map(pname)
    if s == 2026:
        COEF26, CURVE26, H26, C26 = coef, curve, h, c
    print(f"{s}: Spearman model {rm:.2f} vs naive {rn:.2f}; MAE {mae:.1f} pts; link h {h:.2f} c {c:.2f}", flush=True)
T = pd.concat(teams_out)
T.to_pickle("out/proj_teams.pkl")
json.dump({"backtest": bt, "mean_spearman_model": float(np.mean([b["spearman_model"] for b in bt])),
           "mean_spearman_naive": float(np.mean([b["spearman_naive"] for b in bt]))}, open("out/proj_backtest.json", "w"), indent=1)
print("mean Spearman: model", round(np.mean([b["spearman_model"] for b in bt]), 3), "| naive", round(np.mean([b["spearman_naive"] for b in bt]), 3))


# ---------- 2027 outlook from end-of-2026 rosters (for the Roster Investment Simulator) ----------
last = S[S.season_name == 2026].sort_values("rel").groupby("player_id").agg(team=("team_id", "last"), sal=("guaranteed_compensation", "last"),
                                                                             pos=("position", "last")).reset_index()
U = pd.DataFrame({"season": 2027, "asa_id": last.player_id, "team": last.team, "sal": last.sal, "pos": last.pos})
U["team_e"] = U.team.map(t_a2e); U["pid"] = U.asa_id.map(a2e)
U["age"] = [(pd.Timestamp("2027-03-01") - pd.Timestamp(born[a])).days / 365.25 if born.get(a) else np.nan for a in U.asa_id]
U = U.merge(prev, on=["season", "pid"], how="left").merge(prev2, on=["season", "pid"], how="left")
U[["r1", "m1", "r2", "m2"]] = U[["r1", "m1", "r2", "m2"]].fillna(0)
U["p1"] = U.r1 * U.m1 / (U.m1 + 900); U["p2"] = U.r2 * U.m2 / (U.m2 + 1800)
U["new"] = ((U.m1 == 0) & (U.m2 == 0)).astype(float); U["ls"] = np.log(U.sal.clip(lower=50000)); U["ag"] = (U.age.fillna(26) - 27) / 5
U = U.dropna(subset=["team_e"])
coef = fit_player(D); curve = minute_curve(D, coef)
h, c = fit_link(range(2022, 2027), coef, curve)
P27, Q27 = team_Q(U, coef, curve)
P27["name"] = P27.asa_id.map(pname); P27["club"] = P27.team_e.map(ename)
sched = REG[REG.season == 2026][["home_id", "away_id"]]
R27 = simulate(sched, Q27, h, c, cup=True)
# residual spread of a player's next-season rating around the projection (for uncertainty in the tool)
tr = D[D["min"].fillna(0) > 900]; resid_sd = float(np.sqrt(np.average((tr.rating - Xmat(tr) @ coef) ** 2, weights=tr["min"])))
P27.to_pickle("out/proj_players_2027.pkl"); R27.to_pickle("out/proj_teams_2027.pkl")
json.dump({"coef": coef.tolist(), "curve": curve.tolist(), "h": h, "c": c, "total_goals": TOTAL, "player_resid_sd": resid_sd},
          open("out/proj_model.json", "w"), indent=1)
print(R27.sort_values("proj_pts", ascending=False)[["name", "proj_pts", "pts_lo", "pts_hi", "playoff", "cup"]].round(3).to_string(index=False))

D.to_pickle("out/proj_D.pkl")
