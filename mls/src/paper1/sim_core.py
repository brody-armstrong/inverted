"""Shared pieces for the randomness study (targets, Dixon-Coles probabilities, calibration, brackets).

Task 16: cost out each league-design feature.

For every posterior draw of every MLS season (2013-2026, 2020 excluded as a bubble season) we switch MLS features,
one combination at a time, to European values and recompute the table's luck share:
  schedule  actual unbalanced fixture list  -> perfectly balanced (every team faces the league average)
  length    34 games                        -> 38 games
  talent    MLS spread of team strength     -> European spread (talent SD in points per game)
  home      MLS home advantage              -> European home points share
Luck share is computed exactly from Dixon-Coles outcome probabilities:
  E[var of ppg across teams] = var(team expected ppg) + mean(team ppg variance)
  chance variance            = pooled per-game points variance / games
  luck share                 = chance variance / E[var]
Shapley values over the 16 combinations split the MLS-Europe gap; intervals are 2.5/97.5% over posterior draws.

Then Monte Carlo replays of whole seasons (outcomes drawn from the same probabilities) give, per format,
the probability the strongest team wins the trophy (Shield table or MLS Cup bracket).
Writes out/sim_shapley.csv, out/sim_formats.csv, out/sim_checks.json, out/sim_rankcorr.json.
"""
import itertools, json, math, pickle, glob, re
import numpy as np, pandas as pd
from dc_model import outcome_probs

rng = np.random.default_rng(11)
P = pickle.load(open("out/dc_posteriors.pkl", "rb"))
M = pd.read_pickle("out/mls_matches.pkl")
R = M[~M.knockout.astype(bool)]
SEASONS = [s for s in range(2013, 2027) if s != 2020]
D = 400

# ---------- European targets ----------
BS = pd.read_csv("out/balance_seasons.csv")
EU_NAMES = ["Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1", "Eredivisie", "Liga Portugal",
            "Belgian Pro League", "Süper Lig", "Scottish Premiership", "Greek Super League"]
eu = BS[BS.league.isin(EU_NAMES)].copy()
eu["talent_sd"] = np.sqrt((eu.var_obs - eu.var_luck).clip(lower=0))
T_EU = float(eu.talent_sd.mean())
mls_bs = BS[(BS.league == "MLS") & ~BS.short]
T_MLS_OBS = float(np.sqrt((mls_bs.var_obs - mls_bs.var_luck).clip(lower=0)).mean())
hp = []
for f in glob.glob("data/ext/odds_*_*.csv"):
    if "USA" in f:
        continue
    d = pd.read_csv(f, encoding="latin-1", usecols=lambda c: c in {"FTHG", "FTAG"}).dropna()
    hpts = (3 * (d.FTHG > d.FTAG) + (d.FTHG == d.FTAG)).sum(); apts = (3 * (d.FTAG > d.FTHG) + (d.FTHG == d.FTAG)).sum()
    hp.append((hpts, apts))
hp = np.array(hp).sum(0); H_EU = float(hp[0] / hp.sum())
LUCK_EU = float(eu.luck_share.mean())
print(f"targets: Europe talent SD {T_EU:.3f} (MLS observed {T_MLS_OBS:.3f}); Europe home points share {H_EU:.3f}; Europe luck {LUCK_EU:.3f}")


def probs(dr, idx, i, j, neu, k, r, chunk=60):
    if len(idx) > chunk:
        out = [probs(dr, idx[a:a + chunk], i, j, neu, k[a:a + chunk], r[a:a + chunk]) for a in range(0, len(idx), chunk)]
        return tuple(np.concatenate([o[q] for o in out]) for q in range(3))
    """Outcome probabilities for fixtures i (home) v j (away) for draws idx; k scales att/def, r scales home advantage."""
    c = dr["c"][idx][:, None]; rho = dr["rho"][idx][:, None]
    att, dfn, hc, h = dr["att"][idx] * k[:, None], dr["def"][idx] * k[:, None], dr["home_club"][idx], dr["home"][idx][:, None]
    home = r[:, None] * (h + hc[:, i]) * (1 - neu)[None, :]
    lh = np.exp(c + home + att[:, i] - dfn[:, j]); la = np.exp(c + att[:, j] - dfn[:, i])
    pH, pD, pA = outcome_probs(lh, la, np.broadcast_to(rho, lh.shape))
    return pH.reshape(lh.shape), pD.reshape(lh.shape), pA.reshape(lh.shape)


def team_moments(pH, pD, pA, i, j, n):
    """Per draw: each team's summed expected points, summed variance, games; pooled P(win), P(draw)."""
    eh, ea = 3 * pH + pD, 3 * pA + pD
    vh, va = 9 * pH + pD - eh ** 2, 9 * pA + pD - ea ** 2
    Dn = pH.shape[0]
    E = np.zeros((Dn, n)); V = np.zeros((Dn, n))
    for d in range(Dn):
        E[d] = np.bincount(i, eh[d], n) + np.bincount(j, ea[d], n)
        V[d] = np.bincount(i, vh[d], n) + np.bincount(j, va[d], n)
    g = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    pw = (pH + pA).mean(1) / 2; pdw = pD.mean(1)
    return E, V, g, pw, pdw


def drr(n):
    i, j = np.where(~np.eye(n, dtype=bool)); return i, j, np.zeros(len(i))


def balanced_sd_and_home(dr, idx, n, k, r):
    i, j, neu = drr(n)
    pH, pD, pA = probs(dr, idx, i, j, neu, k, r)
    E, V, g, _, _ = team_moments(pH, pD, pA, i, j, n)
    hs = (3 * pH + pD).sum(1) / ((3 * pH + pD).sum(1) + (3 * pA + pD).sum(1))
    return (E / g).std(1, ddof=1), hs


def calibrate(dr, idx, n, target, which, k_fixed=None):
    """Bisection per draw: k so that balanced talent SD = target, or r so that home points share = target."""
    lo = np.full(len(idx), 0.0 if which == "r" else 0.3); hi = np.full(len(idx), 3.0 if which == "r" else 6.0)
    for _ in range(22):
        mid = (lo + hi) / 2
        kk = mid if which == "k" else (np.ones(len(idx)) if k_fixed is None else k_fixed)
        sd, hs = balanced_sd_and_home(dr, idx, n, kk, mid if which == "r" else np.ones(len(idx)))
        val = sd if which == "k" else hs
        up = val < target
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return (lo + hi) / 2


FEAT = ["schedule", "length", "talent", "home"]


def luck_share(dr, idx, n, fix, combo, k_eu, r_eu):
    k = k_eu if "talent" in combo else np.ones(len(idx)); r = r_eu if "home" in combo else np.ones(len(idx))
    L = 38 if "length" in combo else 34
    if "schedule" in combo:
        i, j, neu = drr(n)
    else:
        i, j, neu = fix
    pH, pD, pA = probs(dr, idx, i, j, neu, k, r)
    E, V, g, pw, pdw = team_moments(pH, pD, pA, i, j, n)
    mean_pg = E / g; var_pg = V / g                     # per-game mean and variance for each team
    exp_var_obs = mean_pg.var(1, ddof=1) + (var_pg / L).mean(1)
    p3, p1 = pw, pdw
    var_match = 9 * p3 + p1 - (3 * p3 + p1) ** 2
    return np.minimum(var_match / L / exp_var_obs, 1), mean_pg, var_pg



PO = {2013: (10, "a"), 2014: (10, "a"), 2015: (12, "b"), 2016: (12, "b"), 2017: (12, "b"), 2018: (12, "b"),
      2019: (14, "c"), 2021: (14, "c"), 2022: (14, "c"), 2023: (18, "d"), 2024: (18, "d"), 2025: (18, "d"), 2026: (18, "d")}


def circle_rounds(n, i0, j0):
    """Double round robin by the circle method (bye if n is odd); returns rounds as index arrays into the DRR pair list."""
    pos = {(a, b): x for x, (a, b) in enumerate(zip(i0, j0))}
    t = list(range(n)) + ([None] if n % 2 else [])
    m = len(t); out = []
    for rd in range(m - 1):
        pr = [(t[q], t[m - 1 - q]) for q in range(m // 2)]
        pr = [(a, b) if (rd + q) % 2 == 0 else (b, a) for q, (a, b) in enumerate(pr) if a is not None and b is not None]
        out.append(np.array([pos[p] for p in pr])); out.append(np.array([pos[(b, a)] for a, b in pr]))
        t = [t[0]] + [t[-1]] + t[1:-1]
    return out


def sim_table(pH, pD, i, j, n, L_scale=None):
    """pH, pD: (S, F) per-replay probabilities. Returns points per game (S, n) with tiny random tiebreak."""
    u = rng.random(pH.shape)
    hpts = np.where(u < pH, 3, np.where(u < pH + pD, 1, 0)); apts = np.where(u < pH, 0, np.where(u < pH + pD, 1, 3))
    S_ = pH.shape[0]; pts = np.zeros((S_, n))
    for t in range(n):
        pts[:, t] = hpts[:, i == t].sum(1) + apts[:, j == t].sum(1)
    g = np.maximum(np.bincount(i, minlength=n) + np.bincount(j, minlength=n), 1)
    return pts / g + rng.random((S_, n)) * 1e-6


def rates(dr, d, k, r, a, b):
    c = dr["c"][d]; lh = np.exp(c + r * (dr["home"][d] + dr["home_club"][d][a]) + k * (dr["att"][d][a] - dr["def"][d][b]))
    la = np.exp(c + k * (dr["att"][d][b] - dr["def"][d][a])); return lh, la


def game(dr, d, k, r, a, b, legs=1, pens=True):
    """a hosts (or hosts 2nd leg); returns True if a advances."""
    lh, la = rates(dr, d, k, r, a, b)
    if legs == 2:   # first leg at b, second at a; aggregate, then 50/50 (away goals ignored)
        lh2, la2 = rates(dr, d, k, r, b, a)
        ga = rng.poisson(la2) + rng.poisson(lh); gb = rng.poisson(lh2) + rng.poisson(la)
    else:
        ga, gb = rng.poisson(lh), rng.poisson(la)
    return ga > gb if ga != gb else rng.random() < 0.5


def bo3(dr, d, k, r, a, b):
    w = 0
    for gi in range(3):
        h, aw = (a, b) if gi != 1 else (b, a)
        won = game(dr, d, k, r, h, aw)
        w += won if h == a else (not won)
        if w == 2 or (gi - w + 1) == 2:
            break
    return w >= 2


def bracket(dr, d, k, r, order, fmt, po_n):
    """order: teams by table rank. Snake into two conferences; conference brackets; final at higher seed."""
    nteams = po_n
    seeds = order[:nteams]
    conf = [seeds[0::2], seeds[1::2]]
    champs = []
    for cs in conf:
        m = len(cs)
        if fmt in ("a", "b"):                      # knockout round single game, then two-legged semis and final
            if m == 5:  # 2013-14: 4 v 5
                w = cs[3] if game(dr, d, k, r, cs[3], cs[4]) else cs[4]; alive = [cs[0], cs[1], cs[2], w]
            else:       # 2015-18: 3 v 6, 4 v 5
                w1 = cs[2] if game(dr, d, k, r, cs[2], cs[5]) else cs[5]; w2 = cs[3] if game(dr, d, k, r, cs[3], cs[4]) else cs[4]
                alive = [cs[0], cs[1], w1, w2]
            rk = {t: x for x, t in enumerate(cs)}
            alive.sort(key=rk.get)
            s1 = alive[0] if game(dr, d, k, r, alive[0], alive[3], legs=2) else alive[3]
            s2 = alive[1] if game(dr, d, k, r, alive[1], alive[2], legs=2) else alive[2]
            a_, b_ = sorted([s1, s2], key=rk.get)
            champs.append(a_ if game(dr, d, k, r, a_, b_, legs=2) else b_)
        elif fmt == "c":                           # 7 per conference, top seed bye, single games
            rk = {t: x for x, t in enumerate(cs)}
            r1 = [cs[1] if game(dr, d, k, r, cs[1], cs[6]) else cs[6], cs[2] if game(dr, d, k, r, cs[2], cs[5]) else cs[5],
                  cs[3] if game(dr, d, k, r, cs[3], cs[4]) else cs[4]]
            alive = sorted([cs[0]] + r1, key=rk.get)
            s1 = alive[0] if game(dr, d, k, r, alive[0], alive[3]) else alive[3]
            s2 = alive[1] if game(dr, d, k, r, alive[1], alive[2]) else alive[2]
            a_, b_ = sorted([s1, s2], key=rk.get)
            champs.append(a_ if game(dr, d, k, r, a_, b_) else b_)
        else:                                      # 9 per conference: 8v9 wild card, best-of-3 round one, single games after
            rk = {t: x for x, t in enumerate(cs)}
            wc = cs[7] if game(dr, d, k, r, cs[7], cs[8]) else cs[8]
            e = [cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], wc]
            pairs = [(0, 7), (3, 4), (1, 6), (2, 5)]
            r1 = [e[a] if bo3(dr, d, k, r, e[a], e[b]) else e[b] for a, b in pairs]
            q1 = sorted(r1[:2], key=rk.get); q2 = sorted(r1[2:], key=rk.get)
            s1 = q1[0] if game(dr, d, k, r, q1[0], q1[1]) else q1[1]
            s2 = q2[0] if game(dr, d, k, r, q2[0], q2[1]) else q2[1]
            a_, b_ = sorted([s1, s2], key=rk.get)
            champs.append(a_ if game(dr, d, k, r, a_, b_) else b_)
    pos = {t: x for x, t in enumerate(order)}
    a_, b_ = sorted(champs, key=pos.get)
    return a_ if game(dr, d, k, r, a_, b_) else b_


