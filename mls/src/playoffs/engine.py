"""Team ratings and playoff simulation for the MLS Cup Playoffs simulator.

Ratings (log scoring-rate model, same form as the Paper 1 Dixon-Coles model):
  log rate_home = c + home + att[h] - def[a]      log rate_away = c + att[a] - def[h]
  market : fitted to the closing-market scoring rates of every regular-season match (src/playoffs/data.py),
           weighted least squares with an exponential time decay; uncertainty from a match bootstrap.
  results: Paper 1's Bayesian Dixon-Coles model (src/paper1/dc_model.py) fitted to the season's goals; posterior draws.

Playoff rules (2023-present format, unchanged for 2026):
  Wild card (8 v 9): one game at the 8 seed, a draw after 90 minutes goes straight to penalties.
  Round One: best of three, games at the higher seed, the lower seed, then the higher seed; every drawn game goes to penalties.
  Conference semifinals, finals and MLS Cup: one game at the higher seed (MLS Cup: better regular-season record),
  30 minutes of extra time, then penalties.
Shootouts are coin flips (50/50). Extra time scores at one third of the 90-minute rates.
"""
import numpy as np


def fit_market(M, teams, half_life_days=None, end=None, boot=0, rng=None):
    """Weighted least squares on log market rates. Returns dict(c, home, att, def) or a list of them for boot>0."""
    idx = {t: i for i, t in enumerate(teams)}
    D = M.dropna(subset=["mkt_lh"])
    n, m = len(teams), len(D)
    hi, ai = D.home.map(idx).values, D.away.map(idx).values
    y = np.r_[np.log(D.mkt_lh.values), np.log(D.mkt_la.values)]
    X = np.zeros((2 * m, 2 + 2 * n))
    r = np.arange(m)
    X[r, 0] = 1; X[r, 1] = 1; X[r, 2 + hi] = 1; X[r, 2 + n + ai] = -1
    X[m + r, 0] = 1; X[m + r, 2 + ai] = 1; X[m + r, 2 + n + hi] = -1
    # sum-to-zero constraints on att and def as heavily weighted pseudo-rows
    C = np.zeros((2, 2 + 2 * n)); C[0, 2:2 + n] = 1; C[1, 2 + n:] = 1
    if half_life_days:
        age = (end - D.kickoff_utc).dt.total_seconds().values / 86400
        w = 0.5 ** (age / half_life_days)
    else:
        w = np.ones(m)

    def solve(ww):
        W = np.r_[ww, ww]
        A = np.vstack([X * np.sqrt(W)[:, None], C * 1e3]); b = np.r_[y * np.sqrt(W), 0, 0]
        p = np.linalg.lstsq(A, b, rcond=None)[0]
        return {"c": p[0], "home": p[1], "att": p[2:2 + n], "def": p[2 + n:]}

    if not boot:
        return solve(w)
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(boot):
        k = np.bincount(rng.integers(0, m, m), minlength=m)
        out.append(solve(w * k))
    return out


def lam(R, i, j, at_home=True):
    """90-minute scoring rates for team i hosting team j."""
    h = R["home"] if at_home else 0.0
    return np.exp(R["c"] + h + R["att"][i] - R["def"][j]), np.exp(R["c"] + R["att"][j] - R["def"][i])


def probs90(R, i, j, maxg=11):
    """P(home win), P(draw), P(away win) after 90 minutes (independent Poisson)."""
    from scipy.stats import poisson
    lh, la = lam(R, i, j)
    g = np.arange(maxg)
    p = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return np.tril(p, -1).sum(), np.trace(p), np.triu(p, 1).sum()


# ---------- exact tie probabilities (no simulation noise) ----------
def p_game(R, host, guest, extra_time):
    """P(host advances) from one game: 90 minutes, optional extra time, then a 50/50 shootout."""
    pH, pD, pA = probs90(R, host, guest)
    if extra_time:
        R3 = dict(R, c=R["c"] + np.log(1 / 3))      # 30 minutes at a third of the rate
        eH, eD, eA = probs90(R3, host, guest)
        return pH + pD * (eH + eD * 0.5)
    return pH + pD * 0.5


def p_bo3(R, hi, lo):
    """P(higher seed wins a best-of-three; games at hi, lo, hi; drawn games decided by penalties)."""
    g1 = p_game(R, hi, lo, False)
    g2 = 1 - p_game(R, lo, hi, False)
    g3 = g1
    return g1 * g2 + g1 * (1 - g2) * g3 + (1 - g1) * g2 * g3


def dc_draws(R, teams, n=200):
    import sys; sys.path.insert(0, "src/paper1")
    from dc_model import fit_nuts
    idx = {t: i for i, t in enumerate(teams)}
    m = fit_nuts(R.home.map(idx).values, R.away.map(idx).values, len(teams), R.hg.values, R.ag.values)
    s = m.get_samples()
    k = np.linspace(0, len(s["c"]) - 1, n).astype(int)
    out = []
    for q in k:
        a = np.asarray(s["att"][q]); d = np.asarray(s["def"][q])
        out.append({"c": float(s["c"][q]), "home": float(s["home"][q]), "att": a - a.mean(), "def": d - d.mean()})
    # Paper 1's model also has club-specific home effects; the playoff engine uses the league-wide home effect only.
    return out
