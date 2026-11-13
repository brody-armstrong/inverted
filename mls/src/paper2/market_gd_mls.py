"""Closing-market expected goal difference for every MLS match -> out/market_gd_mls.pkl.

Recreated 2026-11-13: the original working-session code was not saved. Follows the method recorded in the Paper 2
results log and used for the peer leagues in src/paper2/peer.py:
  closing 1X2 odds (Pinnacle, market average where Pinnacle is missing), bookmaker margin removed
  (both done in src/paper1/random_matches.py -> mkt_H, mkt_D, mkt_A), then inverted through independent Poisson
  scoring rates: find home and away rates whose win and loss probabilities match the market, mkt_gd = rate_home - rate_away.
Matches are keyed by ESPN event id through the ESPN <-> ASA match link (out/espn_asa_link.csv).

Inputs:  out/mls_matches.pkl (src/paper1/random_matches.py), out/espn_asa_link.csv (src/paper2/shot_xg.py)
Output:  out/market_gd_mls.pkl with columns event, game_id, mkt_gd
Usage:   python3 src/paper2/market_gd_mls.py
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

G = np.arange(0, 11)
cache = {}


def solve(ph, pa):
    """Same inversion as peer.py: home and away Poisson rates matching P(home win) and P(away win)."""
    k = (round(ph, 3), round(pa, 3))
    if k not in cache:
        f = lambda x: np.array([np.tril(np.outer(poisson.pmf(G, np.exp(x[0])), poisson.pmf(G, np.exp(x[1]))), -1).sum(),
                                np.triu(np.outer(poisson.pmf(G, np.exp(x[0])), poisson.pmf(G, np.exp(x[1]))), 1).sum()]) - [ph, pa]
        x = least_squares(f, [0.4, 0.1]).x
        cache[k] = np.exp(x[0]) - np.exp(x[1])
    return cache[k]


M = pd.read_pickle("out/mls_matches.pkl").dropna(subset=["mkt_H", "mkt_A"])
M["mkt_gd"] = [solve(h, a) for h, a in zip(M.mkt_H, M.mkt_A)]
link = pd.read_csv("out/espn_asa_link.csv", dtype=str)
out = link.merge(M[["game_id", "mkt_gd"]], on="game_id")[["event", "game_id", "mkt_gd"]]
out.to_pickle("out/market_gd_mls.pkl")
print(f"matches with a market line {len(M)}; linked to ESPN events {len(out)}; mean {out.mkt_gd.mean():.3f}; sd {out.mkt_gd.std():.3f}")
