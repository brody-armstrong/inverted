"""Bayesian Dixon-Coles team-strength model (NumPyro / NUTS).

log mu_home = c + home + home_club[h] + att[h] - def[a]
log mu_away = c + att[a] - def[h]
att ~ N(0, s_att), def ~ N(0, s_def), home_club ~ N(0, s_hc); Dixon-Coles low-score correction rho.
Optional match weights (time decay) multiply the log-likelihood.
"""
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from jax.scipy.special import gammaln

numpyro.set_host_device_count(2)


def dc_logtau(x, y, lh, la, rho):
    t = jnp.where((x == 0) & (y == 0), 1 - lh * la * rho,
        jnp.where((x == 0) & (y == 1), 1 + lh * rho,
        jnp.where((x == 1) & (y == 0), 1 + la * rho,
        jnp.where((x == 1) & (y == 1), 1 - rho, 1.0))))
    return jnp.log(jnp.clip(t, 1e-6, None))


def model(h, a, n_teams, hg=None, ag=None, w=None, neutral=None):
    c = numpyro.sample("c", dist.Normal(0.3, 0.5))
    home = numpyro.sample("home", dist.Normal(0.25, 0.25))
    s_att = numpyro.sample("s_att", dist.HalfNormal(0.5))
    s_def = numpyro.sample("s_def", dist.HalfNormal(0.5))
    s_hc = numpyro.sample("s_hc", dist.HalfNormal(0.2))
    with numpyro.plate("teams", n_teams):   # non-centred: avoids the funnel between scales and effects
        z_att = numpyro.sample("z_att", dist.Normal(0, 1))
        z_def = numpyro.sample("z_def", dist.Normal(0, 1))
        z_hc = numpyro.sample("z_hc", dist.Normal(0, 1))
    att = numpyro.deterministic("att", s_att * z_att)
    dfn = numpyro.deterministic("def", s_def * z_def)
    hc = numpyro.deterministic("home_club", s_hc * z_hc)
    rho = numpyro.sample("rho", dist.Uniform(-0.25, 0.25))
    att = att - att.mean(); dfn = dfn - dfn.mean()
    is_home = 1.0 if neutral is None else (1.0 - neutral)
    lh = jnp.exp(c + (home + hc[h]) * is_home + att[h] - dfn[a])
    la = jnp.exp(c + att[a] - dfn[h])
    if hg is not None:
        ll = (hg * jnp.log(lh) - lh - gammaln(hg + 1) + ag * jnp.log(la) - la - gammaln(ag + 1)
              + dc_logtau(hg, ag, lh, la, rho))
        if w is not None:
            ll = ll * w
        numpyro.factor("ll", ll.sum())


def fit_nuts(h, a, n, hg, ag, w=None, neutral=None, warmup=500, samples=500, chains=2, seed=0):
    k = NUTS(model, target_accept_prob=0.85)
    m = MCMC(k, num_warmup=warmup, num_samples=samples, num_chains=chains, progress_bar=False, chain_method="parallel")
    m.run(jax.random.PRNGKey(seed), jnp.array(h), jnp.array(a), n, jnp.array(hg, dtype=float), jnp.array(ag, dtype=float),
          None if w is None else jnp.array(w), None if neutral is None else jnp.array(neutral, dtype=float))
    return m


def fit_svi(h, a, n, hg, ag, w=None, neutral=None, steps=3000, seed=0):
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.03), Trace_ELBO())
    r = svi.run(jax.random.PRNGKey(seed), steps, jnp.array(h), jnp.array(a), n, jnp.array(hg, dtype=float),
                jnp.array(ag, dtype=float), None if w is None else jnp.array(w),
                None if neutral is None else jnp.array(neutral, dtype=float), progress_bar=False)
    return guide, r.params


def outcome_probs(lh, la, rho, maxg=10):
    """P(home win, draw, away win) from Dixon-Coles; lh, la arrays (any shape)."""
    from scipy.stats import poisson
    g = np.arange(maxg + 1)
    ph = poisson.pmf(g[None, :], np.asarray(lh).reshape(-1, 1))
    pa = poisson.pmf(g[None, :], np.asarray(la).reshape(-1, 1))
    M = ph[:, :, None] * pa[:, None, :]
    lhf, laf, r = np.asarray(lh).reshape(-1), np.asarray(la).reshape(-1), np.broadcast_to(np.asarray(rho).reshape(-1), lhf.shape if False else np.asarray(lh).reshape(-1).shape)
    M[:, 0, 0] *= 1 - lhf * laf * r
    M[:, 0, 1] *= 1 + lhf * r
    M[:, 1, 0] *= 1 + laf * r
    M[:, 1, 1] *= 1 - r
    M /= M.sum(axis=(1, 2), keepdims=True)
    hw = np.tril(np.ones((maxg + 1, maxg + 1)), -1)
    pH = (M * hw).sum(axis=(1, 2)); pD = np.trace(M, axis1=1, axis2=2); pA = 1 - pH - pD
    return pH, pD, pA
