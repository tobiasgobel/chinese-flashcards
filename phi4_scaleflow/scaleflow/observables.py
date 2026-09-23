"""Observables from magnetisation samples, with jackknife errors (HMC) or weights (flow)."""

import numpy as np

from .stats import jackknife, tau_int


def mc_observables(M, L, G=None, n_blocks=20):
    """M: (n_traj, n_chains) Markov-chain magnetisations. Blocks along the trajectory axis."""
    N = L * L
    M = np.asarray(M, np.float64)
    absM, M2, M4 = np.abs(M), M**2, M**4
    # chain-average per trajectory, then jackknife over trajectory blocks
    series = [x.mean(axis=1) for x in (M, absM, M2, M4)]
    out = {}
    out["M"] = jackknife(lambda a: a.mean(), (series[0],), n_blocks)
    out["absM"] = jackknife(lambda a: a.mean(), (series[1],), n_blocks)
    out["chi"] = jackknife(lambda a: N * a.mean(), (series[2],), n_blocks)
    out["U"] = jackknife(lambda a, b: 1 - b.mean() / (3 * a.mean() ** 2), (series[2], series[3]), n_blocks)
    out["tau_M2"] = tau_int(M2.T)
    out["tau_absM"] = tau_int(absM.T)
    if G is not None:
        out["G"] = jackknife(lambda g: g.mean(axis=0), (np.asarray(G),), n_blocks)
    return out


def weighted_observables(M, logw, L, G=None, n_blocks=20):
    """Reweighted estimates from independent flow samples with log-weights logw."""
    N = L * L
    M = np.asarray(M, np.float64)
    logw = np.asarray(logw, np.float64)
    w = np.exp(logw - logw.max())

    def wm(o, w):
        return np.sum(w * o, axis=0) / np.sum(w)

    out = {}
    out["M"] = jackknife(lambda m, w: wm(m, w), (M, w), n_blocks)
    out["absM"] = jackknife(lambda m, w: wm(np.abs(m), w), (M, w), n_blocks)
    out["chi"] = jackknife(lambda m, w: N * wm(m**2, w), (M, w), n_blocks)
    out["U"] = jackknife(lambda m, w: 1 - wm(m**4, w) / (3 * wm(m**2, w) ** 2), (M, w), n_blocks)
    if G is not None:
        G = np.asarray(G, np.float64)
        out["G"] = jackknife(lambda g, w: np.sum(w[:, None] * g, 0) / np.sum(w), (G, w), n_blocks)
    return out


def ess(logw):
    logw = np.asarray(logw, np.float64)
    w = np.exp(logw - logw.max())
    return w.sum() ** 2 / (len(w) * np.sum(w**2))
