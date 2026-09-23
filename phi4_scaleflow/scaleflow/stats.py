"""Error analysis: jackknife, binning and integrated autocorrelation times."""

import numpy as np


def tau_int(x, c=6.0):
    """Integrated autocorrelation time with Sokal's automatic window.

    x: (T,) or (chains, T). Returns tau_int (in units of the sampling step)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    x = x - x.mean(axis=1, keepdims=True)
    T = x.shape[1]
    n = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(x, n=n, axis=1)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=1)[:, :T].mean(axis=0)
    if acf[0] <= 0:
        return 0.5
    rho = acf / acf[0]
    tau = 0.5
    for W in range(1, T):
        tau += rho[W]
        if W >= c * tau:
            break
    return max(tau, 0.5)


def jackknife(fn, data, n_blocks=20):
    """Block jackknife of fn(*arrays) where each array has samples on axis 0.

    data: tuple of arrays. Returns (estimate, error)."""
    n = data[0].shape[0]
    n_blocks = min(n_blocks, n)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    full = fn(*data)
    reps = []
    for i in range(n_blocks):
        keep = np.r_[0 : edges[i], edges[i + 1] : n]
        reps.append(fn(*(d[keep] for d in data)))
    reps = np.asarray(reps)
    err = np.sqrt((n_blocks - 1) * np.mean((reps - reps.mean(axis=0)) ** 2, axis=0))
    return full, err


def binder(m2, m4):
    return 1.0 - np.mean(m4) / (3.0 * np.mean(m2) ** 2)


def weighted_mean(o, w):
    return np.sum(w * o, axis=0) / np.sum(w)
