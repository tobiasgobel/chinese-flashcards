"""Vectorised HMC (leapfrog) for lattice phi^4 with many independent chains.

Measurements per trajectory: M, and the axial two-point function G(r)."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from .theory import action, magnetization, two_point_axis


def _force(phi, m2, lam):
    # dS/dphi, analytic: 2 * (4 phi - sum_nn phi) + 2 m2 phi + 4 lam phi^3
    nn = sum(jnp.roll(phi, s, ax) for ax in (-2, -1) for s in (1, -1))
    return 2.0 * (4.0 * phi - nn) + 2.0 * m2 * phi + 4.0 * lam * phi**3


@functools.partial(jax.jit, static_argnames=("n_leap",))
def hmc_step(key, phi, m2, lam, eps, n_leap):
    k1, k2 = jax.random.split(key)
    p0 = jax.random.normal(k1, phi.shape, phi.dtype)
    h0 = action(phi, m2, lam) + 0.5 * jnp.sum(p0**2, axis=(-2, -1))

    p = p0 - 0.5 * eps * _force(phi, m2, lam)

    def body(i, carry):
        q, p = carry
        q = q + eps * p
        p = p - eps * _force(q, m2, lam)
        return q, p

    q, p = jax.lax.fori_loop(0, n_leap - 1, body, (phi, p))
    q = q + eps * p
    p = p - 0.5 * eps * _force(q, m2, lam)
    h1 = action(q, m2, lam) + 0.5 * jnp.sum(p**2, axis=(-2, -1))
    dh = h1 - h0
    acc = jnp.log(jax.random.uniform(k2, dh.shape)) < -dh
    phi_new = jnp.where(acc[:, None, None], q, phi)
    return phi_new, acc, dh


@functools.partial(jax.jit, static_argnames=("n_leap", "n_traj"))
def _run_block(key, phi, m2, lam, eps, n_leap, n_traj):
    def body(carry, k):
        phi = carry
        phi, acc, dh = hmc_step(k, phi, m2, lam, eps, n_leap)
        # Z2 flip (exact symmetry of the action: always accepted) is NOT applied,
        # so that autocorrelation times reflect plain HMC.
        return phi, (magnetization(phi), acc, two_point_axis(phi).mean(axis=0))

    keys = jax.random.split(key, n_traj)
    phi, (m, acc, g) = jax.lax.scan(body, phi, keys)
    return phi, m, acc, g


def run_hmc(L, m2, lam, *, n_chains=64, n_traj=4000, n_therm=500, traj_len=1.0,
            eps=None, target_acc=(0.7, 0.9), seed=0, block=250, verbose=True,
            dtype=jnp.float32):
    """Returns dict with M (n_traj, n_chains), acceptance, G(r) (chain-averaged per trajectory),
    last configs and timing."""
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    phi = 0.5 * jax.random.normal(k0, (n_chains, L, L), dtype)
    if eps is None:
        eps = 0.3 / np.sqrt(np.sqrt(L))  # crude start, tuned below
    eps = float(eps)

    # thermalise + tune step size for acceptance in target window
    done = 0
    while done < n_therm:
        n_leap = max(1, int(round(traj_len / eps)))
        nb = min(50, n_therm - done)
        key, kb = jax.random.split(key)
        phi, _, acc, _ = _run_block(kb, phi, m2, lam, eps, n_leap, nb)
        a = float(np.mean(acc))
        if a < target_acc[0]:
            eps *= 0.8
        elif a > target_acc[1]:
            eps *= 1.15
        done += nb
    n_leap = max(1, int(round(traj_len / eps)))

    Ms, accs, Gs = [], [], []
    t0 = time.time()
    done = 0
    while done < n_traj:
        nb = min(block, n_traj - done)
        key, kb = jax.random.split(key)
        phi, m, acc, g = _run_block(kb, phi, m2, lam, eps, n_leap, nb)
        Ms.append(np.asarray(m)); accs.append(np.asarray(acc)); Gs.append(np.asarray(g))
        done += nb
    jax.block_until_ready(phi)
    dt = time.time() - t0
    M = np.concatenate(Ms).astype(np.float64)
    acc = np.concatenate(accs)
    G = np.concatenate(Gs).astype(np.float64)
    if verbose:
        print(f"HMC L={L} lam={lam}: eps={eps:.4f} n_leap={n_leap} acc={acc.mean():.3f} "
              f"time={dt:.1f}s", flush=True)
    return dict(M=M, acc=acc, G=G, phi=np.asarray(phi), eps=eps, n_leap=n_leap,
                seconds=dt, n_chains=n_chains, n_traj=n_traj)
