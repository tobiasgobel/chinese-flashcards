"""Lattice phi^4 action, lattice momenta and observables.

Convention (Albergo et al. 2019; Gerdes et al. 2023; Mate & Fleuret 2024):
    S[phi] = sum_x [ sum_mu (phi_{x+mu} - phi_x)^2 + m2 phi_x^2 + lam phi_x^4 ]
"""

import jax
import jax.numpy as jnp
import numpy as np
from bijx.lattice.scalar import phi4_term

ISING = dict(nu=1.0, eta=0.25, delta_sigma=0.125, gamma=1.75, binder_star=0.61069)


def action(phi, m2, lam):
    """phi: (..., L, L) -> (...)."""
    return jnp.sum(_batched_phi4(phi, m2, lam), axis=(-2, -1))


def _batched_phi4(phi, m2, lam):
    kin = sum((jnp.roll(phi, 1, ax) - phi) ** 2 for ax in (-2, -1))
    p2 = phi**2
    return kin + m2 * p2 + lam * p2**2


def free_action(phi, m0sq):
    return action(phi, m0sq, 0.0)


def phat2(L, dtype=jnp.float32):
    """Lattice momentum squared p^2 = sum_mu 4 sin^2(p_mu/2) on the full (L, L) FFT grid."""
    p = 2 * np.pi * np.fft.fftfreq(L, d=1.0 / L) / L
    s = 4 * np.sin(p / 2) ** 2
    return jnp.asarray(s[:, None] + s[None, :], dtype=dtype)


def magnetization(phi):
    return jnp.mean(phi, axis=(-2, -1))


def two_point_axis(phi):
    """G(r) = <phi_x phi_{x+r e_mu}> averaged over x and both axes, per configuration.

    Returns (..., L//2 + 1)."""
    L = phi.shape[-1]
    fk = jnp.fft.fft2(phi)
    corr = jnp.fft.ifft2(fk * jnp.conj(fk)).real / (L * L)  # full connected-on-torus correlator
    g = 0.5 * (corr[..., 0, :] + corr[..., :, 0])
    return g[..., : L // 2 + 1]


def _check_consistency():  # pragma: no cover - used in tests
    phi = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    return jnp.allclose(action(phi, -4.0, 5.0), jnp.sum(phi4_term(phi, -4.0, 5.0)))
