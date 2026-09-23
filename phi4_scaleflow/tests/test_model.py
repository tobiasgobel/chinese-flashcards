"""Unit tests of handout section 8 (run in float64)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaleflow.model import (AnomalousScaling, Conditioner, FlowConfig, RK4Flow, ScaleFlow,
                             Velocity, cast_state, fft_conv, free_base, scales)
from scaleflow.theory import free_action, phat2

MODES = ["scale", "untied", "local", "single"]


def make_velocity(mode="scale", seed=0, out_scale=1.0):
    v = Velocity(FlowConfig(mode=mode, out_scale=out_scale), rngs=nnx.Rngs(seed))
    return cast_state(v, jnp.float64)


def rand_phi(L, B=2, seed=1):
    return jax.random.normal(jax.random.PRNGKey(seed), (B, L, L), jnp.float64)


# 1. cheap divergence == brute-force Jacobian trace --------------------------------------------
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("L", [4, 6])
def test_divergence_matches_bruteforce(mode, L):
    vf = make_velocity(mode)
    phi = rand_phi(L)
    t = 0.37
    _, div = vf.velocity_and_div(phi, t)

    def v_single(p):
        return vf.velocity_and_div(p[None], t)[0][0].reshape(-1)

    for b in range(phi.shape[0]):
        J = jax.jacfwd(v_single)(phi[b])
        tr = jnp.trace(J.reshape(L * L, L * L))
        np.testing.assert_allclose(div[b], tr, rtol=1e-10, atol=1e-10)


# 2. integrated divergence == log|det| of the discrete solver map --------------------------------
def test_cnf_logdensity_matches_solver_jacobian():
    L = 4
    vf = make_velocity("scale", out_scale=2.0)
    phi0 = 1.5 * rand_phi(L, B=1)[0]

    def run(steps):
        flow = RK4Flow(vf, steps=steps, remat=False)
        f = lambda p: flow.forward(p[None], jnp.zeros(1))[0][0].reshape(-1)
        J = jax.jacfwd(f)(phi0).reshape(L * L, L * L)
        _, logdet = jnp.linalg.slogdet(J)
        y, ell = flow.forward(phi0[None], jnp.zeros(1))
        return float(logdet), float(-ell[0]), float(jnp.abs(y - phi0).max())

    logdet, neg_ell, moved = run(64)
    assert moved > 0.05, "flow is (nearly) identity; test would be vacuous"
    assert abs(logdet - neg_ell) < 1e-6, (logdet, neg_ell)
    # coarse solver disagrees more: the tolerance is a genuine solver-error bound
    l8, n8, _ = run(4)
    assert abs(l8 - n8) >= abs(logdet - neg_ell)


# 3. base distribution ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [6, 8])
def test_base_logq_matches_gaussian_formula(L):
    mu0 = 2.0
    noise, scale, m0sq = free_base(L, mu0)
    xi = jax.random.normal(jax.random.PRNGKey(0), (5, L, L), jnp.float64)
    phi0, lq = scale.forward(xi, noise.log_density(xi))
    N = L * L
    ref = -free_action(phi0, m0sq) + 0.5 * jnp.sum(jnp.log(phat2(L, jnp.float64) + m0sq)) - 0.5 * N * np.log(np.pi)
    np.testing.assert_allclose(lq, ref, rtol=1e-10)


def test_base_covariance_matches_C():
    L, n = 8, 40000
    noise, scale, m0sq = free_base(L, 2.0)
    xi = jax.random.normal(jax.random.PRNGKey(1), (n, L, L), jnp.float64)
    phi0, _ = scale.forward(xi, jnp.zeros(n))
    pk = jnp.fft.fft2(phi0, norm="ortho")
    emp = jnp.mean(jnp.abs(pk) ** 2, axis=0)
    C = 1.0 / (2.0 * (phat2(L, jnp.float64) + m0sq))
    np.testing.assert_allclose(emp, C, rtol=0.05)


# 4. M layer -------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [4, 5])
def test_mdelta_logdet_bruteforce(L):
    m = cast_state(AnomalousScaling(delta=0.3, log_a=0.1, log_b=-0.2), jnp.float64)
    x = rand_phi(L, B=1)[0]
    J = jax.jacfwd(lambda p: m.forward(p[None], jnp.zeros(1))[0][0].reshape(-1))(x)
    _, ld = jnp.linalg.slogdet(J.reshape(L * L, L * L))
    np.testing.assert_allclose(m.log_det(L, jnp.float64), ld, rtol=1e-10)
    p2 = phat2(L, jnp.float64)
    la, lb, dl = (m.log_a.get_value(), m.log_b.get_value(), m.delta.get_value())
    closed = (L * L - 1) * la + lb + dl * jnp.sum(jnp.where(p2 > 0, 0.5 * jnp.log(jnp.where(p2 > 0, p2, 1)), 0))
    np.testing.assert_allclose(closed, ld, rtol=1e-10)
    y, l1 = m.forward(x[None], jnp.zeros(1))
    xr, l2 = m.reverse(y, l1)
    np.testing.assert_allclose(xr[0], x, atol=1e-12)
    np.testing.assert_allclose(l2, 0.0, atol=1e-10)


# 5. symmetries ----------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_symmetries(mode):
    L = 8
    vf = make_velocity(mode)
    phi = rand_phi(L, B=2)
    t = 0.6
    v, d = vf.velocity_and_div(phi, t)
    vm, dm = vf.velocity_and_div(-phi, t)
    np.testing.assert_allclose(vm, -v, atol=1e-12)
    np.testing.assert_allclose(dm, d, atol=1e-10)
    # translations
    sh = lambda a: jnp.roll(a, (3, -2), axis=(-2, -1))
    np.testing.assert_allclose(vf.velocity_and_div(sh(phi), t)[0], sh(v), atol=1e-10)
    # D4 (radial kernels / symmetrised stencil)
    for g in (lambda a: jnp.rot90(a, 1, axes=(-2, -1)), lambda a: jnp.swapaxes(a, -1, -2),
              lambda a: a[..., ::-1, :]):
        np.testing.assert_allclose(vf.velocity_and_div(g(phi), t)[0], g(v), atol=1e-9)
        np.testing.assert_allclose(vf.velocity_and_div(g(phi), t)[1], d, atol=1e-9)


# 6. conditioner independence --------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_conditioner_independent_of_own_site(mode):
    L = 6
    vf = make_velocity(mode)
    phi = rand_phi(L, B=1)[0]
    J = jax.jacrev(lambda p: vf.cond(p[None])[0])(phi)  # (L, L, C, L, L)
    diag = J[np.arange(L)[:, None], np.arange(L)[None, :], :, np.arange(L)[:, None], np.arange(L)[None, :]]
    assert float(jnp.abs(diag).max()) < 1e-12
    assert float(jnp.abs(J).max()) > 1e-4  # and it does depend on the other sites


# 7. scale covariance of the multiscale responses -----------------------------------------------
def smooth_field(L):
    y = np.arange(L) / L
    Y1, Y2 = np.meshgrid(y, y, indexing="ij")
    f = (np.cos(2 * np.pi * Y1) + 0.6 * np.sin(2 * np.pi * (Y1 + 2 * Y2)) + 0.4 * np.cos(2 * np.pi * 2 * Y2)
         + 0.3 * np.sin(2 * np.pi * (3 * Y1 - Y2)))
    return jnp.asarray(f)[None]


def test_scale_covariance_of_responses():
    cfg = FlowConfig(mode="scale")
    cond = cast_state(Conditioner(cfg, rngs=nnx.Rngs(3)), jnp.float64)
    L = 32
    R1 = cond.responses(smooth_field(L))[0]  # (L, L, K, E), scales s_k
    R2 = cond.responses(smooth_field(2 * L))[0]  # scales s_k on 2L; index k+2 is 2 s_k
    s1, s2 = scales(L, cfg), scales(2 * L, cfg)
    errs = []
    for k in range(len(s1)):
        assert np.isclose(s2[k + 2], 2 * s1[k])
        a = R1[:, :, k]
        b = R2[::2, ::2, k + 2]
        errs.append(float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a)))
    errs = np.array(errs)
    # inside the window (s >= 2) the responses agree up to discretisation error ...
    inside = (s1 >= 2) & (s1 <= L / 4)
    assert errs[inside].max() < 0.01, errs
    # even at the lattice cutoff (s = 1) the mismatch is only a few percent, because
    # k(0) = 0 makes the centre mask harmless and k is smooth in u
    assert errs.max() < 0.03, errs


def test_model_transfers_across_sizes_and_log_density_roundtrip():
    cfg = FlowConfig(mode="scale", out_scale=1.0)
    model = cast_state(ScaleFlow(cfg, rngs=nnx.Rngs(0)), jnp.float64)
    model.mdelta.delta.set_value(jnp.asarray(0.2))
    for L in (6, 9, 12):  # includes an odd size
        phi, lq = model.sample(jax.random.PRNGKey(L), 3, L, steps=40, remat=False, dtype=jnp.float64)
        lq2 = model.log_density(phi, steps=40)
        np.testing.assert_allclose(lq, lq2, rtol=1e-6, atol=1e-5)
