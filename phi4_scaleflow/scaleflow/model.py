"""Scale-covariant neural-operator CNF for 2D lattice phi^4 (handout section 3), built on bijx.

Pipeline (for any lattice size L, same parameters):

    xi ~ N(0, 1)^{L x L}
      --FreeTheoryScaling(m0^2 = (mu0/L)^2)-->  phi_0 (free field, covariance 1/(2(p^2+m0^2)))
      --CNF  d phi/dt = v_theta(phi, t)-->       phi_1
      --AnomalousScaling  phi(p) *= A |p|^Delta (p != 0),  B (p = 0)-->  phi

v_theta is odd in phi, translation- and D4-equivariant (radial kernels), and has the form
v_x = tau(delta_x, c_x, t) with c_x independent of phi_x, so its divergence is exact and costs one
forward-mode JVP.
"""

import dataclasses
import typing as tp

import bijx
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .theory import phat2

# ----------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FlowConfig:
    # conditioner mode: "scale" (full model), "untied" (ablation a), "local" (ablation c),
    # "single" (ablation d: one continuous kernel of distance, no scale sharing)
    mode: str = "scale"
    use_delta_layer: bool = True  # ablation b sets False
    embed: int = 4  # E
    hidden: int = 8  # H (scale mixer width; B*N*K*H sets the cost)
    cond: int = 8  # C
    tau_hidden: int = 32
    tau_terms: int = 8  # J
    kernel_hidden: int = 32
    s_min: float = 1.0
    ratio: float = float(np.sqrt(2.0))  # r
    n_uv: int = 1
    n_ir: int = 2
    max_scales: int = 16  # only used by the "untied" ablation (one kernel per absolute index)
    local_size: int = 9  # receptive field of the "local" ablation
    mu0: float = 2.0  # base IR regulator m0 = mu0 / L
    out_scale: float = 0.1  # init scale of the velocity output layer
    # DEVIATION (flagged, see README): feed the global magnetisation M = mean(phi) to tau. c_x stays
    # independent of phi_x; the exact divergence gains the term (1/N) d tau_x / d M, obtained in the
    # same single JVP. Without it v is nearly blind to the zero mode (kernels sum to zero, delta
    # removes the local mean), so it cannot build the double-peaked P(M) at criticality.
    zero_mode_input: bool = True
    # untie the UV end of M_Delta: log s(p) += u(p^2) * q^2 / (1 + q^2), q = p^2 / uv_q. The
    # correction vanishes like p^4 in the IR, so Delta is fitted inside the window only (off = spec)
    mdelta_uv: bool = False
    # scale aggregation. "sum": c = W sum_k h_k (handout 3.3.5). Each h_k has a nonzero mean, so the
    # sum grows with K ~ log L and tau sees out-of-distribution c on held-out L (measured: rms(c)
    # 1.5 -> 8.3 from L=8 to 128). "mean_ends" (DEVIATION, default): mean over the shared middle
    # scales plus separate readouts of the untied UV/IR end scales, which is L-independent.
    pool: str = "mean_ends"
    # how kernels are made to sum to zero. "global" (handout 3.3.3): subtract the mean over all
    # u != 0 of the torus, so R_s = (local s-average of e) - (global average): in the log-correlated
    # free frame its variance grows like log(L/s). "local" (DEVIATION, default): subtract a multiple
    # of the fixed envelope g(u/s) = |z|^2 exp(-|z|^2/2) so that sum_u K = 0 on the scale s itself;
    # R_s is then a band-pass (wavelet) coefficient whose statistics do not depend on L.
    zero_sum: str = "local"
    uv_q: float = 1.0


def scales(L, cfg: FlowConfig):
    """s_k = s_min r^k, k = 0..K-1 with s_max <= L/2 (continuous scale set, any L)."""
    K = int(np.floor(np.log(L / 2 / cfg.s_min) / np.log(cfg.ratio) + 1e-9)) + 1
    return cfg.s_min * cfg.ratio ** np.arange(K)


def min_image_offsets(L):
    u = (np.arange(L) + L // 2) % L - L // 2
    return u[:, None], u[None, :]


# ----------------------------------------------------------------------------------------------
# small building blocks
# ----------------------------------------------------------------------------------------------


class MLP(nnx.Module):
    def __init__(self, sizes, *, rngs, act=nnx.gelu, final_scale=1.0):
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            lin = nnx.Linear(a, b, rngs=rngs)
            if i == len(sizes) - 2 and final_scale != 1.0:
                lin.kernel.set_value(lin.kernel.get_value() * final_scale)
            layers.append(lin)
        self.layers = nnx.List(layers)
        self.act = act

    def __call__(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


class RadialKernelNet(nnx.Module):
    """k_theta(z) = MLP(|z|^2) * |z|^2 exp(-|z|^2 / 2), one output per embedding channel (depthwise).

    The single coordinate network that is re-evaluated at rescaled offsets u/s for every scale."""

    def __init__(self, cfg: FlowConfig, *, rngs, envelope=True, log_input=False):
        self.net = MLP([2, cfg.kernel_hidden, cfg.kernel_hidden, cfg.embed], rngs=rngs, act=nnx.tanh)
        self.envelope = envelope
        self.log_input = log_input

    def __call__(self, z):
        z = z[..., None]
        # |z|^2 (not |z|) keeps the kernel smooth in the offset u, so lattice sums converge fast to
        # the continuum integral and the responses are covariant under L -> 2L, s -> 2s.
        if self.log_input:
            feat = [jnp.log(z + 1e-6), z]
        else:
            feat = [z**2, jnp.exp(-0.5 * z**2)]
        out = self.net(jnp.concatenate(feat, -1))
        if self.envelope:
            # vanishes at z = 0: masking the centre then removes nothing in the continuum, so the
            # mask introduces no O(1/s^2) scale-dependent error
            out = out * z**2 * jnp.exp(-0.5 * z**2)
        return out


def _normalise_kernel(K, g=None):
    """K: (L, L, E). Mask the centre, then make each channel sum to zero over u != 0: globally
    (subtract the mean over the torus) if g is None, else locally by subtracting a * g, g >= 0."""
    L = K.shape[0]
    mask = jnp.ones((L, L)).at[0, 0].set(0.0)[..., None]
    K = K * mask
    if g is None:
        return K - mask * jnp.sum(K, axis=(0, 1), keepdims=True) / (L * L - 1)
    g = (g * mask[..., 0])[..., None]
    return K - g * jnp.sum(K, axis=(0, 1), keepdims=True) / jnp.sum(g)


def fft_conv(e, K):
    """Periodic depthwise correlation R(x) = sum_u K(u) e(x+u).

    e: (B, L, L, E), K: (n, L, L, E) -> (B, L, L, n, E). For radial (inversion-symmetric)
    kernels correlation and convolution coincide."""
    ek = jnp.fft.rfft2(e, axes=(1, 2))  # (B, L, L//2+1, E)
    Kk = jnp.fft.rfft2(K, axes=(1, 2))  # (n, L, L//2+1, E)
    # correlation: conj of kernel spectrum (kernel is real)
    Rk = ek[:, :, :, None, :] * jnp.conj(jnp.moveaxis(Kk, 0, 2))[None]
    return jnp.fft.irfft2(Rk, s=e.shape[1:3], axes=(1, 2))


# ----------------------------------------------------------------------------------------------
# conditioner: multiscale responses + scale mixing (pointwise in x)
# ----------------------------------------------------------------------------------------------


class Conditioner(nnx.Module):
    def __init__(self, cfg: FlowConfig, *, rngs):
        self.cfg = cfg
        E, H, C = cfg.embed, cfg.hidden, cfg.cond
        self.embed = MLP([1, 16, E], rngs=rngs)
        if cfg.mode == "scale":
            self.kernel = RadialKernelNet(cfg, rngs=rngs)
        elif cfg.mode == "single":
            self.kernel = RadialKernelNet(cfg, rngs=rngs, envelope=False, log_input=True)
        elif cfg.mode == "untied":

            @nnx.split_rngs(splits=cfg.max_scales)
            @nnx.vmap(in_axes=(0,), out_axes=0)
            def _make(rngs):
                return RadialKernelNet(cfg, rngs=rngs)

            self.kernel = _make(rngs)
        elif cfg.mode == "local":
            n = cfg.local_size
            self.local_kernel = nnx.Param(0.1 * jax.random.normal(rngs.params(), (n, n, E)))
        else:
            raise ValueError(cfg.mode)
        self.lin_in = nnx.Linear(E, H, rngs=rngs)
        n_emb = cfg.max_scales if cfg.mode == "untied" else cfg.n_uv + cfg.n_ir
        self.scale_emb = nnx.Param(jnp.zeros((max(n_emb, 1), H)))
        # width-3 convolution along the log-scale axis, shared weights: (3H -> H)
        self.kconv = nnx.Linear(3 * H, H, rngs=rngs)
        # final per-(x, k) layer is linear, so it is applied after the pooling over k
        self.lin_out = nnx.Linear(H, C, rngs=rngs)
        if cfg.pool == "mean_ends":
            self.lin_ends = nnx.Param(jax.random.normal(rngs.params(), (cfg.n_uv + cfg.n_ir, H, C)) / np.sqrt(H))
        elif cfg.pool != "sum":
            raise ValueError(cfg.pool)

    # -- kernels ---------------------------------------------------------------------------
    def kernels(self, L):
        """Returns (K_stack (n, L, L, E), scale values or None)."""
        cfg = self.cfg
        u1, u2 = min_image_offsets(L)
        r = jnp.sqrt(jnp.asarray(u1**2 + u2**2, dtype=jnp.result_type(float)))
        if cfg.mode in ("scale", "untied"):
            ss = scales(L, cfg)
            if cfg.mode == "scale":
                Ks = [s**-2 * self.kernel(r / s) for s in ss]
            else:
                assert len(ss) <= cfg.max_scales

                @nnx.vmap(in_axes=(0, 0), out_axes=0)
                def _eval(knet, s):
                    return s**-2 * knet(r / s)

                graph, st = nnx.split(self.kernel)
                st = jax.tree.map(lambda a: a[: len(ss)], st)
                Ks = list(_eval(nnx.merge(graph, st), jnp.asarray(ss, r.dtype)))
            if cfg.zero_sum == "local":
                gs = [(r / s) ** 2 * jnp.exp(-0.5 * (r / s) ** 2) for s in ss]
                return jnp.stack([_normalise_kernel(K, g) for K, g in zip(Ks, gs)]), ss
            return jnp.stack([_normalise_kernel(K) for K in Ks]), ss
        if cfg.mode == "single":
            return _normalise_kernel(self.kernel(r))[None], None
        # local: fixed receptive field, D4-symmetrised learned stencil, embedded in (L, L)
        n = min(cfg.local_size, L if L % 2 else L - 1)  # never wrap onto itself on tiny lattices
        c0 = (cfg.local_size - n) // 2
        w = self.local_kernel.get_value()[c0 : c0 + n, c0 : c0 + n]
        w = sum(jnp.rot90(ww, k, axes=(0, 1)) for ww in (w, w[::-1]) for k in range(4)) / 8
        K = jnp.zeros((L, L, w.shape[-1]), w.dtype)
        idx = (np.arange(n) - n // 2) % L
        K = K.at[idx[:, None], idx[None, :]].set(w)
        return _normalise_kernel(K)[None], None

    def scale_embeddings(self, n_k):
        cfg = self.cfg
        emb = self.scale_emb.get_value()
        out = jnp.zeros((n_k, emb.shape[-1]), emb.dtype)
        if cfg.mode == "untied":
            return emb[:n_k]
        if cfg.mode != "scale":
            return out
        # untied ends: first n_uv indices from the bottom, last n_ir indices counted from the top
        for i in range(min(cfg.n_uv, n_k)):
            out = out.at[i].add(emb[i])
        for j in range(min(cfg.n_ir, n_k)):
            out = out.at[n_k - 1 - j].add(emb[cfg.n_uv + j])
        return out

    def responses(self, phi):
        """R: (B, L, L, K, E) multiscale responses (depthwise masked convolutions)."""
        L = phi.shape[-1]
        e = self.embed(phi[..., None])
        K, _ = self.kernels(L)
        return fft_conv(e, K)

    def __call__(self, phi):
        R = self.responses(phi)  # (B, L, L, K, E)
        B, L, _, n_k, E = R.shape
        # per-(x, k) layers as flat matmuls (much faster than 5-D einsums on CPU)
        h = self.lin_in(R.reshape(-1, E)).reshape(B * L * L, n_k, -1)
        h = nnx.gelu(h + self.scale_embeddings(n_k))
        # 1D convolution along the log-scale axis (width 3, zero padding), residual
        hp = jnp.pad(h, ((0, 0), (1, 1), (0, 0)))
        hcat = jnp.concatenate([hp[:, :-2], hp[:, 1:-1], hp[:, 2:]], axis=-1)
        h = h + nnx.gelu(self.kconv(hcat.reshape(B * L * L * n_k, -1)).reshape(h.shape))
        # pooling over k is pointwise in x: no spatial mixing after the multiscale convolution
        c = self.lin_out(self._pool_mid(h)) + self._ends(h)
        return c.reshape(B, L, L, -1)

    def _ends_index(self, n_k):
        cfg = self.cfg
        if cfg.pool == "sum" or n_k <= 1:
            return [], []
        uv = list(range(min(cfg.n_uv, n_k)))
        ir = [n_k - 1 - j for j in range(cfg.n_ir) if n_k - 1 - j >= len(uv)]
        return uv, ir

    def _pool_mid(self, h):
        n_k = h.shape[1]
        if self.cfg.pool == "sum":
            return jnp.sum(h, axis=1)
        uv, ir = self._ends_index(n_k)
        mid = [k for k in range(n_k) if k not in uv and k not in ir]
        if not mid:
            return jnp.zeros_like(h[:, 0])
        return jnp.mean(h[:, mid[0] : mid[-1] + 1], axis=1)

    def _ends(self, h):
        uv, ir = self._ends_index(h.shape[1])
        if not uv and not ir:
            return 0.0
        W = self.lin_ends.get_value()
        out = 0.0
        for i, k in enumerate(uv):
            out = out + h[:, k] @ W[i]
        for j, k in enumerate(ir):
            out = out + h[:, k] @ W[self.cfg.n_uv + j]
        return out


# ----------------------------------------------------------------------------------------------
# velocity field with exact divergence
# ----------------------------------------------------------------------------------------------


def nn_mean(phi):
    return 0.25 * sum(jnp.roll(phi, s, ax) for ax in (-2, -1) for s in (1, -1))


class Velocity(nnx.Module):
    """bijx-style vector field: (t, phi) -> (dphi/dt, d log q/dt = -div v)."""

    def __init__(self, cfg: FlowConfig, *, rngs):
        self.cfg = cfg
        self.cond = Conditioner(cfg, rngs=rngs)
        self.g = MLP([1 + cfg.cond + (1 if cfg.zero_mode_input else 0), cfg.tau_hidden, cfg.tau_hidden, cfg.tau_terms], rngs=rngs,
                     final_scale=cfg.out_scale)
        self.kappa = MLP([8, 16, cfg.tau_terms], rngs=rngs)

    def _time_feats(self, t):
        k = jnp.arange(1, 5)
        return jnp.concatenate([jnp.sin(np.pi * k * t), jnp.cos(np.pi * k * t)])

    def tau(self, delta, c, t, m=None):
        feats = [delta[..., None], c]
        if m is not None:
            feats.append(jnp.broadcast_to(m[:, None, None, None], delta.shape + (1,)))
        g = self.g(jnp.concatenate(feats, axis=-1))
        kap = self.kappa(self._time_feats(jnp.asarray(t, delta.dtype)))
        return g @ kap

    def v_tilde_and_div(self, phi, t):
        c = self.cond(phi)  # depends on phi, but c_x is independent of phi_x
        delta = phi - nn_mean(phi)  # d delta_x / d phi_x = 1
        ones = jnp.ones_like(delta)
        if self.cfg.zero_mode_input:
            N = phi.shape[-1] * phi.shape[-2]
            m = jnp.mean(phi, axis=(-2, -1))  # d m / d phi_x = 1/N
            out, ddiag = jax.jvp(lambda d, mm: self.tau(d, c, t, mm), (delta, m),
                                 (ones, jnp.full_like(m, 1.0 / N)))
        else:
            out, ddiag = jax.jvp(lambda d: self.tau(d, c, t), (delta,), (ones,))
        return out, jnp.sum(ddiag, axis=(-2, -1))

    def velocity_and_div(self, phi, t):
        """Z2-odd velocity v(phi) = (v~(phi) - v~(-phi))/2 and its exact divergence."""
        B = phi.shape[0]
        both = jnp.concatenate([phi, -phi], axis=0)
        vt, dv = self.v_tilde_and_div(both, t)
        v = 0.5 * (vt[:B] - vt[B:])
        div = 0.5 * (dv[:B] + dv[B:])
        return v, div

    def __call__(self, t, phi):
        v, div = self.velocity_and_div(phi, t)
        return v, -div


# ----------------------------------------------------------------------------------------------
# bijections
# ----------------------------------------------------------------------------------------------


class RK4Flow(bijx.Bijection):
    """Fixed-step RK4 CNF on the augmented state (phi, log q); gradients by direct backprop
    through the solver (discretise-then-optimise). Use bijx.ContFlowDiffrax for adaptive solves."""

    def __init__(self, vf, steps=12, remat=True):
        self.vf = vf
        self.steps = steps
        self.remat = remat

    def _solve(self, x, ld, t0, t1, steps):
        dt = (t1 - t0) / steps

        def f(t, y):
            return self.vf(t, y[0])

        if self.remat:  # store only the inputs of each velocity evaluation for the backward pass
            f = jax.checkpoint(f)

        def step(y, i):
            t = t0 + i * dt
            k1 = f(t, y)
            k2 = f(t + dt / 2, jax.tree.map(lambda a, b: a + dt / 2 * b, y, k1))
            k3 = f(t + dt / 2, jax.tree.map(lambda a, b: a + dt / 2 * b, y, k2))
            k4 = f(t + dt, jax.tree.map(lambda a, b: a + dt * b, y, k3))
            y = jax.tree.map(lambda a, b1, b2, b3, b4: a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4),
                             y, k1, k2, k3, k4)
            return y, None

        (x, ld), _ = jax.lax.scan(step, (x, ld), jnp.arange(steps, dtype=x.dtype))
        return x, ld

    def forward(self, x, log_density, steps=None, **kwargs):
        return self._solve(x, log_density, 0.0, 1.0, steps or self.steps)

    def reverse(self, x, log_density, steps=None, **kwargs):
        return self._solve(x, log_density, 1.0, 0.0, steps or self.steps)


class AnomalousScaling(bijx.ApplyBijection):
    """M_Delta: phi(p) <- A |p|^Delta phi(p) for p != 0, B phi(0) for the zero mode.

    log|det M| = (N-1) log A + log B + Delta sum_{p != 0} log|p|, with |p| the lattice momentum.
    Optional UV-untied correction (uv=True): log s(p) += u(p^2) q^2/(1+q^2), q = p^2/uv_q, with u a
    small MLP; still diagonal in Fourier space, so log|det| = sum_p log s(p) stays closed-form."""

    def __init__(self, delta=0.0, log_a=0.0, log_b=0.0, trainable_delta=True, uv=False, uv_q=1.0,
                 rngs=None):
        V = nnx.Param if trainable_delta else bijx.Const
        self.delta = V(jnp.asarray(delta, jnp.float32))
        self.log_a = nnx.Param(jnp.asarray(log_a, jnp.float32))
        self.log_b = nnx.Param(jnp.asarray(log_b, jnp.float32))
        self.uv_q = uv_q
        self.uv_net = MLP([1, 16, 1], rngs=rngs, act=nnx.tanh, final_scale=0.0) if uv else None

    def log_scale(self, L, dtype):
        p2 = phat2(L, dtype)
        logp = 0.5 * jnp.log(jnp.where(p2 > 0, p2, 1.0))
        d = self.delta.get_value().astype(dtype)
        ls = jnp.where(p2 > 0, self.log_a.get_value().astype(dtype) + d * logp,
                       self.log_b.get_value().astype(dtype))
        if self.uv_net is not None:
            q = p2 / self.uv_q
            ls = ls + self.uv_net(p2[..., None] / 8.0)[..., 0] * q**2 / (1 + q**2)
        return ls

    def log_det(self, L, dtype=jnp.float32):
        return jnp.sum(self.log_scale(L, dtype))

    def apply(self, x, log_density, reverse=False, **kwargs):
        L = x.shape[-1]
        ls = self.log_scale(L, x.dtype)
        ls = -ls if reverse else ls
        y = jnp.fft.ifft2(jnp.fft.fft2(x) * jnp.exp(ls)).real.astype(x.dtype)
        return y, log_density - jnp.sum(ls)


def free_base(L, mu0, dtype=None):
    """White noise + bijx.FreeTheoryScaling: covariance 1/(2(p^2 + m0^2)), m0 = mu0 / L."""
    m0sq = float((mu0 / L) ** 2)
    noise = bijx.IndependentNormal(event_shape=(L, L), rngs=nnx.Rngs(0))
    scale = bijx.FreeTheoryScaling(m0sq, (L, L), finite_size=True, half=False)
    return noise, scale, m0sq


class ScaleFlow(nnx.Module):
    """Full model; parameters are independent of L."""

    def __init__(self, cfg: FlowConfig, *, rngs):
        self.cfg = cfg
        self.velocity = Velocity(cfg, rngs=rngs)
        self.mdelta = AnomalousScaling(trainable_delta=cfg.use_delta_layer,
                                       uv=cfg.mdelta_uv and cfg.use_delta_layer, uv_q=cfg.uv_q, rngs=rngs)
        if not cfg.use_delta_layer:
            self.mdelta.log_a = bijx.Const(self.mdelta.log_a.get_value())
            self.mdelta.log_b = bijx.Const(self.mdelta.log_b.get_value())

    def flow(self, steps=12, remat=True, cnf=None):
        cnf = cnf if cnf is not None else RK4Flow(self.velocity, steps, remat)
        if self.cfg.use_delta_layer:
            return bijx.Chain(cnf, self.mdelta)
        return cnf

    def sample(self, key, batch, L, steps=12, remat=True, dtype=jnp.float32, cnf=None):
        noise, scale, _ = free_base(L, self.cfg.mu0)
        xi = jax.random.normal(key, (batch, L, L), dtype)
        lq = noise.log_density(xi)
        phi0, lq = scale.forward(xi, lq)
        phi0 = phi0.astype(dtype)
        return self.flow(steps, remat, cnf).forward(phi0, lq.astype(dtype))

    def log_density(self, phi, steps=12):
        """log q(phi) by integrating the flow backwards (density evaluation of given configs)."""
        L = phi.shape[-1]
        noise, scale, _ = free_base(L, self.cfg.mu0)
        # bijx convention: reverse(y, 0) returns +log|det J_forward|
        z, dl = self.flow(steps, remat=False).reverse(phi, jnp.zeros(phi.shape[0], phi.dtype))
        xi, dl = scale.reverse(z, dl)
        return noise.log_density(xi) - dl


def cast_state(model, dtype):
    graph, state = nnx.split(model)
    state = jax.tree.map(lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, state)
    return nnx.merge(graph, state)
