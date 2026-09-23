"""Reverse-KL training of the flow on several lattice sizes (float32, RK4, backprop through solver)."""

import argparse
import dataclasses
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .model import FlowConfig, ScaleFlow
from .theory import action

DEFAULT_TRAIN = dict(
    m2=-4.0, lam=4.25, train_Ls=[8, 16, 32], batch=64, batch_by_L={"8": 64, "16": 32, "32": 16}, lr=1e-3, steps=2000, ode_steps=8,
    clip=1.0, seed=0, log_every=25,
    # curriculum: list of [fraction_of_steps_at_which_phase_starts, allowed Ls]
    curriculum=[[0.0, [8]], [0.1, [8, 16]], [0.3, [8, 16, 32]]],
    # relative sampling weights of L inside a phase (larger L is costlier per step)
    L_weights={"8": 1.0, "16": 1.0, "32": 1.0},
)


def load_config(path):
    with open(path) as f:
        c = json.load(f)
    flow = FlowConfig(**c.get("flow", {}))
    train = dict(DEFAULT_TRAIN)
    train.update(c.get("train", {}))
    return flow, train


def save_checkpoint(path, model, flow_cfg, train_cfg, history):
    state = nnx.state(model, nnx.Param)
    pure = jax.tree.map(np.asarray, state.to_pure_dict())
    with open(path, "wb") as f:
        pickle.dump(dict(params=pure, flow=dataclasses.asdict(flow_cfg), train=train_cfg,
                         history=history), f)


def load_checkpoint(path, dtype=jnp.float32):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    ck["flow"].setdefault("zero_mode_input", False)  # checkpoints from before the option existed
    cfg = FlowConfig(**ck["flow"])
    model = ScaleFlow(cfg, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(state, jax.tree.map(jnp.asarray, ck["params"]))
    nnx.update(model, state)
    if dtype != jnp.float32:
        from .model import cast_state
        model = cast_state(model, dtype)
    return model, ck


def batch_for(tc, L):
    return int(tc.get("batch_by_L", {}).get(str(L), tc["batch"]))


def make_step(graphdef, rest, opt, tc):
    m2, lam, ode_steps = tc["m2"], tc["lam"], tc["ode_steps"]

    def loss_fn(params, key, L):
        B = batch_for(tc, L)
        model = nnx.merge(graphdef, params, rest)
        phi, lq = model.sample(key, B, L, steps=ode_steps)
        S = action(phi, m2, lam)
        logw = -S - lq
        N = L * L
        return jnp.mean(lq + S) / N, (logw, model.mdelta.delta.get_value(), jnp.mean(phi, (-2, -1)))

    @jax.jit
    def _step(params, opt_state, key, L_dummy):
        L = L_dummy.shape[0]
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, key, L)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    return _step


def train(flow_cfg, tc, outdir, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    model = ScaleFlow(flow_cfg, rngs=nnx.Rngs(tc["seed"]))
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    sched = optax.cosine_decay_schedule(tc["lr"], tc["steps"], alpha=0.02)
    opt = optax.chain(optax.clip_by_global_norm(tc["clip"]), optax.adam(sched))
    opt_state = opt.init(params)
    step_fn = make_step(graphdef, rest, opt, tc)
    rng = np.random.default_rng(tc["seed"])
    key = jax.random.PRNGKey(tc["seed"] + 12345)
    history, t0 = [], time.time()
    per_L = {}
    for it in range(tc["steps"]):
        frac = it / tc["steps"]
        allowed = [p[1] for p in tc["curriculum"] if p[0] <= frac][-1]
        w = np.array([tc["L_weights"].get(str(L), 1.0) for L in allowed], float)
        L = int(rng.choice(allowed, p=w / w.sum()))
        key, k = jax.random.split(key)
        params, opt_state, loss, (logw, delta, M) = step_fn(params, opt_state, k, jnp.zeros(L))
        logw = np.asarray(logw, np.float64)
        d = per_L.setdefault(L, [])
        d.append((float(loss), float(np.var(logw) / (L * L)), float(_ess(logw)), float(np.mean(M))))
        if (it + 1) % tc["log_every"] == 0 or it == tc["steps"] - 1:
            rec = dict(step=it + 1, time=time.time() - t0, delta=float(delta))
            for LL, v in per_L.items():
                v = np.array(v)
                rec[f"L{LL}"] = dict(loss=v[:, 0].mean(), varlogw_N=v[:, 1].mean(), ess=v[:, 2].mean(),
                                     M=v[:, 3].mean(), n=len(v))
            per_L = {}
            history.append(rec)
            if verbose:
                s = " ".join(f"L{LL}: F={r['loss']:.4f} var/N={r['varlogw_N']:.4f} ess={r['ess']:.2f}"
                             for LL, r in ((k2[1:], v2) for k2, v2 in rec.items() if k2.startswith("L")))
                print(f"[{it+1:5d} {rec['time']:7.0f}s] Delta={rec['delta']:+.4f} {s}", flush=True)
            if not np.isfinite(float(loss)):
                raise FloatingPointError("non-finite loss")
    nnx.update(model, params)
    save_checkpoint(os.path.join(outdir, "model.pkl"), model, flow_cfg, tc, history)
    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


def _ess(logw):
    w = np.exp(logw - logw.max())
    return w.sum() ** 2 / (len(w) * np.sum(w**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    a = ap.parse_args()
    flow_cfg, tc = load_config(a.config)
    for k in ("seed", "steps", "lam"):
        if getattr(a, k) is not None:
            tc[k] = getattr(a, k)
    print("flow:", flow_cfg, "\ntrain:", tc, flush=True)
    train(flow_cfg, tc, a.out)


if __name__ == "__main__":
    main()
