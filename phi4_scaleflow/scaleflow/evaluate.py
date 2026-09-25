"""Float64 evaluation: exact log-weights, per-site metrics, reweighted observables, timing.

The ODE step count is doubled until log q changes by less than `tol` per configuration
(max over a probe batch), so the sampling map and the integrated divergence agree."""

import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .observables import ess, weighted_observables  # noqa: E402
from .theory import action, magnetization, two_point_axis  # noqa: E402
from .train import load_checkpoint  # noqa: E402


def make_sampler(model, L, batch, steps):
    @jax.jit
    def f(key):
        phi, lq = model.sample(key, batch, L, steps=steps, remat=False, dtype=jnp.float64)
        return phi, lq, magnetization(phi), two_point_axis(phi)

    return f


def converged_steps(model, L, batch, steps0=8, tol=1e-3, max_steps=256, key=None):
    key = jax.random.PRNGKey(99) if key is None else key
    steps = steps0
    _, lq, _, _ = make_sampler(model, L, batch, steps)(key)
    history = []
    while steps < max_steps:
        _, lq2, _, _ = make_sampler(model, L, batch, 2 * steps)(key)
        diff = float(jnp.max(jnp.abs(lq2 - lq)))
        history.append((steps, diff))
        if diff < tol:
            return steps, history
        steps, lq = 2 * steps, lq2
    return steps, history


def evaluate(model, L, m2, lam, n_samples, batch, seed=0, steps=None, tol=1e-3):
    if steps is None:
        steps, conv = converged_steps(model, L, min(batch, 16), tol=tol)
    else:
        conv = []
    sampler = make_sampler(model, L, batch, steps)
    key = jax.random.PRNGKey(seed)
    jax.block_until_ready(sampler(key))  # compile
    t0 = time.time()
    Ms, logws, Gs, Ss = [], [], [], []
    for i in range(int(np.ceil(n_samples / batch))):
        key, k = jax.random.split(key)
        phi, lq, M, G = sampler(k)
        S = action(phi, m2, lam)
        logws.append(np.asarray(-S - lq))
        Ms.append(np.asarray(M)); Gs.append(np.asarray(G)); Ss.append(np.asarray(S))
    seconds = time.time() - t0
    logw = np.concatenate(logws)[:n_samples]
    M = np.concatenate(Ms)[:n_samples]
    G = np.concatenate(Gs)[:n_samples]
    N = L * L
    e = ess(logw)
    obs = weighted_observables(M, logw, L, G=G)
    # unweighted (raw model) observables, for diagnosing what reweighting has to fix
    raw = weighted_observables(M, np.zeros_like(logw), L, G=G)
    fin = np.isfinite(logw)
    # bootstrap error of Var(log w)/N
    rng = np.random.default_rng(0)
    boots = [np.var(logw[rng.integers(0, len(logw), len(logw))]) / N for _ in range(200)]
    res = dict(
        L=L, n=int(n_samples), ode_steps=int(steps), convergence=conv,
        varlogw_N=float(np.var(logw) / N), varlogw_N_err=float(np.std(boots)),
        kl_per_site=float(-np.mean(logw) / N),  # E_q[log q + S]/N = F_q/N (up to log Z)
        ess=float(e), seconds=seconds, sec_per_sample=seconds / n_samples,
        sec_per_eff_sample=seconds / max(n_samples * e, 1e-300), finite_frac=float(fin.mean()),
        obs={k: [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()] for k, v in obs.items()},
        raw={k: [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()] for k, v in raw.items()},
        M_hist=np.histogram(M, bins=41, range=(-1.2, 1.2))[0].tolist(),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--Ls", default="8,16,32,48,64,128")
    ap.add_argument("--n", default="4096,4096,2048,1024,1024,512")
    ap.add_argument("--batch", default="256,128,64,32,32,16")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--append", action="store_true", help="keep existing eval.json results, skip done Ls")
    a = ap.parse_args()
    model, ck = load_checkpoint(os.path.join(a.run_dir, "model.pkl"), dtype=jnp.float64)
    tc = ck["train"]
    delta = float(model.mdelta.delta.get_value())
    out = dict(run=a.run_dir, flow=ck["flow"], train=tc, delta=delta,
               log_a=float(model.mdelta.log_a.get_value()), log_b=float(model.mdelta.log_b.get_value()),
               results=[])
    path = os.path.join(a.run_dir, "eval.json")
    if a.append and os.path.exists(path):
        out["results"] = json.load(open(path))["results"]
    done = {r["L"] for r in out["results"]}
    for L, n, b in zip(map(int, a.Ls.split(",")), map(int, a.n.split(",")), map(int, a.batch.split(","))):
        if L in done:
            continue
        r = evaluate(model, L, tc["m2"], tc["lam"], n, b, seed=a.seed + L, tol=a.tol)
        print(json.dumps({k: r[k] for k in ("L", "ode_steps", "varlogw_N", "ess", "sec_per_sample")}),
              "U=", r["obs"]["U"], "chi=", r["obs"]["chi"], flush=True)
        out["results"].append(r)
        out["results"].sort(key=lambda x: x["L"])
        with open(path, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
