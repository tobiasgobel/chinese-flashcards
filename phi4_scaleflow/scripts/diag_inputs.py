"""Diagnostic (handout 3.3.6): statistics of the inputs to tau (delta, c, M) across L, on base samples
and on model samples at t=1, for a trained run."""
import json, sys
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from scaleflow.train import load_checkpoint
from scaleflow.model import free_base, nn_mean

run = sys.argv[1]
model, ck = load_checkpoint(f"{run}/model.pkl")
out = []
for L in (8, 16, 32, 48, 64, 128):
    B = max(4, 2048 // L)
    noise, scale, _ = free_base(L, model.cfg.mu0)
    xi = jax.random.normal(jax.random.PRNGKey(L), (B, L, L))
    phi0, _ = scale.forward(xi, jnp.zeros(B))
    c = model.velocity.cond(phi0)
    d = phi0 - nn_mean(phi0)
    row = dict(L=L, phi_std=float(phi0.std()), delta_std=float(d.std()), M_std=float(phi0.mean((1, 2)).std()),
               c_rms=float(jnp.sqrt(jnp.mean(c**2))), c_std_per_channel=np.asarray(c.std((0, 1, 2))).round(3).tolist())
    print(json.dumps(row), flush=True); out.append(row)
json.dump(out, open(f"{run}/input_stats.json", "w"), indent=1)
