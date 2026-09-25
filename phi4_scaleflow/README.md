# Scale-covariant neural-operator flow for 2D φ⁴ at criticality

A continuous normalizing flow (CNF) that samples 2D lattice φ⁴ at its critical point. The same
parameters work on any lattice size L. It is built on [bijx](https://github.com/mathisgerdes/bijx),
using JAX and flax.nnx.

The report is [`report/REPORT.md`](report/REPORT.md). Raw numbers are in `results/` and `runs/*/eval.json`.

## Layout

```
scaleflow/
  theory.py       action, lattice momenta, observables (M, G(r))
  hmc.py          vectorised leapfrog HMC, many chains (ground truth)
  stats.py        jackknife, Sokal-window integrated autocorrelation time
  observables.py  chi, Binder U, G(r): MC estimates (HMC) and reweighted estimates (flow)
  model.py        base, scale-covariant velocity, exact divergence, M_Delta layer, RK4 CNF
  train.py        reverse-KL training on several L with a curriculum (float32)
  evaluate.py     float64 evaluation, ODE-step convergence check, exact log-weights
tests/            unit tests of handout section 8 (float64)
scripts/
  m0_critical_scan.py   Binder-cumulant scan for lambda_c
  m0_hmc_reference.py   HMC reference at lambda_c for L in {8,16,32,48,64,128}
  run_all.py            train and evaluate every variant and seed (parallel single-threaded workers)
  make_report.py        figures, summary.csv / summary.json
configs/          full model (scale.json) and ablations (untied, nodelta, local, single)
```

## Reproduce

```bash
pip install bijx pytest matplotlib
python -m pytest tests                                  # section 8 unit tests (~2 min, CPU)
python scripts/m0_critical_scan.py                      # M0: lambda_c from Binder crossings
python scripts/m0_hmc_reference.py --lam 4.25           # M0: ground truth
python scripts/run_all.py                               # M2 and M3: train and evaluate all variants and seeds
python scripts/make_report.py                           # figures and tables
```

## How the model maps to the handout

| Handout | Where | bijx pieces used |
|---|---|---|
| §3.1 free-field base, m₀ = μ₀/L | `model.free_base` | `IndependentNormal` and `FreeTheoryScaling(half=False)`, giving C(p) = 1/(2(p̂²+m₀²)) |
| §3.2 CNF on the augmented state (φ, ℓ) | `model.RK4Flow` | a `bijx.Bijection` with the bijx vector-field convention `(t, x) -> (dx/dt, dlogq/dt)` |
| §3.2 Z₂-odd velocity | `Velocity.velocity_and_div` | |
| §3.2 M_Δ layer | `model.AnomalousScaling` | `bijx.ApplyBijection`, closed-form log det |
| §3.3 scale-covariant velocity | `Conditioner`, `RadialKernelNet` | |
| §3.4 exact divergence | `Velocity.v_tilde_and_div` | one forward-mode JVP (no backward pass needed) |
| composition | `ScaleFlow.flow` | `bijx.Chain(cnf, M_Delta)` |

Each unit test in §8 has a test in `tests/test_model.py`:

1. Divergence matches the brute-force Jacobian trace on 4×4 and 6×6, for every variant.
2. The integrated ℓ matches log|det| of the discrete RK4 map.
3. Base covariance and log q₀.
4. The M_Δ log det matches brute force.
5. Z₂, translation and D₄ symmetry.
6. ∂c_x/∂φ_x = 0.
7. Responses at scale s on L match those at scale 2s on 2L.

There is also a round-trip test of sampling against density evaluation on L = 6, 9 and 12.

## Decisions on the open questions (§10)

- **Kernels: radial.** With radial kernels, D₄ equivariance is exact and costs nothing. The
  kernel is k_θ(z) = MLP(|z|²) · |z|² e^{−|z|²/2}. The |z|² input keeps it smooth in the offset. The
  |z|² prefactor makes k(0) = 0, so masking the centre removes nothing in the continuum. Before
  this change the L → 2L covariance error was about 12% at s = 2 and fell like 1/s². After it, the
  error is about 1.5% at s = 1 and at most 0.6% elsewhere (test 7).
- **Scales:** r = √2, s_min = 1, s_max ≤ L/2. That gives K = 5, 7, 9, 10, 11 and 13 scales for
  L = 8, 16, 32, 48, 64 and 128. n_UV = 1 and n_IR = 2 scale indices get learned embeddings, and
  the IR ones are counted from the top.
- **Order:** CNF, then M_Δ, as the handout plans. Δ is a single learnable scalar, initialised at 0.
- **Envelope:** a Gaussian times |z|², as above.
- **μ₀ = 2:** the base zero mode then has Var(M) = 1/(2μ₀²) = 0.125, independent of L.
- **Coupling:** m² = −4. The Binder crossings of L = 16/32 and 32/64 both sit at λ ≈ 4.245
  (`results/m0_scan.json`), so the runs use λ = 4.25.

## Deviations from the handout (flagged, see §9)

1. **Global zero-mode input to τ** (`FlowConfig.zero_mode_input`, on by default and in every
   variant).
   - Why it's needed: every kernel sums to zero and δ_x removes the local mean, so the velocity of
     §3.3 is almost blind to the magnetization M. It cannot build the double-peaked P(M) found at
     criticality.
   - What I added: M = mean(φ) as an extra input to τ.
   - What's unchanged: c_x is still independent of φ_x. The divergence stays exact, because the
     same single JVP with tangent (1 on δ, 1/N on M) returns ∂τ_x/∂δ_x + (1/N)·∂τ_x/∂M, the true
     diagonal. Test 1 checks this against the brute-force trace.
   - Effect: in a 600-step L = 8 run, it lowered the training Var(log w)/N from about 0.5 to 0.045.
   - Symmetry: M is a pure IR, L-independent quantity in the free frame, so this fits §1 decision 4
     (the IR end is untied).
2. **The last per-(x, k) layer of the scale mixer is linear and applied after the sum over k.**
   Σ_k W h_k = W Σ_k h_k, so this is the same function class at 1/K of the cost.
3. **The v2 conditioner** (`pool="mean_ends"`, `zero_sum="local"`, both the defaults). v1 used the
   handout's sum over scales and globally zero-sum kernels, and its conditioner statistics grew with L.
   v2 averages over the shared scales, reads out the untied end scales separately, and uses
   band-pass kernels that are zero-sum over their own scale. `runs_v1/` holds the v1 runs. See
   report §2 for the details and the (negative) effect on transfer.
4. **Model size and budget are scaled down for CPU** (see the next section). The spec defaults
   (batch 128, wider channels, more steps) are one config edit away.
5. **HMC autocorrelations** are measured without Z₂ flips, so they show plain-HMC critical slowing
   down.

## Compute caveat

Everything was run on a 4-core CPU with no GPU. The elementwise work per velocity evaluation scales
like B·N·K·H, and this machine processes it at about 1 GB/s of memory traffic. That is roughly
100 times slower than a single GPU.

The runs therefore use:

- channel widths E = 4, H = 8 and C = 8;
- training batches of 64, 32 and 16 at L = 8, 16 and 32;
- 2000 Adam steps;
- one seed for each ablation and 3 seeds for the full model;
- 64–2048 evaluation samples per L, falling as L grows (float64 on one core costs about 50 s per sample at L = 128).

These numbers test the machinery and the transfer trend. They are not converged samplers.
