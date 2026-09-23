import sys, time
sys.path.insert(0, ".")
from scaleflow.hmc import run_hmc
from scaleflow.observables import mc_observables
for L in (8, 16, 32):
    for lam in (4.0, 4.5, 5.0):
        r = run_hmc(L, -4.0, lam, n_chains=32, n_traj=1000, n_therm=300)
        o = mc_observables(r["M"], L)
        print(L, lam, "U=%.3f±%.3f chi=%.1f tau=%.1f" % (*o["U"], o["chi"][0], o["tau_M2"]), flush=True)
