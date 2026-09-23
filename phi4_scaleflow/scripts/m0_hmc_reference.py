"""M0: HMC ground truth at the chosen critical coupling, all lattice sizes, with autocorrelations."""
import argparse, json, sys
sys.path.insert(0, ".")
import numpy as np
from scaleflow.hmc import run_hmc
from scaleflow.observables import mc_observables

ap = argparse.ArgumentParser()
ap.add_argument("--m2", type=float, default=-4.0)
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--Ls", default="8,16,32,48,64,128")
ap.add_argument("--out", default="results/hmc_reference.json")
a = ap.parse_args()
plan = {8: (64, 8000), 16: (64, 10000), 32: (64, 12000), 48: (64, 14000), 64: (64, 16000), 128: (32, 24000)}
res = []
for L in map(int, a.Ls.split(",")):
    nc, nt = plan[L]
    r = run_hmc(L, a.m2, a.lam, n_chains=nc, n_traj=nt, n_therm=2000, seed=L)
    o = mc_observables(r["M"], L, G=r["G"])
    n_indep = nc * nt / (2 * o["tau_M2"])
    row = dict(L=L, lam=a.lam, m2=a.m2, n_chains=nc, n_traj=nt, eps=r["eps"], n_leap=r["n_leap"],
               acc=float(r["acc"].mean()), seconds=r["seconds"],
               sec_per_traj_per_chain=r["seconds"] / (nc * nt),
               sec_per_indep_sample=r["seconds"] / n_indep,
               tau_M2=o["tau_M2"], tau_absM=o["tau_absM"],
               obs={k: [np.asarray(o[k][0]).tolist(), np.asarray(o[k][1]).tolist()] for k in ("M", "absM", "chi", "U", "G")},
               M_hist=np.histogram(r["M"].ravel(), bins=41, range=(-1.2, 1.2))[0].tolist())
    print(json.dumps({k: row[k] for k in ("L", "acc", "tau_M2", "tau_absM", "sec_per_indep_sample")}),
          "U=", row["obs"]["U"], "chi=", row["obs"]["chi"], flush=True)
    res.append(row)
    json.dump(res, open(a.out, "w"))
