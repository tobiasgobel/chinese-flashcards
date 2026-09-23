"""M0: locate lambda_c at fixed m2 from Binder-cumulant crossings (HMC)."""
import argparse, json, sys
sys.path.insert(0, ".")
import numpy as np
from scaleflow.hmc import run_hmc
from scaleflow.observables import mc_observables

ap = argparse.ArgumentParser()
ap.add_argument("--m2", type=float, default=-4.0)
ap.add_argument("--lams", type=str, default="4.0,4.1,4.15,4.2,4.25,4.3,4.4,4.5")
ap.add_argument("--Ls", type=str, default="8,16,32,64")
ap.add_argument("--out", default="results/m0_scan.json")
a = ap.parse_args()
rows = []
for L in map(int, a.Ls.split(",")):
    for lam in map(float, a.lams.split(",")):
        n_traj = {8: 4000, 16: 6000, 32: 8000, 64: 10000}.get(L, 10000)
        r = run_hmc(L, a.m2, lam, n_chains=64, n_traj=n_traj, n_therm=1000, seed=int(1000 * lam) + L)
        o = mc_observables(r["M"], L)
        row = dict(L=L, lam=lam, U=o["U"][0], dU=o["U"][1], chi=o["chi"][0], dchi=o["chi"][1],
                   absM=o["absM"][0], tau_M2=o["tau_M2"], acc=float(r["acc"].mean()))
        print(json.dumps(row), flush=True)
        rows.append(row)
        json.dump(rows, open(a.out, "w"), indent=1)
