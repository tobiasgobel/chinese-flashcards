"""Collect eval.json files + HMC reference into plots, a results table (CSV/JSON) and figures."""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
TRAIN_LS = {8, 16, 32}

LABEL = {"scale": "full (scale-covariant)", "untied": "(a) untied scales", "nodelta": "(b) no $M_\\Delta$",
         "local": "(c) local CNN", "single": "(d) single kernel"}
COLOR = {"scale": "#1f5fbf", "untied": "#d9822b", "nodelta": "#7a4fb3", "local": "#3a9d5d", "single": "#c23b3b"}


def variant(ev):
    f = ev["flow"]
    return "nodelta" if not f["use_delta_layer"] else f["mode"]


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "eval.json"))):
        ev = json.load(open(p))
        ev["variant"] = variant(ev)
        ev["name"] = os.path.basename(os.path.dirname(p))
        runs.append(ev)
    return runs


def main():
    runs = load_runs()
    hmc = {r["L"]: r for r in json.load(open(os.path.join(ROOT, "results", "hmc_reference.json")))}
    rows = []
    for ev in runs:
        for r in ev["results"]:
            h = hmc.get(r["L"])
            row = dict(run=ev["name"], variant=ev["variant"], seed=ev["train"]["seed"], L=r["L"],
                       held_out=r["L"] not in TRAIN_LS, n=r["n"], ode_steps=r["ode_steps"],
                       varlogw_N=r["varlogw_N"], varlogw_N_err=r["varlogw_N_err"], ess=r["ess"],
                       kl_per_site=r["kl_per_site"], delta=ev["delta"],
                       U=r["obs"]["U"][0], dU=r["obs"]["U"][1], chi=r["obs"]["chi"][0], dchi=r["obs"]["chi"][1],
                       U_raw=r["raw"]["U"][0], chi_raw=r["raw"]["chi"][0],
                       sec_per_eff_sample=r["sec_per_eff_sample"])
            if h:
                row.update(U_hmc=h["obs"]["U"][0], dU_hmc=h["obs"]["U"][1], chi_hmc=h["obs"]["chi"][0],
                           dchi_hmc=h["obs"]["chi"][1], hmc_sec_per_indep=h["sec_per_indep_sample"],
                           tau_M2=h["tau_M2"])
            rows.append(row)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(ROOT, "results", "summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    variants = [v for v in LABEL if any(e["variant"] == v for e in runs)]

    def agg(v, key, L):
        xs = [r[key] for r in rows if r["variant"] == v and r["L"] == L]
        return (np.mean(xs), np.std(xs) / np.sqrt(max(len(xs) - 1, 1)) if len(xs) > 1 else 0.0, len(xs)) if xs else None

    Ls = sorted({r["L"] for r in rows})

    # 1. Var(log w)/N vs L
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for v in variants:
        pts = [(L, *agg(v, "varlogw_N", L)[:2]) for L in Ls if agg(v, "varlogw_N", L)]
        pts = np.array(pts)
        ax.errorbar(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", ms=4, capsize=2, color=COLOR[v], label=LABEL[v])
    ax.axvspan(7, 34, color="0.92", zorder=0)
    ax.text(9, ax.get_ylim()[1] * 0.9 if False else 0, "")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(Ls); ax.set_xticklabels(Ls)
    ax.set_xlabel("L  (shaded: training sizes)"); ax.set_ylabel("Var(log w) / N")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "varlogw_vs_L.png"), dpi=160); plt.close(fig)

    # 2. observables vs L, full model vs HMC
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    hL = sorted(hmc)
    axs[0].errorbar(hL, [hmc[L]["obs"]["chi"][0] for L in hL], [hmc[L]["obs"]["chi"][1] for L in hL],
                    marker="s", color="k", ls="none", label="HMC", capsize=2)
    axs[1].errorbar(hL, [hmc[L]["obs"]["U"][0] for L in hL], [hmc[L]["obs"]["U"][1] for L in hL],
                    marker="s", color="k", ls="none", label="HMC", capsize=2)
    full = [e for e in runs if e["variant"] == "scale"]
    for i, e in enumerate(full):
        rr = e["results"]
        off = 1 + 0.03 * (i - 1)
        axs[0].errorbar([r["L"] * off for r in rr], [r["obs"]["chi"][0] for r in rr], [r["obs"]["chi"][1] for r in rr],
                        marker="o", ms=3, ls="none", color=COLOR["scale"], alpha=0.8, capsize=2,
                        label="flow, reweighted" if i == 0 else None)
        axs[0].plot([r["L"] * off for r in rr], [r["raw"]["chi"][0] for r in rr], "x", color=COLOR["scale"], alpha=0.5,
                    label="flow, raw" if i == 0 else None)
        axs[1].errorbar([r["L"] * off for r in rr], [r["obs"]["U"][0] for r in rr], [r["obs"]["U"][1] for r in rr],
                        marker="o", ms=3, ls="none", color=COLOR["scale"], alpha=0.8, capsize=2)
        axs[1].plot([r["L"] * off for r in rr], [r["raw"]["U"][0] for r in rr], "x", color=COLOR["scale"], alpha=0.5)
    xx = np.array([8, 128])
    c0 = hmc[hL[len(hL) // 2]]["obs"]["chi"][0] / hL[len(hL) // 2] ** 1.75
    axs[0].plot(xx, c0 * xx**1.75, ":", color="0.5", label="$\\propto L^{7/4}$")
    axs[1].axhline(0.61069, ls=":", color="0.5", label="$U^*$ (Ising)")
    for a in axs:
        a.set_xscale("log", base=2); a.set_xticks(hL); a.set_xticklabels(hL); a.set_xlabel("L")
    axs[0].set_yscale("log"); axs[0].set_ylabel("$\\chi = N\\langle M^2\\rangle$"); axs[1].set_ylabel("Binder U")
    axs[0].legend(fontsize=7, frameon=False); axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "observables_vs_L.png"), dpi=160); plt.close(fig)

    # 3. FSS collapse of G(r) L^{1/4} vs r/L (HMC and flow)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(hL):
        g = np.array(hmc[L]["obs"]["G"][0]); r = np.arange(len(g))
        axs[0].plot(r[1:] / L, g[1:] * L**0.25, "-", color=cmap(i / max(len(hL) - 1, 1)), label=f"L={L}")
    if full:
        e = full[0]
        for i, rr in enumerate(e["results"]):
            L = rr["L"]; g = np.array(rr["obs"]["G"][0]); r = np.arange(len(g))
            axs[1].plot(r[1:] / L, g[1:] * L**0.25, "-", color=cmap(hL.index(L) / max(len(hL) - 1, 1)) if L in hL else "k",
                        label=f"L={L}")
    axs[0].set_title("HMC", fontsize=9); axs[1].set_title("flow (reweighted, seed 0)", fontsize=9)
    for a in axs:
        a.set_xlabel("r / L")
    axs[0].set_ylabel("$G(r)\\,L^{1/4}$"); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "G_collapse.png"), dpi=160); plt.close(fig)

    # 4. cost
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(hL, [hmc[L]["sec_per_indep_sample"] for L in hL], "s-", color="k", label="HMC (per independent sample)")
    for v in variants:
        pts = [(L, agg(v, "sec_per_eff_sample", L)[0]) for L in Ls if agg(v, "sec_per_eff_sample", L)]
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, color=COLOR[v], label=LABEL[v] + " (per eff. sample)")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(hL); ax.set_xticklabels(hL)
    ax.set_xlabel("L"); ax.set_ylabel("seconds (4-core CPU)"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cost_vs_L.png"), dpi=160); plt.close(fig)

    # Delta summary
    deltas = {v: [e["delta"] for e in runs if e["variant"] == v] for v in variants}
    json.dump(deltas, open(os.path.join(ROOT, "results", "deltas.json"), "w"), indent=1)
    for v, d in deltas.items():
        print(v, "Delta = %.4f +- %.4f (n=%d)" % (np.mean(d), np.std(d, ddof=1) if len(d) > 1 else 0, len(d)))
    print("table:")
    for v in variants:
        print(v, " ".join("L%d:%.4f±%.4f" % (L, *agg(v, "varlogw_N", L)[:2]) for L in Ls if agg(v, "varlogw_N", L)))


if __name__ == "__main__":
    sys.exit(main())
