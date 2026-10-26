import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, FIXED_H, LABEL, RES, WINDOWS, logit, sigmoid

r = json.load(open(os.path.join(RES, "results.json")))
Z = np.load(os.path.join(RES, "probs.npz"))
yte = Z["yte"]
models = sorted({k.split("|")[0] for k in r["discrimination"]}, key=["lr", "lr_miss", "grud"].index)
j24 = WINDOWS.index(FIXED_H)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
for ax, met in zip(axes, ("auroc", "auprc")):
    for m in models:
        v = np.array([r["discrimination"][f"{m}|{h}"][met] for h in WINDOWS])
        lo = np.array([r["discrimination"][f"{m}|{h}"][met + "_ci"][0] for h in WINDOWS])
        hi = np.array([r["discrimination"][f"{m}|{h}"][met + "_ci"][1] for h in WINDOWS])
        ax.errorbar(WINDOWS, v, yerr=[v - lo, hi - v], marker="o", ms=3, capsize=2, label=LABEL[m])
    ax.set_xticks(WINDOWS)
    ax.set_xlabel("Hours observed")
    ax.set_ylabel(met.upper())
    ax.grid(alpha=.3)
axes[0].axhline(0.5, color="grey", lw=.8, ls=":")
axes[1].axhline(yte.mean(), color="grey", lw=.8, ls=":")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_earliness.pdf"))

yva = Z["yva"]
D = np.load(os.path.join(RES, "decisions.npz"))
fig, ax = plt.subplots(figsize=(3.2, 2.6))
for m in [x for x in models if x != "lr_miss"]:
    color = None
    for jj, ls in ((j24, "-"), (len(WINDOWS) - 1, "--")):
        a, b = r["recalibration"][m][jj]
        p = sigmoid(a * logit(Z[f"{m}_test"][:, jj]) + b)
        pv = sigmoid(a * logit(Z[f"{m}_val"][:, jj]) + b)
        thr = float(np.quantile(pv[yva == 0], 0.9))
        dist = np.abs(logit(p) - logit(thr))
        cov, err = [], []
        for delta in np.quantile(dist, np.linspace(0, 0.95, 40)):
            keep = dist >= delta
            yy, fl = yte[keep], p[keep] >= thr
            if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
                continue
            cov.append(keep.mean())
            err.append(0.5 * ((~fl[yy == 1]).mean() + fl[yy == 0].mean()))
        line, = ax.plot(cov, err, ls, ms=3, color=color, label=f"{LABEL[m]}, {WINDOWS[jj]} h")
        color = line.get_color()
    dcd, fl = D[f"{m}_decided"], D[f"{m}_flag"]
    yd, fd = yte[dcd], fl[dcd]
    ax.scatter([dcd.mean()], [0.5 * ((~fd[yd == 1]).mean() + fd[yd == 0].mean())],
               marker="*", s=70, zorder=5, color=color)
ax.set_xlabel("Coverage (fraction of patients decided)")
ax.set_ylabel("Balanced error among decided")
ax.grid(alpha=.3)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_risk_coverage.pdf"))
print("figures written to", FIG)
