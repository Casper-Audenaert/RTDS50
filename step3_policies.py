import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (COVERAGE_FLOOR, FA_BUDGET, FIXED_H, N_BOOT, RES, SEED, WINDOWS, ece, logit,
                    policy_metrics, run_fixed, run_halting, sigmoid)

rng = np.random.default_rng(SEED)
Z = np.load(os.path.join(RES, "probs.npz"))
yva, yte = Z["yva"], Z["yte"]
MODELS = sorted({k.rsplit("_", 1)[0] for k in Z.files if k.endswith("_val")})
P = {m: {"val": Z[f"{m}_val"], "test": Z[f"{m}_test"]} for m in MODELS}
j24 = WINDOWS.index(FIXED_H)


def platt(p_val, y_val):
    c = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p_val)[:, None], y_val)
    return float(c.coef_[0, 0]), float(c.intercept_[0])


PC, RECAL = {}, {}
for m in MODELS:
    PC[m] = {"val": np.zeros_like(P[m]["val"]), "test": np.zeros_like(P[m]["test"])}
    RECAL[m] = []
    for j in range(len(WINDOWS)):
        a, b = platt(P[m]["val"][:, j], yva)
        RECAL[m].append([a, b])
        for s in ("val", "test"):
            PC[m][s][:, j] = sigmoid(a * logit(P[m][s][:, j]) + b)


def tune_fixed(p_val, y_val):
    return float(np.quantile(p_val[y_val == 0], 1 - FA_BUDGET))


def tune_halting(Pv, y_val):
    qs = np.quantile(Pv, np.linspace(0.01, 0.99, 99))
    best, best_key = None, None
    for thi in qs:
        for tlo in qs:
            if tlo >= thi:
                continue
            mt = policy_metrics(*run_halting(Pv, tlo, thi), y_val)
            if mt["fpr"] > FA_BUDGET or mt["coverage"] < COVERAGE_FLOOR:
                continue
            key = (round(mt["sens"], 3), -round(mt["mean_hour"], 1))
            if best_key is None or key > best_key:
                best, best_key = (float(tlo), float(thi)), key
    return best


THR, RUNS = {}, {}
for m in MODELS:
    for cal, PP in (("raw", P), ("cal", PC)):
        thr_f = tune_fixed(PP[m]["val"][:, j24], yva)
        tlo, thi = tune_halting(PP[m]["val"], yva)
        THR[f"{m}|{cal}"] = dict(fixed=thr_f, tlo=tlo, thi=thi)
        RUNS[f"fixed|{m}|{cal}"] = run_fixed(PP[m]["test"][:, j24], thr_f)
        RUNS[f"halting|{m}|{cal}"] = run_halting(PP[m]["test"], tlo, thi)
        print(f"{m}/{cal}: fixed thr={thr_f:.3f}  halting tlo={tlo:.3f} thi={thi:.3f}")

if "grud" in MODELS and "lr" in MODELS:
    for src, dst in (("lr", "grud"), ("grud", "lr")):
        for cal, PP in (("raw", P), ("cal", PC)):
            t = THR[f"{src}|{cal}"]
            RUNS[f"fixed|{dst}<-{src}|{cal}"] = run_fixed(PP[dst]["test"][:, j24], t["fixed"])
            RUNS[f"halting|{dst}<-{src}|{cal}"] = run_halting(PP[dst]["test"], t["tlo"], t["thi"])


def boot_ci(fn, n=N_BOOT):
    vals = np.array([fn(rng.integers(0, len(yte), len(yte))) for _ in range(n)])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


results = dict(windows=WINDOWS, fixed_h=FIXED_H, fa_budget=FA_BUDGET, coverage_floor=COVERAGE_FLOOR,
               n_test=int(len(yte)), n_test_deaths=int(yte.sum()),
               recalibration=RECAL, thresholds=THR, discrimination={}, policies={}, tests={})

disc_rows = []
for m in MODELS:
    for j, h in enumerate(WINDOWS):
        pt, pct = P[m]["test"][:, j], PC[m]["test"][:, j]
        r = dict(auroc=float(roc_auc_score(yte, pt)), auroc_ci=boot_ci(lambda i: roc_auc_score(yte[i], pt[i])),
                 auprc=float(average_precision_score(yte, pt)),
                 auprc_ci=boot_ci(lambda i: average_precision_score(yte[i], pt[i])),
                 ece_raw=ece(pt, yte), ece_cal=ece(pct, yte))
        results["discrimination"][f"{m}|{h}"] = r
        disc_rows.append(dict(model=m, hours=h, **{k: v for k, v in r.items() if not k.endswith("_ci")},
                              auroc_lo=r["auroc_ci"][0], auroc_hi=r["auroc_ci"][1]))
pd.DataFrame(disc_rows).to_csv(os.path.join(RES, "table_discrimination.csv"), index=False)

if "grud" in MODELS and "lr" in MODELS:
    for j, h in enumerate(WINDOWS):
        a, b = P["grud"]["test"][:, j], P["lr"]["test"][:, j]
        results["tests"][f"auroc_diff_grud_minus_lr|{h}"] = dict(
            diff=float(roc_auc_score(yte, a) - roc_auc_score(yte, b)),
            ci=boot_ci(lambda i: roc_auc_score(yte[i], a[i]) - roc_auc_score(yte[i], b[i])))

pol_rows = []
for name, (f, hr, dcd) in RUNS.items():
    mt = policy_metrics(f, hr, dcd, yte)
    mt["sens_ci"] = boot_ci(lambda i: f[i][yte[i] == 1].mean())
    mt["fpr_ci"] = boot_ci(lambda i: f[i][yte[i] == 0].mean())
    mt["coverage_ci"] = boot_ci(lambda i: dcd[i].mean())
    mt["mean_hour_ci"] = boot_ci(lambda i: hr[i].mean())
    results["policies"][name] = mt
    pol_rows.append(dict(policy=name, **{k: v for k, v in mt.items() if not k.endswith("_ci")}))
pd.DataFrame(pol_rows).to_csv(os.path.join(RES, "table_policies.csv"), index=False)

for m in MODELS:
    for cal in ("raw", "cal"):
        f1, h1, _ = RUNS[f"fixed|{m}|{cal}"]
        f2, h2, _ = RUNS[f"halting|{m}|{cal}"]
        pos = yte == 1
        b = int((f1[pos] & ~f2[pos]).sum())
        c = int((~f1[pos] & f2[pos]).sum())
        results["tests"][f"fixed_vs_halting|{m}|{cal}"] = dict(
            fixed_only=b, halting_only=c,
            mcnemar_p=float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0,
            hour_diff=float((h2 - h1).mean()), hour_diff_ci=boot_ci(lambda i: (h2[i] - h1[i]).mean()))

def sens_at_fpr(y, p, fpr):
    thr = np.quantile(p[y == 0], 1 - fpr)
    return (p[y == 1] >= thr).mean()


for m in MODELS:
    f_h = RUNS[f"halting|{m}|cal"][0]
    p24 = PC[m]["test"][:, j24]
    fpr_h = f_h[yte == 0].mean()
    sens_h = f_h[yte == 1].mean()
    sens_f = sens_at_fpr(yte, p24, fpr_h)

    def diff(i):
        y, f, p = yte[i], f_h[i], p24[i]
        return f[y == 1].mean() - sens_at_fpr(y, p, f[y == 0].mean())
    results["tests"][f"matched_fpr|{m}"] = dict(
        fpr=float(fpr_h), halting_sens=float(sens_h), fixed_sens_at_same_fpr=float(sens_f),
        diff=float(sens_h - sens_f), diff_ci=boot_ci(diff))

dec_out = {"main_model": "grud" if "grud" in MODELS else "lr"}
for m in MODELS:
    f, hr, dcd = RUNS[f"halting|{m}|cal"]
    dec_out.update({f"{m}_flag": f, f"{m}_hour": hr, f"{m}_decided": dcd, f"{m}_p_cal": PC[m]["test"],
                    f"{m}_thi": THR[f"{m}|cal"]["thi"], f"{m}_tlo": THR[f"{m}|cal"]["tlo"]})
np.savez(os.path.join(RES, "decisions.npz"), **dec_out)

with open(os.path.join(RES, "results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print(pd.DataFrame(pol_rows).round(3).to_string(index=False))
print("saved results.json")
