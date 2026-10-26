import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from common import RES, WINDOWS, policy_metrics, run_halting

P = np.array([[0.9, 0.9, 0.9, 0.9],
              [0.05, 0.9, 0.9, 0.9],
              [0.5, 0.5, 0.95, 0.5],
              [0.5, 0.5, 0.5, 0.5]])
flag, hour, decided = run_halting(P, tlo=0.1, thi=0.8)
assert flag.tolist() == [True, False, True, False]
assert hour.tolist() == [6, 6, 24, 48]
assert decided.tolist() == [True, True, True, False]
m = policy_metrics(flag, hour, decided, np.array([1, 1, 0, 1]))
assert abs(m["sens"] - 1 / 3) < 1e-9 and m["fpr"] == 1.0 and m["coverage"] == 0.75
print("halting rule unit checks: ok")

r = json.load(open(os.path.join(RES, "results.json")))
Z = np.load(os.path.join(RES, "probs.npz"))
yte = Z["yte"]
for m_ in ("lr", "grud"):
    if f"{m_}_test" not in Z.files:
        continue
    for j, h in enumerate(WINDOWS):
        a = roc_auc_score(yte, Z[f"{m_}_test"][:, j])
        assert abs(a - r["discrimination"][f"{m_}|{h}"]["auroc"]) < 1e-9, (m_, h)
print("AUROC per window matches results.json: ok")

thr = r["thresholds"]["lr|raw"]["fixed"]
p24 = Z["lr_test"][:, WINDOWS.index(24)]
sens = (p24[yte == 1] >= thr).mean()
fpr = (p24[yte == 0] >= thr).mean()
pol = r["policies"]["fixed|lr|raw"]
assert abs(sens - pol["sens"]) < 1e-9 and abs(fpr - pol["fpr"]) < 1e-9
print(f"fixed rule (LR, raw) re-derived: sens={sens:.3f} fpr={fpr:.3f}: ok")

t = r["thresholds"]["lr|raw"]
Pm = Z["lr_test"]
flags, hours = [], []
for i in range(len(yte)):
    f, hr = False, 48.0
    for j, h in enumerate(WINDOWS):
        if Pm[i, j] >= t["thi"]:
            f, hr = True, h
            break
        if Pm[i, j] <= t["tlo"]:
            f, hr = False, h
            break
    flags.append(f)
    hours.append(hr)
flags, hours = np.array(flags), np.array(hours)
pol = r["policies"]["halting|lr|raw"]
assert abs(flags[yte == 1].mean() - pol["sens"]) < 1e-9
assert abs(flags[yte == 0].mean() - pol["fpr"]) < 1e-9
assert abs(hours.mean() - pol["mean_hour"]) < 1e-9
print(f"halting rule (LR, raw) re-derived: sens={flags[yte==1].mean():.3f} "
      f"fpr={flags[yte==0].mean():.3f} mean hour={hours.mean():.1f}: ok")
print("all checks passed")
