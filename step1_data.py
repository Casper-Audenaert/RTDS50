import json
import os

import numpy as np

from common import RES, WINDOWS, load_data, window

d = load_data()
Xtr, Xva, Xte = d["Xtr"], d["Xva"], d["Xte"]
summary = dict(
    n=dict(train=int(len(d["ytr"])), val=int(len(d["yva"])), test=int(len(d["yte"])),
           total=int(len(d["ytr"]) + len(d["yva"]) + len(d["yte"]))),
    n_steps=int(Xtr.shape[1]), n_features=int(Xtr.shape[2]),
    prevalence=dict(train=float(d["ytr"].mean()), val=float(d["yva"].mean()), test=float(d["yte"].mean())),
    missing_rate_train=float(np.isnan(Xtr).mean()),
    observed_values_per_test_patient={
        str(h): dict(mean=float((~np.isnan(window(Xte, h))).sum((1, 2)).mean()),
                     median=float(np.median((~np.isnan(window(Xte, h))).sum((1, 2)))))
        for h in WINDOWS},
    icu_type_share_test={str(int(k)): float(v) for k, v in
                         zip(*np.unique(d["icu_te"], return_counts=True)) for v in [v / len(d["icu_te"])]},
)
with open(os.path.join(RES, "data_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
