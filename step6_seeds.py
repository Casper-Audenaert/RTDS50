import glob
import json
import os

import numpy as np

from common import HERE

runs = {0: os.path.join(HERE, "..", "results", "results.json")}
for p in sorted(glob.glob(os.path.join(HERE, "..", "results_seed*", "results.json"))):
    runs[int(os.path.basename(os.path.dirname(p)).replace("results_seed", ""))] = p
R = {s: json.load(open(p)) for s, p in runs.items()}

rows = {
    "AUROC 6 h, GRU-D": lambda r: r["discrimination"]["grud|6"]["auroc"],
    "AUROC 6 h, LR": lambda r: r["discrimination"]["lr|6"]["auroc"],
    "AUROC 48 h, GRU-D": lambda r: r["discrimination"]["grud|48"]["auroc"],
    "AUROC 48 h, LR": lambda r: r["discrimination"]["lr|48"]["auroc"],
    "AUROC 6 h, missingness only": lambda r: r["discrimination"]["lr_miss|6"]["auroc"],
    "fixed FPR, LR": lambda r: r["policies"]["fixed|lr|cal"]["fpr"],
    "fixed FPR, GRU-D": lambda r: r["policies"]["fixed|grud|cal"]["fpr"],
    "halting FPR, LR": lambda r: r["policies"]["halting|lr|cal"]["fpr"],
    "halting FPR, GRU-D": lambda r: r["policies"]["halting|grud|cal"]["fpr"],
    "halting coverage, GRU-D": lambda r: r["policies"]["halting|grud|cal"]["coverage"],
    "halting sens - fixed sens, GRU-D": lambda r: r["policies"]["halting|grud|cal"]["sens"] - r["policies"]["fixed|grud|cal"]["sens"],
    "halting sens - fixed sens, LR": lambda r: r["policies"]["halting|lr|cal"]["sens"] - r["policies"]["fixed|lr|cal"]["sens"],
    "matched-FPR sens diff, GRU-D": lambda r: r["tests"]["matched_fpr|grud"]["diff"],
    "matched-FPR sens diff, LR": lambda r: r["tests"]["matched_fpr|lr"]["diff"],
    "transfer LR->GRU-D coverage shift (raw)": lambda r: r["policies"]["halting|grud<-lr|raw"]["coverage"] - r["policies"]["halting|grud|raw"]["coverage"],
    "transfer LR->GRU-D coverage shift (recal)": lambda r: r["policies"]["halting|grud<-lr|cal"]["coverage"] - r["policies"]["halting|grud|cal"]["coverage"],
    "transfer GRU-D->LR coverage shift (raw)": lambda r: r["policies"]["halting|lr<-grud|raw"]["coverage"] - r["policies"]["halting|lr|raw"]["coverage"],
    "transfer GRU-D->LR coverage shift (recal)": lambda r: r["policies"]["halting|lr<-grud|cal"]["coverage"] - r["policies"]["halting|lr|cal"]["coverage"],
    "transfer LR->GRU-D FPR shift (raw)": lambda r: r["policies"]["halting|grud<-lr|raw"]["fpr"] - r["policies"]["halting|grud|raw"]["fpr"],
    "transfer GRU-D->LR FPR shift (raw)": lambda r: r["policies"]["halting|lr<-grud|raw"]["fpr"] - r["policies"]["halting|lr|raw"]["fpr"],
    "fixed 24 h FPR shift LR->GRU-D (raw)": lambda r: r["policies"]["fixed|grud<-lr|raw"]["fpr"] - r["policies"]["fixed|grud|raw"]["fpr"],
    "fixed 24 h FPR shift GRU-D->LR (raw)": lambda r: r["policies"]["fixed|lr<-grud|raw"]["fpr"] - r["policies"]["fixed|lr|raw"]["fpr"],
}
out = {}
print(f"{'quantity':45s}" + "".join(f"seed{s:<8d}" for s in R) + "  mean    min     max")
for name, fn in rows.items():
    v = np.array([fn(R[s]) for s in R])
    out[name] = dict(per_seed={str(s): float(x) for s, x in zip(R, v)}, mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    print(f"{name:45s}" + "".join(f"{x:<12.3f}" for x in v) + f"  {v.mean():.3f}  {v.min():.3f}  {v.max():.3f}")
with open(os.path.join(HERE, "..", "results", "seed_summary.json"), "w") as fh:
    json.dump(out, fh, indent=1)
