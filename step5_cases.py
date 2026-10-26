import os

import numpy as np
import pandas as pd

from common import RES, WINDOWS, load_data, window

D = np.load(os.path.join(RES, "decisions.npz"))
m = str(D["main_model"])
Z = np.load(os.path.join(RES, "probs.npz"))
yte, icu = Z["yte"], Z["icu_te"]
Xte = load_data()["Xte"]
thi, tlo = float(D[f"{m}_thi"]), float(D[f"{m}_tlo"])
print("halting rule, recalibrated", m)

df = pd.DataFrame({"y": yte, "icu_type": icu, "flag": D[f"{m}_flag"], "hour": D[f"{m}_hour"],
                   "decided": D[f"{m}_decided"],
                   "n_vars_48h": (~np.isnan(Xte)).any(1).sum(1),
                   **{f"n_obs_{h}h": (~np.isnan(window(Xte, h))).sum((1, 2)) for h in WINDOWS},
                   **{f"p_{h}": D[f"{m}_p_cal"][:, j] for j, h in enumerate(WINDOWS)}})
df["group"] = np.select(
    [~df.decided, df.flag & (df.y == 0), df.flag & (df.y == 1), ~df.flag & (df.y == 1)],
    ["abstain", "false_alarm", "true_alarm", "miss"], default="true_clear")
df.to_csv(os.path.join(RES, "cases.csv"), index=False)
g = df.groupby("group").agg(n=("y", "size"), death_rate=("y", "mean"), median_obs_48h=("n_obs_48h", "median"),
                            median_vars=("n_vars_48h", "median"), mean_hour=("hour", "mean"),
                            **{f"median_p{h}": (f"p_{h}", "median") for h in WINDOWS})
g.to_csv(os.path.join(RES, "case_groups.csv"))
print(g.round(3).to_string())

ab, fa, ta, ms = (df[df.group == k] for k in ("abstain", "false_alarm", "true_alarm", "miss"))
print(f"\nthresholds: clear <= {tlo:.3f}, flag >= {thi:.3f}")
print(f"decided by 6 h: {(df.hour <= 6).mean():.3f}; by 12 h: {(df.hour <= 12).mean():.3f}; "
      f"mean hour among decided: {df[df.decided].hour.mean():.1f}")
print(f"abstained: {len(ab)} ({len(ab)/len(df):.3f}); deaths among them {ab.y.mean():.3f}; "
      f"median obs 48h {ab.n_obs_48h.median()} vs decided {df[df.decided].n_obs_48h.median()}; "
      f"p48 IQR {ab.p_48.quantile(.25):.2f}-{ab.p_48.quantile(.75):.2f}")
print(f"false alarms: {len(fa)}; flagged at 6 h: {(fa.hour == 6).sum()}, of which p24 < flag thr: "
      f"{((fa.hour == 6) & (fa.p_24 < thi)).sum()}; p48 < flag thr: {(fa.p_48 < thi).sum()}")
print(f"MICU share: all {(df.icu_type == 3).mean():.2f}, false alarms {(fa.icu_type == 3).mean():.2f}; "
      f"CSRU share: all {(df.icu_type == 2).mean():.2f}, true clears {(df[df.group == 'true_clear'].icu_type == 2).mean():.2f}")
print(f"misses: {len(ms)}; cleared at 6 h: {(ms.hour == 6).sum()}; median p48 {ms.p_48.median():.3f}")
