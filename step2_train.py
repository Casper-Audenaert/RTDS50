import argparse
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import MODELS, RES, SEED, WINDOWS, load_data, lr_feats, window

ap = argparse.ArgumentParser()
ap.add_argument("--no-grud", action="store_true")
ap.add_argument("--epochs", type=int, default=40)
args = ap.parse_args()
np.random.seed(SEED)
torch.manual_seed(SEED)

d = load_data()
Xtr, Xva, Xte, ytr, yva, yte = d["Xtr"], d["Xva"], d["Xte"], d["ytr"], d["yva"], d["yte"]
N, T, F = Xtr.shape
models = [m for m in MODELS if not (args.no_grud and m == "grud")]


def fit_lr(Xa, ya, Xb_list, missingness_only=False):
    clf = LogisticRegression(C=0.1, max_iter=2000)
    clf.fit(lr_feats(Xa, missingness_only), ya)
    return [clf.predict_proba(lr_feats(Xb, missingness_only))[:, 1] for Xb in Xb_list]


def fit_grud(Xa, ya, Xv, yv, Xb_list):
    from pypots.classification import GRUD
    from pypots.utils.random import set_random_seed
    set_random_seed(SEED)
    m = GRUD(n_steps=T, n_features=F, n_classes=2, rnn_hidden_size=64, batch_size=64,
             epochs=args.epochs, patience=5, device="cpu", model_saving_strategy=None, verbose=False)
    m.fit({"X": Xa, "y": ya}, {"X": Xv, "y": yv})
    return [m.predict({"X": Xb})["classification_proba"][:, 1] for Xb in Xb_list]


out = {"yva": yva, "yte": yte, "icu_te": d["icu_te"]}
for m in models:
    out[f"{m}_val"] = np.zeros((len(yva), len(WINDOWS)))
    out[f"{m}_test"] = np.zeros((len(yte), len(WINDOWS)))

for j, h in enumerate(WINDOWS):
    Xtr_h, Xva_h, Xte_h = window(Xtr, h), window(Xva, h), window(Xte, h)
    out["lr_val"][:, j], out["lr_test"][:, j] = fit_lr(Xtr_h, ytr, [Xva_h, Xte_h])
    out["lr_miss_val"][:, j], out["lr_miss_test"][:, j] = fit_lr(Xtr_h, ytr, [Xva_h, Xte_h], missingness_only=True)
    if "grud" in models:
        out["grud_val"][:, j], out["grud_test"][:, j] = fit_grud(Xtr_h, ytr, Xva_h, yva, [Xva_h, Xte_h])
    print(f"h={h:2d}: " + "  ".join(f"{m} test AUROC={roc_auc_score(yte, out[f'{m}_test'][:, j]):.3f}" for m in models), flush=True)

np.savez(os.path.join(RES, "probs.npz"), **out)
print("saved", os.path.join(RES, "probs.npz"))
