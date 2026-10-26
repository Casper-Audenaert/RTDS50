import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get("SEED", 0))
RES = os.path.join(HERE, "..", "results" if SEED == 0 else f"results_seed{SEED}")
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "data")
for folder in (RES, FIG, DATA):
    os.makedirs(folder, exist_ok=True)

WINDOWS = [6, 12, 24, 48]
FIXED_H = 24
FA_BUDGET = 0.10
COVERAGE_FLOOR = 0.80
N_BOOT = 1000
EPS = 1e-6
MODELS = ["lr", "lr_miss", "grud"]
LABEL = {"lr": "Logistic regression", "lr_miss": "LR, missingness only", "grud": "GRU-D"}


def load_data():
    path = os.path.join(DATA, f"physionet2012_seed{SEED}.npz")
    if not os.path.exists(path):
        from benchpots.datasets import preprocess_physionet2012
        d = preprocess_physionet2012(subset="all", rate=0, task_type="classification", random_state=SEED)
        np.savez_compressed(
            path, Xtr=d["train_X"], Xva=d["val_X"], Xte=d["test_X"],
            ytr=d["train_y"].astype(int), yva=d["val_y"].astype(int), yte=d["test_y"].astype(int),
            icu_te=d["test_ICUType"])
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def window(X, h):
    X = X.copy()
    X[:, h:, :] = np.nan
    return X


def lr_feats(X, missingness_only=False):
    obs = ~np.isnan(X)
    ind = obs.any(1).astype(float)
    cnt = np.log1p(obs.sum(1))
    if missingness_only:
        return np.concatenate([ind, cnt], 1)
    t_idx = np.where(obs, np.arange(X.shape[1])[None, :, None], -1).max(1)
    last = np.take_along_axis(np.nan_to_num(X), t_idx.clip(0)[:, None, :], 1)[:, 0, :]
    last[t_idx < 0] = 0.0
    return np.concatenate([last, ind, cnt], 1)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def run_fixed(p_fixed, thr):
    flag = p_fixed >= thr
    n = len(p_fixed)
    return flag, np.full(n, float(FIXED_H)), np.ones(n, bool)


def run_halting(Pm, tlo, thi):
    n = Pm.shape[0]
    dec = np.zeros(n, int)
    hour = np.full(n, float(WINDOWS[-1]))
    for j, h in enumerate(WINDOWS):
        open_ = dec == 0
        hi = open_ & (Pm[:, j] >= thi)
        lo = open_ & (Pm[:, j] <= tlo)
        dec[hi], dec[lo] = 1, -1
        hour[hi | lo] = h
    return dec == 1, hour, dec != 0


def policy_metrics(flag, hour, decided, y):
    pos, neg = y == 1, y == 0
    cov = decided.mean()
    return dict(
        sens=float(flag[pos].mean()),
        fpr=float(flag[neg].mean()),
        coverage=float(cov),
        mean_hour=float(hour.mean()),
        err_covered=float((flag[decided] != y[decided]).mean()) if cov > 0 else float("nan"),
    )
