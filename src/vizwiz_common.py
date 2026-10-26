"""Shared code for the VizWiz answerability project (Group 4).

Every model in the project (baselines AND fusion) should use the same
loading, training, threshold and metric code from this file, so that
differences between models come only from the inputs and the fusion method.

Usage in any notebook:
    import sys; sys.path.append("/content/drive/MyDrive/vizwiz_fusion/code")
    from vizwiz_common import *
    meta, feats = load_handoff("/content/drive/MyDrive/vizwiz_fusion")
"""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default training recipe (same as the pilot). Fusion models may change lr / epochs,
# but should keep the loss, early stopping rule and threshold rule.
TRAIN_CFG = {          # same as the pilot (pilot models stopped after 5-7 epochs)
    "epochs": 15,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "patience": 3,
}


# ---------------------------------------------------------------- setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peak_vram_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None


# ---------------------------------------------------------------- data hand-off
def load_handoff(out_dir, names=None, mmap=True):
    """Load meta.csv and the cached feature arrays written by 02_data_and_baselines.

    Row i of every array matches row i of meta. Token arrays are large, so by default
    they are memory-mapped (read from disk only when a batch needs them).
    Returns (meta DataFrame, dict of name -> np.ndarray).
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "features" / "manifest.json").read_text())
    meta = pd.read_csv(out_dir / "features" / "meta.csv")
    assert len(meta) == manifest["n_rows"], "meta.csv and manifest disagree; re-run the data notebook"
    names = names or list(manifest["features"])
    feats = {}
    for n in names:
        path = out_dir / "features" / manifest["features"][n]["file"]
        feats[n] = np.load(path, mmap_mode="r" if mmap else None)
        assert len(feats[n]) == len(meta), f"{n} has {len(feats[n])} rows, meta has {len(meta)}"
    return meta, feats


def split_index(meta):
    """Row positions of each split: train = official train, dev = official val, test = official test."""
    return {s: np.where(meta["split"].values == s)[0] for s in ("train", "dev", "test")}


def vector_batch(X):
    """Batch function for one 2-D feature array (e.g. a pooled vector per pair)."""
    X = np.asarray(X)
    return lambda idx: (torch.from_numpy(np.ascontiguousarray(X[np.asarray(idx)])).to(DEVICE).float(),)


# ---------------------------------------------------------------- models
class MLPHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear. Outputs one logit."""

    def __init__(self, d_in, d_hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(d_hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------- training
@torch.no_grad()
def predict(model, get_batch, idx, bs=1024):
    """P(unanswerable) for the rows in idx, in the same order as idx."""
    model.eval()
    out = [torch.sigmoid(model(*get_batch(idx[i:i + bs]))).float().cpu() for i in range(0, len(idx), bs)]
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def log_loss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def train_model(model, get_batch, idx_train, idx_dev, target, y_hard, seed, cfg=None):
    """Train one model and return (best model, history DataFrame, cost dict).

    target : float array over ALL rows, the training target (soft or hard label)
    y_hard : float array over ALL rows, the official 0/1 label (1 = unanswerable)
    Loss    : BCE with pos_weight = (#answerable / #unanswerable) in train, so the
              minority class is not ignored.
    Stopping: best dev AUPRC (suits imbalanced classes), patience from cfg.
    """
    cfg = {**TRAIN_CFG, **(cfg or {})}
    model = model.to(DEVICE)
    pos = float(y_hard[idx_train].mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    target_t = torch.as_tensor(np.asarray(target), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    probe = rng.choice(idx_train, size=min(len(idx_train), 5000), replace=False)
    hist, best_ap, best_state, bad, secs = [], -1.0, None, 0, []
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        for b in np.array_split(rng.permutation(idx_train), max(1, len(idx_train) // cfg["batch_size"])):
            loss = loss_fn(model(*get_batch(b)), target_t[torch.as_tensor(b)].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        secs.append(time.time() - t0)
        p_tr, p_dev = predict(model, get_batch, probe), predict(model, get_batch, idx_dev)
        ap = average_precision_score(y_hard[idx_dev], p_dev)
        hist.append({"epoch": ep + 1, "train_loss": log_loss(p_tr, y_hard[probe]),
                     "dev_loss": log_loss(p_dev, y_hard[idx_dev]), "dev_auprc": ap})
        if ap > best_ap:
            best_ap, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    cost = {"sec_per_epoch": round(float(np.mean(secs)), 2), "epochs_run": len(hist),
            "params": int(sum(p.numel() for p in model.parameters())), "peak_vram_gb": peak_vram_gb()}
    return model, pd.DataFrame(hist), cost


# ---------------------------------------------------------------- threshold and metrics
def tune_threshold(p, y):
    """Threshold that maximises macro-F1 on dev. Always tune on dev, never on test."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, p >= t, average="macro") for t in grid]))])


def expected_calibration_error(p, y, n_bins=15):
    """ECE with equal-width bins: average |accuracy - confidence| weighted by bin size."""
    p, y = np.asarray(p), np.asarray(y)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate(p, y, t):
    """All project metrics for one set of probabilities p, labels y and threshold t."""
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(p) >= t).astype(int)
    return {"macro_f1": f1_score(y, yhat, average="macro"),
            "f1_unanswerable": f1_score(y, yhat, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, yhat),
            "accuracy": accuracy_score(y, yhat),
            "auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "ece": expected_calibration_error(p, y)}


def paired_bootstrap_gain(p_a, t_a, p_b, t_b, y, n_boot=1000, seed=42):
    """95% CI of macro-F1(model a) - macro-F1(model b), resampling the same rows for both."""
    y = np.asarray(y).astype(int)
    ya, yb = (np.asarray(p_a) >= t_a).astype(int), (np.asarray(p_b) >= t_b).astype(int)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        gains.append(f1_score(y[b], ya[b], average="macro") - f1_score(y[b], yb[b], average="macro"))
    return float(np.mean(gains)), [float(x) for x in np.percentile(gains, [2.5, 97.5])]


# ---------------------------------------------------------------- shared prediction files
def save_predictions(out_dir, model_name, seed, p_dev, p_test, threshold, extra=None):
    """Write one model's predictions in the shared format the evaluation notebook reads.

    File: <out_dir>/predictions/<model_name>__seed<seed>.npz with p_dev, p_test (row order =
    split_index(meta)["dev"/"test"]), and the dev-tuned threshold.
    """
    pred_dir = Path(out_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    np.savez(pred_dir / f"{safe}__seed{seed}.npz", model=model_name, seed=seed,
             p_dev=np.asarray(p_dev, np.float32), p_test=np.asarray(p_test, np.float32),
             threshold=float(threshold), extra=json.dumps(extra or {}))


def load_predictions(out_dir):
    """All saved predictions as a list of dicts (model, seed, p_dev, p_test, threshold)."""
    rows = []
    for f in sorted((Path(out_dir) / "predictions").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        rows.append({"model": str(z["model"]), "seed": int(z["seed"]), "p_dev": z["p_dev"],
                     "p_test": z["p_test"], "threshold": float(z["threshold"]),
                     "extra": json.loads(str(z["extra"]))})
    return rows
