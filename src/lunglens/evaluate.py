"""Evaluation on held-out patients: discrimination, calibration and a reader reference.

Intervals come from a bootstrap that resamples PATIENTS, because nodules of one patient
are correlated.
"""
from __future__ import annotations

import statistics

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lunglens.data.schemas import binary_label


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    out = {"n": float(len(y)), "prevalence": float(y.mean()), "brier": float(brier_score_loss(y, p)),
           "ece": expected_calibration_error(y, p)}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    out["sensitivity"] = float(pred[pos].mean()) if pos.any() else float("nan")
    out["specificity"] = float((~pred[neg]).mean()) if neg.any() else float("nan")
    return out


def patient_level(df: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row per patient: label = any malignant nodule, score = highest nodule probability."""
    t = pd.DataFrame({"pid": df["patient_id"].to_numpy(), "y": df["label"].to_numpy(), "p": p})
    g = t.groupby("pid").agg(y=("y", "max"), p=("p", "max"))
    return g["y"].to_numpy(), g["p"].to_numpy()


def bootstrap_ci(df: pd.DataFrame, p: np.ndarray, metric: str, threshold: float, n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pids = df["patient_id"].to_numpy()
    uniq = np.unique(pids)
    by_pid = {u: np.flatnonzero(pids == u) for u in uniq}
    y = df["label"].to_numpy()
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by_pid[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        if metric in ("auroc", "auprc") and len(np.unique(y[idx])) < 2:
            continue
        vals.append(_metrics(y[idx], p[idx], threshold)[metric])
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate(df: pd.DataFrame, p: np.ndarray, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0) -> dict:
    y = df["label"].to_numpy()
    nodule = _metrics(y, p, threshold)
    for m in ("auroc", "auprc", "brier"):
        nodule[f"{m}_ci"] = bootstrap_ci(df, p, m, threshold, n_boot, seed)
    py, pp = patient_level(df, p)
    return {"nodule": nodule, "patient": _metrics(py, pp, threshold), "n_patients": int(df["patient_id"].nunique()),
            "threshold": threshold}


def reader_reference(df: pd.DataFrame) -> dict[str, float]:
    """AUROC of one reader's rating against the consensus of the OTHER readers.

    Only nodules with 2 or more readers count. This is a human reference, not a gold standard.
    """
    ys, ss = [], []
    for ratings in df["ratings"]:
        if len(ratings) < 2:
            continue
        for k, r in enumerate(ratings):
            others = [x for i, x in enumerate(ratings) if i != k]
            lab = binary_label(float(statistics.median(others)))
            if lab is not None:
                ys.append(lab)
                ss.append(r)
    if len(set(ys)) < 2:
        return {"reader_auroc": float("nan"), "n_reads": float(len(ys))}
    return {"reader_auroc": float(roc_auc_score(ys, ss)), "n_reads": float(len(ys))}
