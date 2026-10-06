"""Patient-level splits. All nodules of one patient go to the same side."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


class LeakageError(AssertionError):
    pass


def assert_no_patient_overlap(*groups: pd.Series | np.ndarray) -> None:
    sets = [set(np.asarray(g).tolist()) for g in groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if shared:
                raise LeakageError(f"{len(shared)} patient(s) in two splits, for example {sorted(shared)[:3]}")


def train_test_split_by_patient(df: pd.DataFrame, test_size: float = 0.25, seed: int = 13
                                ) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, df["label"], groups=df["patient_id"]))
    assert_no_patient_overlap(df["patient_id"].iloc[tr], df["patient_id"].iloc[te])
    return tr, te


def patient_folds(labels: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 13
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified group K-fold indices. Used for calibration inside the training set."""
    n_groups = len(np.unique(groups))
    k = max(2, min(n_splits, n_groups))
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(
        np.zeros(len(labels)), labels, groups))
    for tr, va in folds:
        assert_no_patient_overlap(groups[tr], groups[va])
    return folds
