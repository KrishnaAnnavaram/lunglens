"""A small 3-D CNN on nodule patches (optional extra ``deep``). torch is imported lazily.

The class follows the scikit-learn estimator API, so the same patient-grouped calibration
wraps it. It uses the same ``window()`` as every other code path.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from lunglens.preprocess import window


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError('the 3-D CNN needs the "deep" extra: pip install -e ".[deep]"') from exc
    return torch


def _net(torch, width: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv3d(1, width, 3, padding=1), nn.BatchNorm3d(width), nn.ReLU(), nn.MaxPool3d(2),
        nn.Conv3d(width, 2 * width, 3, padding=1), nn.BatchNorm3d(2 * width), nn.ReLU(), nn.MaxPool3d(2),
        nn.Conv3d(2 * width, 4 * width, 3, padding=1), nn.BatchNorm3d(4 * width), nn.ReLU(),
        nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * width, 1),
    )


class CNN3DClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, epochs: int = 15, width: int = 8, lr: float = 1e-3, batch_size: int = 16,
                 seed: int = 13, augment: bool = True):
        self.epochs = epochs
        self.width = width
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.augment = augment

    def _tensor(self, torch, X):
        return torch.from_numpy(window(np.asarray(X, dtype=np.float32))[:, None])

    def fit(self, X, y):  # noqa: N803
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.array([0, 1])
        self.model_ = _net(torch, self.width)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        y = np.asarray(y, dtype=np.float32)
        pos = max(y.mean(), 1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
        xt = self._tensor(torch, X)
        yt = torch.from_numpy(y)
        self.model_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(order), self.batch_size):
                idx = torch.from_numpy(order[s:s + self.batch_size])
                xb = xt[idx]
                if self.augment:  # random flips: a nodule has no fixed orientation
                    for ax in (2, 3, 4):
                        if rng.random() < 0.5:
                            xb = torch.flip(xb, dims=(ax,))
                opt.zero_grad()
                loss = loss_fn(self.model_(xb).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):  # noqa: N803
        torch = _torch()
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(self._tensor(torch, X)).squeeze(1)).numpy()
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_calibrated_cnn(patches: np.ndarray, labels: np.ndarray, groups: np.ndarray, epochs: int = 15,
                       n_folds: int = 3, seed: int = 13):
    from sklearn.calibration import CalibratedClassifierCV

    from lunglens.data.splits import patient_folds

    folds = patient_folds(labels, groups, n_splits=n_folds, seed=seed)
    cal = CalibratedClassifierCV(CNN3DClassifier(epochs=epochs, seed=seed), method="sigmoid", cv=folds)
    return cal.fit(patches, labels)
