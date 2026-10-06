"""scikit-learn models with patient-grouped probability calibration.

* ``size``      - logistic regression on the equivalent diameter (a size-rule baseline)
* ``radiomics`` - 8 shape and density features, scaler, logistic regression
* ``prior``     - the training prevalence for every nodule (the floor of all baselines)
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lunglens.data.splits import patient_folds
from lunglens.features import DiameterOnly, PatchFeatures

CLASSICAL_KINDS = ("prior", "size", "radiomics")


def build_pipeline(kind: str, seed: int = 13) -> Pipeline:
    if kind == "prior":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])
    if kind == "size":
        return Pipeline([("features", DiameterOnly()), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(random_state=seed))])
    if kind == "radiomics":
        return Pipeline([("features", PatchFeatures()), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=seed))])
    raise ValueError(f"unknown model kind {kind!r}; expected one of {CLASSICAL_KINDS}")


def fit_calibrated(kind: str, patches: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                   method: str = "sigmoid", n_folds: int = 5, seed: int = 13):
    """Fit the pipeline and calibrate it on held-out PATIENTS inside the training set."""
    if kind == "prior":
        return build_pipeline(kind, seed).fit(patches, labels)
    folds = patient_folds(labels, groups, n_splits=n_folds, seed=seed)
    cal = CalibratedClassifierCV(build_pipeline(kind, seed), method=method, cv=folds)
    return cal.fit(patches, labels)


def predict_proba(model, patches: np.ndarray) -> np.ndarray:
    """Malignancy probability for each patch."""
    return model.predict_proba(patches)[:, 1]
