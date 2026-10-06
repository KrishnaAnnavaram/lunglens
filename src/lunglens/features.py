"""Hand-made shape and density features of a nodule patch, as a scikit-learn transformer.

The transformer has no learned state, so it cannot leak test data. The scaler and the
classifier that follow it in the pipeline are fitted on training patients only.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = (
    "volume_mm3",
    "equivalent_diameter_mm",
    "max_extent_mm",
    "mean_hu",
    "std_hu",
    "calcified_fraction",
    "sphericity",
    "margin_gradient",
)
SOLID_HU = -400.0
CALCIUM_HU = 200.0


def nodule_mask(patch_hu: np.ndarray) -> np.ndarray:
    """The connected solid region that touches the patch centre (empty if the centre is not solid)."""
    solid = ndimage.binary_opening(patch_hu > SOLID_HU, iterations=1)
    labels, _ = ndimage.label(solid)
    c = tuple(s // 2 for s in patch_hu.shape)
    lab = labels[c]
    if lab == 0:
        near = labels[c[0] - 2:c[0] + 3, c[1] - 2:c[1] + 3, c[2] - 2:c[2] + 3]
        vals = near[near > 0]
        if vals.size == 0:
            return np.zeros_like(solid)
        lab = np.bincount(vals).argmax()
    return labels == lab


def patch_features(patch_hu: np.ndarray) -> np.ndarray:
    mask = nodule_mask(patch_hu)
    vol = float(mask.sum())
    if vol == 0:
        return np.array([0.0, 0.0, 0.0, float(patch_hu.mean()), float(patch_hu.std()), 0.0, 0.0, 0.0])
    eq_d = (6 * vol / np.pi) ** (1 / 3)
    idx = np.argwhere(mask)
    extent = float((idx.max(0) - idx.min(0) + 1).max())
    inside = patch_hu[mask]
    surface = mask & ~ndimage.binary_erosion(mask)
    sphere_area = np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)
    sphericity = float(min(sphere_area / max(surface.sum(), 1), 1.5))
    grad = ndimage.gaussian_gradient_magnitude(patch_hu, 1.0)
    return np.array([
        vol, eq_d, extent, float(inside.mean()), float(inside.std()),
        float((inside > CALCIUM_HU).mean()), sphericity, float(grad[surface].mean()),
    ])


class PatchFeatures(BaseEstimator, TransformerMixin):
    """(n, d, h, w) HU patches -> (n, 8) feature matrix."""

    def fit(self, X, y=None):  # noqa: N803 - scikit-learn naming
        return self

    def transform(self, X):  # noqa: N803
        X = np.asarray(X)
        if X.ndim != 4:
            raise ValueError(f"expected (n, d, h, w) patches, got shape {X.shape}")
        return np.stack([patch_features(p) for p in X])

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


class DiameterOnly(BaseEstimator, TransformerMixin):
    """The size baseline: equivalent diameter only."""

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return PatchFeatures().transform(X)[:, [1]]
