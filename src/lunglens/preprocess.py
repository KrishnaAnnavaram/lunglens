"""Image preprocessing. Training and inference call the same functions.

The prototype normalized images one way on CPU and another way on GPU. Here there is one
definition: resample to 1 mm voxels, cut a cube around the nodule centre, keep Hounsfield
units (HU) for the features, and scale with one fixed window for the CNN.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

PATCH_MM = 32
HU_MIN, HU_MAX = -1000.0, 400.0
AIR_HU = -1000.0


def resample_isotropic(volume: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Resample a (z, y, x) HU volume to 1 mm voxels (linear interpolation)."""
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (z, y, x), got shape {volume.shape}")
    if any(s <= 0 for s in spacing_zyx):
        raise ValueError(f"spacing must be positive, got {spacing_zyx}")
    zoom = tuple(float(s) for s in spacing_zyx)
    if np.allclose(zoom, 1.0):
        return volume.astype(np.float32, copy=False)
    return ndimage.zoom(volume.astype(np.float32), zoom, order=1)


def extract_patch(volume_hu: np.ndarray, spacing_zyx: tuple[float, float, float],
                  center_zyx_vox: tuple[float, float, float], size: int = PATCH_MM) -> np.ndarray:
    """Cut a ``size``^3 cube (1 mm voxels) around a centre given in ORIGINAL voxel coordinates.

    Voxels outside the scan get the HU value of air.
    """
    iso = resample_isotropic(volume_hu, spacing_zyx)
    center = np.round(np.asarray(center_zyx_vox, dtype=float) * np.asarray(spacing_zyx)).astype(int)
    half = size // 2
    out = np.full((size, size, size), AIR_HU, dtype=np.float32)
    src_lo = center - half
    src_hi = src_lo + size
    lo = np.maximum(src_lo, 0)
    hi = np.minimum(src_hi, iso.shape)
    if np.any(hi <= lo):
        raise ValueError(f"centre {center_zyx_vox} is outside the volume {volume_hu.shape}")
    dst_lo = lo - src_lo
    dst_hi = dst_lo + (hi - lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = iso[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return out


def window(patch_hu: np.ndarray) -> np.ndarray:
    """Scale HU to [0, 1] with the fixed lung window."""
    return ((np.clip(patch_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def check_patch(patch: np.ndarray, size: int = PATCH_MM) -> np.ndarray:
    if patch.shape != (size, size, size):
        raise ValueError(f"patch must have shape {(size,) * 3}, got {patch.shape}")
    if not np.isfinite(patch).all():
        raise ValueError("patch has NaN or infinite values")
    if patch.min() < -2000 or patch.max() > 4000:
        raise ValueError("patch values are not in Hounsfield units (expected about -1000..3000)")
    if patch.max() - patch.min() < 50:
        raise ValueError("patch values look scaled (range < 50); give Hounsfield units, not a normalized image")
    return patch
