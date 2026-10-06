"""Seeded synthetic nodule patches (HU, 1 mm voxels) with multi-reader ratings.

This data is NOT medical data. It lets the tests and the offline demo run with no download.
The generator copies a few known trends: malignant nodules are larger and have irregular
margins, and many benign nodules are calcified. Each patient has its own noise level and
intensity offset, so a split that is not by patient leaks patient identity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from lunglens.data.manifest import write_manifest
from lunglens.data.schemas import NoduleRecord
from lunglens.preprocess import PATCH_MM


def _nodule_mask(rng: np.random.Generator, size: int, radius: float, irregularity: float) -> np.ndarray:
    c = (size - 1) / 2
    z, y, x = np.mgrid[0:size, 0:size, 0:size].astype(np.float32) - c
    dist = np.sqrt(z * z + y * y + x * x) + 1e-6
    u = np.stack([z, y, x]) / dist
    f = np.zeros_like(dist)
    for _ in range(5):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        proj = v[0] * u[0] + v[1] * u[1] + v[2] * u[2]
        f += np.sin(rng.uniform(3, 9) * np.pi * proj + rng.uniform(0, 2 * np.pi))
    f /= 5
    return dist <= radius * (1.0 + irregularity * f)


def make_patch(rng: np.random.Generator, malignant: bool, noise_sd: float, offset: float,
               size: int = PATCH_MM) -> tuple[np.ndarray, float]:
    radius = rng.uniform(3.5, 10.0) if malignant else rng.uniform(2.5, 8.0)
    irregular = rng.uniform(0.1, 0.4) if malignant else rng.uniform(0.0, 0.25)
    mask = _nodule_mask(rng, size, radius, irregular).astype(np.float32)
    soft = ndimage.gaussian_filter(mask, 0.7)
    tissue = rng.uniform(-50, 60) + offset
    patch = -850.0 + offset + soft * (tissue + 850.0)
    if not malignant and rng.random() < 0.3:  # central calcification
        core = _nodule_mask(rng, size, radius * 0.5, 0.0)
        patch[core] = rng.uniform(300, 600)
    # a vessel: a thin bright line through the patch
    axis = rng.integers(0, 3)
    a, b = rng.integers(4, size - 4, size=2)
    sl = [slice(None)] * 3
    others = [i for i in range(3) if i != axis]
    sl[others[0]], sl[others[1]] = slice(a, a + 2), slice(b, b + 2)
    patch[tuple(sl)] = np.maximum(patch[tuple(sl)], -100.0)
    patch += rng.normal(0, noise_sd, size=patch.shape)
    return patch.astype(np.float32), 2 * radius


def reader_ratings(rng: np.random.Generator, malignant: bool, diameter: float) -> tuple[int, ...]:
    n = int(rng.choice([1, 2, 3, 4], p=[0.1, 0.15, 0.3, 0.45]))
    latent = 2.0 + 2.0 * malignant + 0.08 * (diameter - 12) + rng.normal(0, 0.9)
    return tuple(int(np.clip(np.rint(latent + rng.normal(0, 0.8)), 1, 5)) for _ in range(n))


def make_cohort(out_dir: str | Path, n_patients: int = 60, seed: int = 13) -> Path:
    """Write ``patches/*.npy`` and ``manifest.jsonl`` into ``out_dir``. Returns the manifest path."""
    out_dir = Path(out_dir)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[NoduleRecord] = []
    for p in range(n_patients):
        pid = f"SYN-{p:04d}"
        noise_sd, offset = rng.uniform(15, 60), rng.uniform(-40, 40)
        for k in range(int(rng.integers(1, 5))):
            malignant = bool(rng.random() < 0.4)
            patch, diameter = make_patch(rng, malignant, noise_sd, offset)
            rel = f"patches/{pid}_n{k}.npy"
            np.save(out_dir / rel, patch)
            records.append(NoduleRecord(patient_id=pid, nodule_id=f"{pid}-n{k}",
                                        ratings=reader_ratings(rng, malignant, diameter),
                                        diameter_mm=round(diameter, 2), patch_file=rel))
    return write_manifest(out_dir / "manifest.jsonl", records)
