"""Input validation from DICOM metadata, not from what the picture looks like.

The prototype accepted any PNG or JPG and guessed with image classifiers whether it was a
lung CT. Here only a CT series is accepted, and the decision uses the DICOM header.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CHEST_PARTS = {"CHEST", "LUNG", "THORAX", "CHEST_TO_PELVIS"}
MAX_SLICE_THICKNESS_MM = 3.0


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_input_path(path: str | Path) -> ValidationResult:
    p = Path(path)
    if p.suffix.lower() in IMAGE_SUFFIXES:
        return ValidationResult(False, [f"{p.name}: screenshots and photos are not accepted; give a DICOM CT series"])
    if not p.exists():
        return ValidationResult(False, [f"{p}: not found"])
    return ValidationResult(True)


def check_metadata(meta: dict) -> ValidationResult:
    """Check the DICOM header fields of a series (as a plain dict)."""
    res = ValidationResult(True)
    modality = str(meta.get("Modality", "")).upper()
    if modality != "CT":
        res.errors.append(f"Modality is {modality or 'missing'}; only CT is accepted")
    part = str(meta.get("BodyPartExamined", "")).upper()
    if not part:
        res.warnings.append("BodyPartExamined is missing; check that the series covers the chest")
    elif part not in CHEST_PARTS:
        res.errors.append(f"BodyPartExamined is {part}; only chest CT is accepted")
    thickness = meta.get("SliceThickness")
    if thickness is None:
        res.warnings.append("SliceThickness is missing")
    elif float(thickness) > MAX_SLICE_THICKNESS_MM:
        res.warnings.append(f"SliceThickness {float(thickness)} mm is above {MAX_SLICE_THICKNESS_MM} mm; small nodules can be missed")
    for key in ("PixelSpacing", "RescaleSlope", "RescaleIntercept"):
        if meta.get(key) is None:
            res.errors.append(f"{key} is missing; Hounsfield units cannot be calculated")
    res.ok = not res.errors
    return res


def load_dicom_series(folder: str | Path) -> tuple[np.ndarray, tuple[float, float, float], dict]:
    """Read a DICOM series into a (z, y, x) HU volume. Needs the ``dicom`` extra (pydicom)."""
    try:
        import pydicom  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError('reading DICOM needs the "dicom" extra: pip install -e ".[dicom]"') from exc
    files = sorted(Path(folder).glob("*.dcm"))
    if not files:
        raise ValueError(f"{folder}: no .dcm files")
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    first = slices[0]
    meta = {k: getattr(first, k, None) for k in
            ("Modality", "BodyPartExamined", "SliceThickness", "PixelSpacing", "RescaleSlope", "RescaleIntercept")}
    check = check_metadata(meta)
    if not check.ok:
        raise ValueError("; ".join(check.errors))
    vol = np.stack([s.pixel_array.astype(np.float32) * float(s.RescaleSlope) + float(s.RescaleIntercept)
                    for s in slices])
    zs = [float(s.ImagePositionPatient[2]) for s in slices]
    dz = float(np.median(np.diff(zs))) if len(zs) > 1 else float(first.SliceThickness)
    dy, dx = (float(v) for v in first.PixelSpacing)
    return vol, (abs(dz), dy, dx), meta
