"""Extract nodule patches from LIDC-IDRI with pylidc (optional extra ``lidc``).

For each scan, pylidc groups the reader annotations of one physical nodule into a cluster.
One cluster becomes one record: its patch (cut around the mean centroid) and the malignancy
rating of each reader. Slices without a nodule are never labelled.

Prerequisites: download LIDC-IDRI from TCIA and write ``~/.pylidcrc`` (``[dicom] path = ...``).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from lunglens.data.manifest import write_manifest
from lunglens.data.schemas import NoduleRecord
from lunglens.preprocess import extract_patch


def _pylidc():
    try:
        import pylidc  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError('LIDC extraction needs the "lidc" extra: pip install -e ".[lidc]"') from exc
    return pylidc


def cluster_to_record(patient_id: str, nodule_id: str, cluster: list, patch_file: str) -> NoduleRecord:
    """Build a record from one pylidc annotation cluster (works on any objects with the same fields)."""
    ratings = tuple(int(a.malignancy) for a in cluster)
    diameter = float(np.mean([a.diameter for a in cluster]))
    return NoduleRecord(patient_id=patient_id, nodule_id=nodule_id, ratings=ratings,
                        diameter_mm=round(max(diameter, 0.1), 2), patch_file=patch_file)


def extract_lidc(out_dir: str | Path, patient_ids: Iterable[str] | None = None, min_readers: int = 1) -> Path:
    pl = _pylidc()
    out_dir = Path(out_dir)
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    query = pl.query(pl.Scan)
    if patient_ids is not None:
        query = query.filter(pl.Scan.patient_id.in_(list(patient_ids)))
    records: list[NoduleRecord] = []
    for scan in query:
        clusters = [c for c in scan.cluster_annotations() if len(c) >= min_readers]
        if not clusters:
            continue
        vol_yxz = scan.to_volume(verbose=False)
        vol = np.transpose(vol_yxz, (2, 0, 1)).astype(np.float32)  # (z, y, x)
        spacing = (float(scan.slice_spacing), float(scan.pixel_spacing), float(scan.pixel_spacing))
        for k, cluster in enumerate(clusters):
            cy, cx, cz = np.mean([a.centroid for a in cluster], axis=0)
            nid = f"{scan.patient_id}-s{scan.id}-n{k}"
            rel = f"patches/{nid}.npy"
            np.save(out_dir / rel, extract_patch(vol, spacing, (cz, cy, cx)))
            records.append(cluster_to_record(scan.patient_id, nid, cluster, rel))
    return write_manifest(out_dir / "manifest.jsonl", records)
