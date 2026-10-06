"""Read and write the nodule manifest (JSONL) and load its patches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from lunglens.data.schemas import NoduleRecord, manifest_frame
from lunglens.preprocess import check_patch


class ManifestError(ValueError):
    pass


def write_manifest(path: str | Path, records: list[NoduleRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json")) + "\n")
    return path


def read_manifest(path: str | Path) -> list[NoduleRecord]:
    path = Path(path)
    records, errors, seen = [], [], set()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = NoduleRecord.model_validate(json.loads(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                errors.append(f"line {n}: {exc}".splitlines()[0])
                continue
            if rec.nodule_id in seen:
                errors.append(f"line {n}: duplicate nodule_id {rec.nodule_id!r}")
                continue
            seen.add(rec.nodule_id)
            records.append(rec)
    if errors:
        raise ManifestError(f"{path}: {len(errors)} invalid line(s): " + "; ".join(errors[:5]))
    if not records:
        raise ManifestError(f"{path}: no records")
    return records


def load_table(path: str | Path, min_readers: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the labelled nodule table and the stacked HU patches, in the same order."""
    path = Path(path)
    df = manifest_frame(read_manifest(path), min_readers=min_readers)
    if df.empty:
        raise ManifestError("no labelled nodules after the indeterminate ones were removed")
    patches = np.stack([check_patch(np.load(path.parent / f)) for f in df["patch_file"]])
    return df.reset_index(drop=True), patches
