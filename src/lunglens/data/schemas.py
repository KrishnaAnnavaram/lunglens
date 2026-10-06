"""Nodule records and the consensus label rules.

One record is one nodule (a cluster of reader annotations), not one slice. The label comes
from the ratings of that nodule only. LIDC malignancy ratings use a 1-5 scale:
1-2 = benign side, 3 = indeterminate, 4-5 = malignant side. There is no 0.
"""
from __future__ import annotations

import statistics

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

PATIENT_ID_PATTERN = r"^(LIDC-IDRI-\d{4}|SYN-\d{4})$"
MANIFEST_COLUMNS = ("patient_id", "nodule_id", "ratings", "diameter_mm", "patch_file")


class NoduleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str = Field(pattern=PATIENT_ID_PATTERN)
    nodule_id: str = Field(min_length=1)
    ratings: tuple[int, ...] = Field(min_length=1, max_length=4, description="one malignancy rating per reader")
    diameter_mm: float = Field(gt=0, lt=100)
    patch_file: str = Field(min_length=1, description="path of the .npy patch, relative to the manifest")

    @field_validator("ratings")
    @classmethod
    def _ratings(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        bad = [r for r in v if r not in (1, 2, 3, 4, 5)]
        if bad:
            raise ValueError(f"ratings must be in 1..5 (got {bad}); 'no rating' is not a rating")
        return v

    @property
    def consensus(self) -> float:
        return float(statistics.median(self.ratings))

    @property
    def n_readers(self) -> int:
        return len(self.ratings)


def binary_label(consensus: float) -> int | None:
    """1 = malignant (median >= 4), 0 = benign (median <= 2), None = indeterminate (excluded)."""
    if consensus >= 4:
        return 1
    if consensus <= 2:
        return 0
    return None


def manifest_frame(records: list[NoduleRecord], min_readers: int = 1) -> pd.DataFrame:
    """A table with one row per labelled nodule. Indeterminate nodules are dropped."""
    rows = []
    for r in records:
        lab = binary_label(r.consensus)
        if lab is None or r.n_readers < min_readers:
            continue
        rows.append({"patient_id": r.patient_id, "nodule_id": r.nodule_id, "consensus": r.consensus,
                     "n_readers": r.n_readers, "ratings": list(r.ratings), "diameter_mm": r.diameter_mm,
                     "patch_file": r.patch_file, "label": lab})
    return pd.DataFrame(rows)
