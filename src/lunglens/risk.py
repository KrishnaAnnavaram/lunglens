"""Map a calibrated probability to a band. The bands are for research display only.

The thresholds are not clinically validated. They are configuration
(``LUNGLENS_BAND_LOW``, ``LUNGLENS_BAND_HIGH``), and the band names describe the model
estimate, not the patient.
"""
from __future__ import annotations

from enum import Enum


class Band(str, Enum):
    LOWER = "lower"
    INTERMEDIATE = "intermediate"
    HIGHER = "higher"
    NOT_ASSESSED = "not_assessed"


def band_for(probability: float | None, low: float = 0.15, high: float = 0.50) -> Band:
    if not 0.0 < low < high < 1.0:
        raise ValueError("need 0 < low < high < 1")
    if probability is None:
        return Band.NOT_ASSESSED
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability < low:
        return Band.LOWER
    if probability < high:
        return Band.INTERMEDIATE
    return Band.HIGHER
