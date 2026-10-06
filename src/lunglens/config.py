"""Settings from environment variables. Every default works offline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("models")
    cache_dir: Path = Path("cache")
    translator: str = "none"  # none | nllb
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    seed: int = 13
    band_low: float = 0.15
    band_high: float = 0.50

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            model_dir=Path(_get("LUNGLENS_MODEL_DIR", "models")),
            cache_dir=Path(_get("LUNGLENS_CACHE_DIR", "cache")),
            translator=_get("LUNGLENS_TRANSLATOR", "none").lower(),
            nllb_model=_get("LUNGLENS_NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
            seed=int(_get("LUNGLENS_SEED", "13")),
            band_low=float(_get("LUNGLENS_BAND_LOW", "0.15")),
            band_high=float(_get("LUNGLENS_BAND_HIGH", "0.50")),
        )
        if s.translator not in ("none", "nllb"):
            raise ValueError("LUNGLENS_TRANSLATOR must be 'none' or 'nllb'")
        if not 0.0 < s.band_low < s.band_high < 1.0:
            raise ValueError("need 0 < LUNGLENS_BAND_LOW < LUNGLENS_BAND_HIGH < 1")
        return s
