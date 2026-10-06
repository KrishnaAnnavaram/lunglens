"""Train, evaluate, save and load models. Write the model card."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from lunglens import __version__
from lunglens.data.manifest import load_table
from lunglens.data.splits import assert_no_patient_overlap, train_test_split_by_patient
from lunglens.evaluate import evaluate, reader_reference
from lunglens.features import patch_features
from lunglens.models.classical import CLASSICAL_KINDS, fit_calibrated, predict_proba
from lunglens.preprocess import check_patch
from lunglens.risk import Band, band_for

MODEL_KINDS = (*CLASSICAL_KINDS, "cnn3d")


@dataclass
class TrainResult:
    kind: str
    bundle_path: Path
    metrics: dict


def _fit(kind: str, X: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int, epochs: int):
    if kind == "cnn3d":
        from lunglens.models.cnn3d import fit_calibrated_cnn  # lazy: needs torch

        return fit_calibrated_cnn(X, y, g, epochs=epochs, seed=seed)
    return fit_calibrated(kind, X, y, g, seed=seed)


def train_and_evaluate(manifest: str | Path, kinds: list[str], out_dir: str | Path, *, seed: int = 13,
                       test_size: float = 0.25, min_readers: int = 1, epochs: int = 15,
                       threshold: float = 0.5, n_boot: int = 1000) -> list[TrainResult]:
    """Fit each model kind on the SAME patient-level split and evaluate on the held-out patients."""
    for k in kinds:
        if k not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {k!r}; expected one of {MODEL_KINDS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, patches = load_table(manifest, min_readers=min_readers)
    tr, te = train_test_split_by_patient(df, test_size=test_size, seed=seed)
    assert_no_patient_overlap(df["patient_id"].iloc[tr], df["patient_id"].iloc[te])
    y, g = df["label"].to_numpy(), df["patient_id"].to_numpy()
    test_df = df.iloc[te].reset_index(drop=True)
    reader = reader_reference(test_df)
    results = []
    for kind in kinds:
        model = _fit(kind, patches[tr], y[tr], g[tr], seed, epochs)
        p = predict_proba(model, patches[te])
        metrics = evaluate(test_df, p, threshold=threshold, n_boot=n_boot, seed=seed)
        metrics["reader_reference"] = reader
        info = {
            "kind": kind, "package_version": __version__, "sklearn_version": sklearn.__version__,
            "python": platform.python_version(), "seed": seed, "test_size": test_size, "min_readers": min_readers,
            "n_train_patients": int(df["patient_id"].iloc[tr].nunique()), "n_train_nodules": int(len(tr)),
            "n_test_patients": int(test_df["patient_id"].nunique()), "n_test_nodules": int(len(te)),
            "train_prevalence": float(y[tr].mean()), "manifest": str(manifest),
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = out_dir / f"lunglens-{kind}.joblib"
        joblib.dump({"model": model, "info": info, "metrics": metrics}, path)
        (out_dir / f"lunglens-{kind}.metrics.json").write_text(json.dumps({"info": info, "metrics": metrics},
                                                                         indent=2), encoding="utf-8")
        (out_dir / f"MODEL_CARD-{kind}.md").write_text(model_card(info, metrics), encoding="utf-8")
        results.append(TrainResult(kind, path, metrics))
    return results


def _fmt(v: float) -> str:
    return "n/a" if v != v else f"{v:.3f}"  # NaN check


def model_card(info: dict, m: dict) -> str:
    n = m["nodule"]
    lo, hi = n["auroc_ci"]
    return "\n".join([
        f"# Model card: lunglens `{info['kind']}`",
        "",
        "**Intended use.** Research only. The model estimates the malignancy likelihood of one marked lung "
        "nodule. It is not a medical device and it does not give a diagnosis.",
        "",
        "**Data.** "
        f"{info['n_train_nodules']} training nodules from {info['n_train_patients']} patients, "
        f"{info['n_test_nodules']} test nodules from {info['n_test_patients']} other patients "
        f"(manifest `{info['manifest']}`). Labels: median reader rating, 1-2 benign, 4-5 malignant, 3 excluded.",
        "",
        "**Results on held-out patients.**",
        "",
        "| Level | AUROC | AUPRC | Brier | ECE | Sensitivity | Specificity |",
        "|---|---|---|---|---|---|---|",
        f"| Nodule | {_fmt(n['auroc'])} [{_fmt(lo)}, {_fmt(hi)}] | {_fmt(n['auprc'])} | {_fmt(n['brier'])} | "
        f"{_fmt(n['ece'])} | {_fmt(n['sensitivity'])} | {_fmt(n['specificity'])} |",
        f"| Patient | {_fmt(m['patient']['auroc'])} | {_fmt(m['patient']['auprc'])} | {_fmt(m['patient']['brier'])} | "
        f"{_fmt(m['patient']['ece'])} | {_fmt(m['patient']['sensitivity'])} | {_fmt(m['patient']['specificity'])} |",
        "",
        f"Reader reference (one reader against the others): AUROC {_fmt(m['reader_reference']['reader_auroc'])}.",
        "",
        "**Limits.** No nodule detection: a person must give the nodule position. Calibration holds only for "
        "data like the training data. Reader ratings are subjective and are not pathology results. "
        "Check bias across scanners, sites and patient groups before any use.",
        "",
        f"Seed {info['seed']}, package {info['package_version']}, scikit-learn {info['sklearn_version']}.",
        "",
    ])


class Predictor:
    def __init__(self, bundle_path: str | Path, band_low: float = 0.15, band_high: float = 0.50):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.info = bundle["info"]
        self.band_low, self.band_high = band_low, band_high

    def predict(self, patch_hu: np.ndarray) -> dict:
        patch = check_patch(np.asarray(patch_hu, dtype=np.float32))
        p = float(predict_proba(self.model, patch[None])[0])
        diameter = float(patch_features(patch)[1])
        band: Band = band_for(p, self.band_low, self.band_high)
        return {"probability": p, "band": band.value, "diameter_mm": diameter, "model": self.info["kind"]}
