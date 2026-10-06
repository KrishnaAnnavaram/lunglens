import numpy as np
import pandas as pd
import pytest

from lunglens.data.manifest import load_table
from lunglens.data.splits import train_test_split_by_patient
from lunglens.data.synthetic import make_patch
from lunglens.evaluate import bootstrap_ci, evaluate, expected_calibration_error, patient_level, reader_reference
from lunglens.features import FEATURE_NAMES, PatchFeatures, patch_features
from lunglens.models.classical import build_pipeline
from lunglens.pipeline import Predictor, train_and_evaluate
from lunglens.preprocess import check_patch, extract_patch, resample_isotropic, window


def test_window_is_one_fixed_definition():
    hu = np.array([-2000.0, -1000.0, -300.0, 400.0, 3000.0])
    assert np.allclose(window(hu), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_resample_and_extract_patch():
    vol = np.full((20, 40, 40), -900.0, dtype=np.float32)
    vol[10, 20, 20] = 100.0
    iso = resample_isotropic(vol, (2.0, 1.0, 1.0))
    assert iso.shape == (40, 40, 40)
    patch = extract_patch(vol, (2.0, 1.0, 1.0), (10, 20, 20))
    assert patch.shape == (32, 32, 32) and patch[16, 16, 16] > -900
    edge = extract_patch(vol, (1.0, 1.0, 1.0), (0, 0, 0))
    assert edge[0, 0, 0] == -1000.0  # outside the scan = air
    with pytest.raises(ValueError):
        extract_patch(vol, (1.0, 1.0, 1.0), (500, 0, 0))


def test_check_patch_rejects_normalized_images():
    with pytest.raises(ValueError, match="scaled"):
        check_patch(np.random.default_rng(0).random((32, 32, 32)).astype(np.float32))
    with pytest.raises(ValueError):
        check_patch(np.zeros((16, 16, 16)))


def test_features_follow_size_and_calcification():
    rng = np.random.default_rng(0)
    small = np.mean([patch_features(make_patch(rng, False, 20, 0)[0])[1] for _ in range(10)])
    large = np.mean([patch_features(make_patch(rng, True, 20, 0)[0])[1] for _ in range(10)])
    assert large > small
    X = np.stack([make_patch(rng, True, 20, 0)[0] for _ in range(3)])
    assert PatchFeatures().fit_transform(X).shape == (3, len(FEATURE_NAMES))


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        build_pipeline("swin")


def test_ece_and_patient_level():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    df = pd.DataFrame({"patient_id": ["a", "a", "b", "b"], "label": y})
    py, pp = patient_level(df, np.array([0.1, 0.3, 0.2, 0.9]))
    assert list(py) == [0, 1] and list(pp) == [0.3, 0.9]


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    p = np.clip(y * 0.6 + rng.random(80) * 0.5, 0, 1)
    df = pd.DataFrame({"patient_id": [f"SYN-{i // 2:04d}" for i in range(80)], "label": y})
    m = evaluate(df, p, n_boot=200)
    lo, hi = m["nodule"]["auroc_ci"]
    assert lo <= m["nodule"]["auroc"] <= hi
    assert bootstrap_ci(df, p, "brier", 0.5, 50) == bootstrap_ci(df, p, "brier", 0.5, 50)


def test_reader_reference_leaves_the_reader_out():
    df = pd.DataFrame({"ratings": [[1, 1, 2], [5, 4, 5], [3]]})
    ref = reader_reference(df)
    assert ref["reader_auroc"] == 1.0 and ref["n_reads"] == 6


def test_models_beat_the_prior_on_held_out_patients(trained):
    _, res = trained
    assert res["prior"].metrics["nodule"]["auroc"] == 0.5
    assert res["radiomics"].metrics["nodule"]["auroc"] > 0.6
    assert res["radiomics"].metrics["n_patients"] < 48


def test_training_is_reproducible(cohort, tmp_path):
    a = train_and_evaluate(cohort, ["radiomics"], tmp_path / "a", seed=5, n_boot=20)[0]
    b = train_and_evaluate(cohort, ["radiomics"], tmp_path / "b", seed=5, n_boot=20)[0]
    assert a.metrics["nodule"]["auroc"] == b.metrics["nodule"]["auroc"]
    assert (tmp_path / "a" / "MODEL_CARD-radiomics.md").read_text().startswith("# Model card")


def test_predictor_round_trip(trained, cohort):
    out, _ = trained
    df, patches = load_table(cohort)
    _, te = train_test_split_by_patient(df, seed=5)
    res = Predictor(out / "lunglens-radiomics.joblib").predict(patches[te[0]])
    assert 0.0 <= res["probability"] <= 1.0 and res["band"] in ("lower", "intermediate", "higher")


def test_cnn3d_trains_with_torch(cohort):
    pytest.importorskip("torch")
    from lunglens.models.cnn3d import CNN3DClassifier, fit_calibrated_cnn

    df, patches = load_table(cohort)
    y, g = df["label"].to_numpy(), df["patient_id"].to_numpy()
    clf = CNN3DClassifier(epochs=1, width=4, seed=0).fit(patches[:16], y[:16])
    p = clf.predict_proba(patches[:4])
    assert p.shape == (4, 2) and np.allclose(p.sum(1), 1.0)
    cal = fit_calibrated_cnn(patches[:40], y[:40], g[:40], epochs=1, n_folds=2, seed=0)
    assert cal.predict_proba(patches[:3]).shape == (3, 2)
