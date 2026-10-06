from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from lunglens.data.lidc import cluster_to_record
from lunglens.data.manifest import ManifestError, load_table, read_manifest
from lunglens.data.schemas import NoduleRecord, binary_label, manifest_frame
from lunglens.data.splits import LeakageError, assert_no_patient_overlap, patient_folds, train_test_split_by_patient
from lunglens.data.synthetic import make_cohort


def _rec(ratings, pid="SYN-0001", nid="n1"):
    return NoduleRecord(patient_id=pid, nodule_id=nid, ratings=ratings, diameter_mm=6.0, patch_file="p.npy")


def test_rating_zero_is_rejected():
    with pytest.raises(ValidationError):
        _rec((0, 3))
    with pytest.raises(ValidationError):
        _rec((6,))


def test_consensus_label_rules():
    assert binary_label(4.0) == 1 and binary_label(5) == 1
    assert binary_label(2.0) == 0 and binary_label(1) == 0
    assert binary_label(3.0) is None and binary_label(3.5) is None
    assert _rec((2, 4, 5)).consensus == 4.0


def test_indeterminate_and_low_reader_nodules_are_dropped():
    recs = [_rec((1, 2), nid="a"), _rec((3, 3, 3), nid="b"), _rec((5,), nid="c"), _rec((4, 5, 5), nid="d")]
    df = manifest_frame(recs, min_readers=2)
    assert list(df["nodule_id"]) == ["a", "d"] and list(df["label"]) == [0, 1]


def test_patient_id_pattern():
    with pytest.raises(ValidationError):
        _rec((1,), pid="patient7")


def test_synthetic_cohort_is_seeded(tmp_path):
    a = read_manifest(make_cohort(tmp_path / "a", 6, seed=1))
    b = read_manifest(make_cohort(tmp_path / "b", 6, seed=1))
    assert a == b
    pa = np.load(tmp_path / "a" / a[0].patch_file)
    pb = np.load(tmp_path / "b" / b[0].patch_file)
    assert np.array_equal(pa, pb) and pa.shape == (32, 32, 32)


def test_manifest_errors_are_reported(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"patient_id": "SYN-0001", "nodule_id": "x", "ratings": [0], "diameter_mm": 3, "patch_file": "a"}\n')
    with pytest.raises(ManifestError, match="line 1"):
        read_manifest(p)


def test_split_has_no_patient_overlap(cohort):
    df, _ = load_table(cohort)
    tr, te = train_test_split_by_patient(df, seed=3)
    assert set(df["patient_id"].iloc[tr]).isdisjoint(df["patient_id"].iloc[te])
    for ftr, fva in patient_folds(df["label"].to_numpy()[tr], df["patient_id"].to_numpy()[tr], 4, 3):
        g = df["patient_id"].to_numpy()[tr]
        assert set(g[ftr]).isdisjoint(g[fva])


def test_leakage_guard_raises():
    with pytest.raises(LeakageError):
        assert_no_patient_overlap(pd.Series(["SYN-0001", "SYN-0002"]), pd.Series(["SYN-0002"]))


def test_lidc_cluster_becomes_one_record():
    cluster = [SimpleNamespace(malignancy=4, diameter=9.0), SimpleNamespace(malignancy=5, diameter=11.0)]
    rec = cluster_to_record("LIDC-IDRI-0001", "LIDC-IDRI-0001-s1-n0", cluster, "patches/x.npy")
    assert rec.ratings == (4, 5) and rec.diameter_mm == 10.0 and binary_label(rec.consensus) == 1


def test_lidc_extraction_without_pylidc_gives_a_clear_error(tmp_path):
    from lunglens.data.lidc import extract_lidc

    try:
        import pylidc  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="lidc"):
            extract_lidc(tmp_path)
    else:
        pytest.skip("pylidc is installed")
