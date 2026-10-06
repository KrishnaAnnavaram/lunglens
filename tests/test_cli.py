import json

import numpy as np
import pytest

from lunglens.cli import main


def test_languages(capsys):
    assert main(["languages"]) == 0
    assert "spa_Latn" in capsys.readouterr().out


def test_synth_train_predict_ask(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LUNGLENS_TRANSLATOR", raising=False)
    assert main(["synth", "--out", str(tmp_path / "d"), "--patients", "30"]) == 0
    assert main(["train", "--manifest", str(tmp_path / "d" / "manifest.jsonl"), "--models", "prior,size",
                 "--out", str(tmp_path / "m"), "--n-boot", "50"]) == 0
    assert "| size |" in capsys.readouterr().out
    patch = next((tmp_path / "d" / "patches").glob("*.npy"))
    assert main(["predict", "--model", str(tmp_path / "m" / "lunglens-size.joblib"), "--patch", str(patch),
                 "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["band"] in ("lower", "intermediate", "higher") and "not a diagnosis" in out["report"]
    assert main(["ask", "What is a nodule?"]) == 0
    assert '"nodule"' in capsys.readouterr().out


def test_predict_refuses_screenshots_and_wrong_modality(tmp_path, capsys):
    png = tmp_path / "slice.png"
    png.write_bytes(b"x")
    assert main(["predict", "--model", "none.joblib", "--patch", str(png), "--json"]) == 4
    out = json.loads(capsys.readouterr().out)
    assert out["band"] == "not_assessed" and out["probability"] is None
    np.save(tmp_path / "p.npy", np.full((32, 32, 32), -800.0, dtype=np.float32))
    (tmp_path / "meta.json").write_text(json.dumps({"Modality": "MR"}))
    assert main(["predict", "--model", "none.joblib", "--patch", str(tmp_path / "p.npy"), "--meta",
                 str(tmp_path / "meta.json"), "--json"]) == 4


def test_bad_language_code_exit_2(capsys):
    assert main(["ask", "hello", "--lang", "nor_Latn"]) == 2


def test_i18n_build_needs_nllb(monkeypatch):
    monkeypatch.delenv("LUNGLENS_TRANSLATOR", raising=False)
    assert main(["i18n-build", "--langs", "spa_Latn"]) == 2


def test_demo(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path), "--patients", "40"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "not a diagnosis" in out


def test_pdf_export(tmp_path):
    pytest.importorskip("reportlab")
    from lunglens.report.builder import build_report
    from lunglens.report.pdf import write_pdf
    from lunglens.risk import Band

    rep = build_report(band=Band.LOWER, probability=0.1, diameter_mm=4.0, position="x", low=0.15, high=0.5)
    assert write_pdf(rep, tmp_path / "r.pdf").stat().st_size > 0
