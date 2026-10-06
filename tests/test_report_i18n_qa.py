import json
import re

import pytest

from lunglens.config import Settings
from lunglens.i18n.languages import LANGUAGES, UnknownLanguageError, check_code, validate_codes
from lunglens.i18n.translator import CachedTranslator, IdentityTranslator, TagTranslator, localize
from lunglens.qa import FALLBACK, answer, match
from lunglens.report.builder import UnsafeReportError, build_report, check_safe, templates
from lunglens.risk import Band, band_for
from lunglens.validation import check_input_path, check_metadata

CT = {"Modality": "CT", "BodyPartExamined": "CHEST", "SliceThickness": 1.25, "PixelSpacing": [0.7, 0.7],
      "RescaleSlope": 1, "RescaleIntercept": -1024}


def test_bands():
    assert band_for(0.05) is Band.LOWER and band_for(0.3) is Band.INTERMEDIATE and band_for(0.5) is Band.HIGHER
    assert band_for(None) is Band.NOT_ASSESSED
    with pytest.raises(ValueError):
        band_for(1.2)
    with pytest.raises(ValueError):
        band_for(0.3, low=0.6, high=0.5)


@pytest.mark.parametrize("band", list(Band))
def test_every_report_is_safe_and_starts_and_ends_with_the_notice(band):
    p = None if band is Band.NOT_ASSESSED else 0.42
    rep = build_report(band=band, probability=p, diameter_mm=7.0, position="x", low=0.15, high=0.5)
    keys = [k for k, _ in rep.lines]
    assert keys[1] == "safety" and keys[-1] == "safety"
    assert ("estimate" in keys) == (band is not Band.NOT_ASSESSED)
    check_safe(rep.text())


def test_safety_check_refuses_reassurance():
    with pytest.raises(UnsafeReportError):
        check_safe("Great job taking care of your health!")
    for text in templates().values():
        check_safe(text)


def test_language_codes_are_valid_flores_codes():
    assert len(LANGUAGES) >= 80
    assert all(re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", c) for c in LANGUAGES)
    for bad in ("nor_Latn", "msa_Latn", "fil_Latn", "tam_Sinh", "geo_Geor", "hye_Latn", "fas_Pers", "pus_Afgh"):
        with pytest.raises(UnknownLanguageError):
            check_code(bad)


def test_validate_codes_against_a_tokenizer():
    class Tok:
        unk_token_id = 3

        def convert_tokens_to_ids(self, t):
            return 3 if t == "kmr_Latn" else 100

    assert validate_codes(Tok()) == ["kmr_Latn"]


def test_cache_translates_each_string_once(tmp_path):
    inner = TagTranslator()
    tr = CachedTranslator(inner, tmp_path / "c.json")
    for _ in range(3):
        build_report(band=Band.LOWER, probability=0.1, diameter_mm=4.0, position="x", low=0.15, high=0.5,
                     lang="fra_Latn", translator=tr)
    first = sum(n for *_, n in inner.calls)
    assert first == len(set(k for k in templates()) & {"title", "safety", "intro", "estimate", "band.lower", "size",
                                                         "limits", "next"})
    inner2 = TagTranslator()
    tr2 = CachedTranslator(inner2, tmp_path / "c.json")
    tr2.translate(["The model estimates a malignancy likelihood of [0] for this nodule."], "eng_Latn", "fra_Latn")
    assert inner2.calls == []
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))


def test_localize_keeps_numbers_out_of_translation():
    out, ok = localize("Size {diameter} mm.", {"diameter": "7.0"}, "deu_Latn", TagTranslator())
    assert ok and out == "[deu_Latn] Size 7.0 mm."

    class Lossy:
        name = "lossy"

        def translate(self, texts, src, tgt):
            return ["Taille mm."]

    out, ok = localize("Size {diameter} mm.", {"diameter": "7.0"}, "fra_Latn", Lossy())
    assert not ok and out == "Size 7.0 mm."


def test_identity_translator_marks_report_as_not_translated():
    rep = build_report(band=Band.LOWER, probability=0.1, diameter_mm=4.0, position="x", low=0.15, high=0.5,
                       lang="spa_Latn", translator=IdentityTranslator())
    assert not rep.fully_translated
    assert build_report(band=Band.LOWER, probability=0.1, diameter_mm=4.0, position="x", low=0.15,
                        high=0.5).fully_translated


def test_faq_matching_ignores_case_and_punctuation():
    assert match("WHAT IS A LUNG NODULE???")[0].key == "nodule"
    assert match("how ACCURATE is this")[0].key == "accuracy"
    assert match("what is the capital of France")[0] is None


def test_question_is_translated_with_its_source_language():
    tr = TagTranslator()
    out = answer("[spa_Latn] What is a lung nodule?", "spa_Latn", tr)
    assert out["key"] == "nodule" and out["answer"].startswith("[spa_Latn] ")
    assert tr.calls[0][:2] == ("spa_Latn", "eng_Latn")
    assert answer("tell me a joke")["answer"] == FALLBACK


def test_input_validation():
    assert not check_input_path("scan.png").ok
    assert check_metadata(CT).ok
    assert not check_metadata({**CT, "Modality": "MR"}).ok
    assert not check_metadata({**CT, "BodyPartExamined": "HEAD"}).ok
    assert not check_metadata({**CT, "RescaleIntercept": None}).ok
    warn = check_metadata({**CT, "SliceThickness": 5.0, "BodyPartExamined": ""})
    assert warn.ok and len(warn.warnings) == 2


def test_settings(monkeypatch):
    monkeypatch.setenv("LUNGLENS_BAND_LOW", "0.2")
    assert Settings.from_env().band_low == 0.2
    monkeypatch.setenv("LUNGLENS_BAND_HIGH", "0.1")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("LUNGLENS_BAND_HIGH", "0.6")
    monkeypatch.setenv("LUNGLENS_TRANSLATOR", "google")
    with pytest.raises(ValueError):
        Settings.from_env()
