"""Report languages with their NLLB-200 (FLORES-200) codes.

The prototype used codes that NLLB does not know (for example ``nor_Latn``, ``msa_Latn``,
``tam_Sinh``), which the tokenizer turned into ``<unk>``. Every code here is a FLORES-200
code, and :func:`validate_codes` checks the list against a loaded tokenizer at start-up.
"""
from __future__ import annotations

import re

SOURCE_LANG = "eng_Latn"

LANGUAGES: dict[str, str] = {
    "eng_Latn": "English", "spa_Latn": "Spanish", "fra_Latn": "French", "deu_Latn": "German",
    "ita_Latn": "Italian", "por_Latn": "Portuguese", "nld_Latn": "Dutch", "swe_Latn": "Swedish",
    "dan_Latn": "Danish", "nob_Latn": "Norwegian Bokmal", "fin_Latn": "Finnish", "isl_Latn": "Icelandic",
    "pol_Latn": "Polish", "ces_Latn": "Czech", "slk_Latn": "Slovak", "slv_Latn": "Slovenian",
    "hrv_Latn": "Croatian", "srp_Cyrl": "Serbian", "bos_Latn": "Bosnian", "mkd_Cyrl": "Macedonian",
    "bul_Cyrl": "Bulgarian", "ron_Latn": "Romanian", "hun_Latn": "Hungarian", "ell_Grek": "Greek",
    "als_Latn": "Albanian (Tosk)", "lit_Latn": "Lithuanian", "lvs_Latn": "Latvian", "est_Latn": "Estonian",
    "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian", "bel_Cyrl": "Belarusian", "kat_Geor": "Georgian",
    "hye_Armn": "Armenian", "azj_Latn": "Azerbaijani (North)", "kaz_Cyrl": "Kazakh", "uzn_Latn": "Uzbek (North)",
    "kir_Cyrl": "Kyrgyz", "tgk_Cyrl": "Tajik", "tur_Latn": "Turkish", "arb_Arab": "Arabic (Modern Standard)",
    "heb_Hebr": "Hebrew", "pes_Arab": "Persian (Western)", "pbt_Arab": "Pashto (Southern)", "urd_Arab": "Urdu",
    "hin_Deva": "Hindi", "ben_Beng": "Bengali", "pan_Guru": "Punjabi (Gurmukhi)", "guj_Gujr": "Gujarati",
    "mar_Deva": "Marathi", "npi_Deva": "Nepali", "sin_Sinh": "Sinhala", "tam_Taml": "Tamil",
    "tel_Telu": "Telugu", "kan_Knda": "Kannada", "mal_Mlym": "Malayalam", "ory_Orya": "Odia",
    "asm_Beng": "Assamese", "mya_Mymr": "Burmese", "khm_Khmr": "Khmer", "lao_Laoo": "Lao",
    "tha_Thai": "Thai", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian", "zsm_Latn": "Malay (Standard)",
    "tgl_Latn": "Tagalog", "jav_Latn": "Javanese", "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)", "jpn_Jpan": "Japanese", "kor_Hang": "Korean",
    "khk_Cyrl": "Mongolian (Halh)", "amh_Ethi": "Amharic", "tir_Ethi": "Tigrinya", "som_Latn": "Somali",
    "swh_Latn": "Swahili", "hau_Latn": "Hausa", "yor_Latn": "Yoruba", "ibo_Latn": "Igbo", "zul_Latn": "Zulu",
    "xho_Latn": "Xhosa", "afr_Latn": "Afrikaans", "kin_Latn": "Kinyarwanda", "cat_Latn": "Catalan",
    "eus_Latn": "Basque", "glg_Latn": "Galician", "cym_Latn": "Welsh", "gle_Latn": "Irish",
    "mlt_Latn": "Maltese", "ltz_Latn": "Luxembourgish", "epo_Latn": "Esperanto", "hat_Latn": "Haitian Creole",
    "plt_Latn": "Malagasy (Plateau)", "ckb_Arab": "Kurdish (Central)", "kmr_Latn": "Kurdish (Kurmanji)",
}

_CODE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


class UnknownLanguageError(ValueError):
    pass


def check_code(code: str) -> str:
    if not _CODE.match(code) or code not in LANGUAGES:
        raise UnknownLanguageError(f"{code!r} is not a supported NLLB-200 language code; see `lunglens languages`")
    return code


def validate_codes(tokenizer, codes: list[str] | None = None) -> list[str]:
    """Return the codes that the tokenizer maps to the unknown token (should be empty)."""
    unk = tokenizer.unk_token_id
    return [c for c in (codes or list(LANGUAGES)) if tokenizer.convert_tokens_to_ids(c) == unk]
