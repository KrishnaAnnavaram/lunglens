"""Translators behind one small interface, with a persistent cache.

* :class:`IdentityTranslator` - offline. Returns English and says so.
* :class:`TagTranslator` - offline fake for tests and the demo. Prefixes the target code.
* :class:`NLLBTranslator` - NLLB-200 with ``transformers`` (``translate`` extra). It sets
  ``tokenizer.src_lang`` for every call, so a question in Spanish is encoded as Spanish.
* :class:`CachedTranslator` - wraps any translator. A string is translated once per language
  and then read from a JSON cache, so static report text is not translated on each request.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

from lunglens.i18n.languages import SOURCE_LANG, check_code, validate_codes


class Translator(Protocol):
    name: str

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]: ...


class IdentityTranslator:
    name = "identity"
    is_identity = True

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        check_code(src)
        check_code(tgt)
        return list(texts)


class TagTranslator:
    """A fake that marks the target language. It records each call for tests."""

    name = "tag"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        check_code(src)
        check_code(tgt)
        self.calls.append((src, tgt, len(texts)))
        if src == tgt:
            return list(texts)
        if tgt == SOURCE_LANG:  # "back" translation for questions: drop the tag
            return [re.sub(r"^\[[a-z]{3}_[A-Z][a-z]{3}\] ", "", t) for t in texts]
        return [f"[{tgt}] {t}" for t in texts]


class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", max_new_tokens: int = 256):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError('NLLB needs the "translate" extra: pip install -e ".[translate]"') from exc
        self.name = f"nllb:{model_name}"
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._max = max_new_tokens
        bad = validate_codes(self._tok)
        if bad:
            raise ValueError(f"the tokenizer does not know these language codes: {bad}")

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        check_code(src)
        check_code(tgt)
        if src == tgt or not texts:
            return list(texts)
        self._tok.src_lang = src
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        out = self._model.generate(**enc, forced_bos_token_id=self._tok.convert_tokens_to_ids(tgt),
                                   max_new_tokens=self._max)
        return self._tok.batch_decode(out, skip_special_tokens=True)


class CachedTranslator:
    def __init__(self, inner: Translator, cache_file: str | Path | None = None):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.is_identity = getattr(inner, "is_identity", False)
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: dict[str, str] = {}
        if self.cache_file and self.cache_file.exists():
            self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))

    def _key(self, text: str, src: str, tgt: str) -> str:
        return hashlib.sha256(f"{self.inner.name}|{src}|{tgt}|{text}".encode()).hexdigest()

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == tgt:
            check_code(src)
            return list(texts)
        keys = [self._key(t, src, tgt) for t in texts]
        missing = sorted({(k, t) for k, t in zip(keys, texts) if k not in self._cache})
        if missing:
            done = self.inner.translate([t for _, t in missing], src, tgt)
            for (k, _), d in zip(missing, done):
                self._cache[k] = d
            self.save()
        return [self._cache[k] for k in keys]

    def save(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self._cache, ensure_ascii=False, indent=0), encoding="utf-8")


_SLOT = re.compile(r"\{([a-z_]+)\}")


def localize(template: str, values: dict[str, str], tgt: str, translator: Translator) -> tuple[str, bool]:
    """Translate a template that has ``{slots}``, then fill the slots.

    The slots become numbered markers before translation. If the translation loses a marker,
    the English sentence is used and the second return value is False.
    """
    names = _SLOT.findall(template)
    masked = template
    for i, n in enumerate(names):
        masked = masked.replace("{" + n + "}", f"[{i}]", 1)
    if tgt == SOURCE_LANG or getattr(translator, "is_identity", False):
        out, ok = masked, tgt == SOURCE_LANG
    else:
        translated = translator.translate([masked], SOURCE_LANG, tgt)[0]
        if all(f"[{i}]" in translated for i in range(len(names))):
            out, ok = translated, True
        else:
            out, ok = masked, False
    for i, n in enumerate(names):
        out = out.replace(f"[{i}]", values[n], 1)
    return out, ok
