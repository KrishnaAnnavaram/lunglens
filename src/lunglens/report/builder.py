"""Build the patient report from fixed templates.

The report has no free-form generated text. Each sentence comes from
``templates.json`` and gets only numbers and codes. The safety notice is always first and
always last. A check refuses reassuring or diagnostic phrases.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from lunglens.i18n.languages import SOURCE_LANG, check_code
from lunglens.i18n.translator import IdentityTranslator, Translator, localize
from lunglens.risk import Band

BANNED_PHRASES = (
    "great job", "congratulations", "you have cancer", "you do not have cancer", "nothing to worry",
    "no need to see", "you are healthy", "diagnosis:", "definitely", "guaranteed",
)


@lru_cache(maxsize=1)
def templates() -> dict[str, str]:
    return json.loads(resources.files(__package__).joinpath("templates.json").read_text(encoding="utf-8"))


@dataclass
class Report:
    lang: str
    lines: list[tuple[str, str]] = field(default_factory=list)
    fully_translated: bool = True

    def text(self) -> str:
        return "\n\n".join(t for _, t in self.lines)


class UnsafeReportError(AssertionError):
    pass


def check_safe(text: str) -> None:
    low = text.lower()
    hits = [p for p in BANNED_PHRASES if p in low]
    if hits:
        raise UnsafeReportError(f"report contains banned phrases: {hits}")


def build_report(*, band: Band, probability: float | None, diameter_mm: float | None, position: str,
                 low: float, high: float, warnings: list[str] | None = None, lang: str = SOURCE_LANG,
                 translator: Translator | None = None) -> Report:
    check_code(lang)
    tr = translator or IdentityTranslator()
    t = templates()
    keys: list[tuple[str, dict[str, str]]] = [("title", {}), ("safety", {}), ("intro", {"position": position})]
    if band is not Band.NOT_ASSESSED and probability is not None:
        keys.append(("estimate", {"probability": f"{probability:.0%}"}))
    keys.append((f"band.{band.value}", {"low": f"{low:.0%}", "high": f"{high:.0%}"}))
    if diameter_mm is not None and band is not Band.NOT_ASSESSED:
        keys.append(("size", {"diameter": f"{diameter_mm:.1f}"}))
    if warnings:
        keys.append(("warnings", {"warnings": "; ".join(warnings)}))
    keys += [("limits", {}), ("next", {}), ("safety", {})]

    rep = Report(lang=lang)
    for key, values in keys:
        tmpl = t[key]
        used = {k: v for k, v in values.items() if "{" + k + "}" in tmpl}
        text, ok = localize(tmpl, used, lang, tr)
        rep.fully_translated &= ok
        rep.lines.append((key, text))
    check_safe(rep.text())
    return rep
