"""Follow-up questions: a fixed FAQ with normalized matching, in any supported language.

There is no generative or extractive model here. The prototype used a BioBERT checkpoint
without a question-answering head, so its answers were random spans. Each answer here is a
fixed, reviewed text. A question with no good match gets a fixed fallback answer.

Matching: casefold and remove punctuation on both sides, so the case of a stored question
does not matter. The score is the mean of the content-word overlap (Jaccard, without stop
words) and the ``difflib`` ratio, against every variant of every FAQ entry.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from lunglens.i18n.languages import SOURCE_LANG, check_code
from lunglens.i18n.translator import IdentityTranslator, Translator

MATCH_THRESHOLD = 0.6
STOP_WORDS = frozenset("a an the is are am do does did what how who which can my me i of to in on for this that it be".split())


@dataclass(frozen=True)
class FaqEntry:
    key: str
    variants: tuple[str, ...]
    answer: str


FAQ: tuple[FaqEntry, ...] = (
    FaqEntry("estimate", ("What does my score mean?", "What is the malignancy likelihood?", "What is this percentage?",
                          "What does the probability mean?"),
             "The percentage is the model estimate that a nodule like this one is malignant. It is not a diagnosis."),
    FaqEntry("band", ("What does the band mean?", "What is the higher band?", "What is the lower band?",
                      "Is my result high or low?"),
             "The band groups the model estimate into lower, intermediate or higher. The limits are research settings and are not clinical rules."),
    FaqEntry("nodule", ("What is a lung nodule?", "What is a nodule?"),
             "A lung nodule is a small, round area in the lung on a CT scan. Most lung nodules are not cancer."),
    FaqEntry("next", ("What should I do next?", "What are the next steps?", "Do I need more tests?"),
             "Talk with your doctor. Your doctor decides on more tests or a follow-up scan."),
    FaqEntry("accuracy", ("How accurate is this?", "Can the model be wrong?", "How reliable is the result?"),
             "The model can be wrong. It was tested on research data only, and a doctor must review every result."),
    FaqEntry("size", ("How big is the nodule?", "What is the size of the nodule?"),
             "The report gives the measured size in millimetres. Your doctor compares it with earlier scans."),
    FaqEntry("data", ("Is my scan stored?", "What happens to my data?", "Who can see my scan?"),
             "This tool does not upload your scan. The files stay on the computer that runs the tool."),
)
FALLBACK = "I cannot answer this question. Ask your doctor or your radiologist."


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    ta, tb = set(a.split()) - STOP_WORDS, set(b.split()) - STOP_WORDS
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return (jac + difflib.SequenceMatcher(None, a, b).ratio()) / 2


def match(question: str) -> tuple[FaqEntry | None, float]:
    q = normalize(question)
    if not q:
        return None, 0.0
    best, score = None, 0.0
    for entry in FAQ:
        for v in entry.variants:
            s = _similarity(q, normalize(v))
            if s > score:
                best, score = entry, s
    return (best, score) if score >= MATCH_THRESHOLD else (None, score)


def answer(question: str, lang: str = SOURCE_LANG, translator: Translator | None = None) -> dict:
    """Translate the question to English (with the right source language), match, translate back."""
    check_code(lang)
    tr = translator or IdentityTranslator()
    q_en = tr.translate([question], lang, SOURCE_LANG)[0] if lang != SOURCE_LANG else question
    entry, score = match(q_en)
    text = entry.answer if entry else FALLBACK
    out = tr.translate([text], SOURCE_LANG, lang)[0] if lang != SOURCE_LANG else text
    return {"key": entry.key if entry else None, "score": round(score, 3), "answer": out}
