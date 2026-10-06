"""The ``lunglens`` command. Research use only."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from lunglens.config import Settings
from lunglens.i18n.languages import LANGUAGES, SOURCE_LANG, UnknownLanguageError, check_code
from lunglens.i18n.translator import CachedTranslator, IdentityTranslator, TagTranslator, Translator
from lunglens.risk import Band

NOTICE = "Research use only. Not a diagnosis. A doctor must review every result."


def make_translator(settings: Settings) -> Translator:
    if settings.translator == "nllb":
        from lunglens.i18n.translator import NLLBTranslator

        return CachedTranslator(NLLBTranslator(settings.nllb_model), settings.cache_dir / "translations.json")
    return IdentityTranslator()


def _synth(a) -> int:
    from lunglens.data.synthetic import make_cohort

    path = make_cohort(a.out, a.patients, a.seed)
    print(f"wrote {path}")
    return 0


def _extract(a) -> int:
    from lunglens.data.lidc import extract_lidc

    pids = a.patients.split(",") if a.patients else None
    print(f"wrote {extract_lidc(a.out, pids, a.min_readers)}")
    return 0


def _train(a) -> int:
    from lunglens.pipeline import train_and_evaluate

    s = Settings.from_env()
    kinds = [k.strip() for k in a.models.split(",") if k.strip()]
    out = Path(a.out or s.model_dir)
    results = train_and_evaluate(a.manifest, kinds, out, seed=a.seed if a.seed is not None else s.seed,
                                 test_size=a.test_size, min_readers=a.min_readers, epochs=a.epochs,
                                 n_boot=a.n_boot)
    print("| Model | Nodule AUROC [95% CI] | AUPRC | Brier | ECE | Patient AUROC |")
    print("|---|---|---|---|---|---|")
    for r in results:
        n, p = r.metrics["nodule"], r.metrics["patient"]
        lo, hi = n["auroc_ci"]
        print(f"| {r.kind} | {n['auroc']:.3f} [{lo:.3f}, {hi:.3f}] | {n['auprc']:.3f} | {n['brier']:.3f} | "
              f"{n['ece']:.3f} | {p['auroc']:.3f} |")
    rr = results[0].metrics["reader_reference"]
    print(f"\nreader reference AUROC: {rr['reader_auroc']:.3f} ({int(rr['n_reads'])} reads); models in {out}")
    return 0


def _load_input(a) -> tuple[np.ndarray, str, list[str]]:
    from lunglens.preprocess import extract_patch
    from lunglens.validation import check_input_path, check_metadata

    if a.patch:
        chk = check_input_path(a.patch)
        if not chk.ok:
            raise ValueError("; ".join(chk.errors))
        warnings: list[str] = []
        if a.meta:
            mchk = check_metadata(json.loads(Path(a.meta).read_text(encoding="utf-8")))
            if not mchk.ok:
                raise ValueError("; ".join(mchk.errors))
            warnings = mchk.warnings
        return np.load(a.patch), "patch centre", warnings
    if a.dicom:
        from lunglens.validation import load_dicom_series

        chk = check_input_path(a.dicom)
        if not chk.ok:
            raise ValueError("; ".join(chk.errors))
        if not a.center:
            raise ValueError("--center z,y,x (voxel indices) is necessary with --dicom")
        center = tuple(float(v) for v in a.center.split(","))
        vol, spacing, meta = load_dicom_series(a.dicom)
        return extract_patch(vol, spacing, center), f"voxel {a.center}", check_metadata(meta).warnings
    raise ValueError("give --patch or --dicom")


def _predict(a) -> int:
    from lunglens.pipeline import Predictor
    from lunglens.report.builder import build_report

    s = Settings.from_env()
    lang = check_code(a.lang)
    tr = make_translator(s)
    try:
        patch, position, warnings = _load_input(a)
        res = Predictor(a.model, s.band_low, s.band_high).predict(patch)
    except ValueError as exc:
        res, position, warnings = {"probability": None, "band": Band.NOT_ASSESSED.value, "diameter_mm": None,
                                   "error": str(exc)}, "unknown", [str(exc)]
    rep = build_report(band=Band(res["band"]), probability=res["probability"], diameter_mm=res["diameter_mm"],
                       position=position, low=s.band_low, high=s.band_high, warnings=warnings, lang=lang,
                       translator=tr)
    if a.json:
        print(json.dumps({**res, "lang": lang, "fully_translated": rep.fully_translated, "report": rep.text()},
                         ensure_ascii=False, indent=2))
    else:
        print(rep.text())
        if lang != SOURCE_LANG and not rep.fully_translated:
            print("\n(note: one or more sentences are in English; set LUNGLENS_TRANSLATOR=nllb)")
    if a.pdf:
        from lunglens.report.pdf import write_pdf

        print(f"PDF: {write_pdf(rep, a.pdf)}")
    return 0 if res["band"] != Band.NOT_ASSESSED.value else 4


def _ask(a) -> int:
    from lunglens.qa import answer

    out = answer(a.question, check_code(a.lang), make_translator(Settings.from_env()))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _languages(_a) -> int:
    for code, name in LANGUAGES.items():
        print(f"{code}\t{name}")
    print(f"\n{len(LANGUAGES)} languages")
    return 0


def _i18n_build(a) -> int:
    from lunglens.qa import FALLBACK, FAQ
    from lunglens.report.builder import templates
    from lunglens.i18n.translator import _SLOT

    s = Settings.from_env()
    tr = make_translator(s)
    if not isinstance(tr, CachedTranslator):
        raise ValueError("set LUNGLENS_TRANSLATOR=nllb to build the translation cache")
    langs = [check_code(c) for c in a.langs.split(",")] if a.langs else [c for c in LANGUAGES if c != SOURCE_LANG]
    texts = []
    for t in templates().values():
        masked = t
        for i, n in enumerate(_SLOT.findall(t)):
            masked = masked.replace("{" + n + "}", f"[{i}]", 1)
        texts.append(masked)
    texts += [e.answer for e in FAQ] + [FALLBACK]
    for lang in langs:
        tr.translate(texts, SOURCE_LANG, lang)
        print(f"{lang}: {len(texts)} strings cached")
    return 0


def _demo(a) -> int:
    from lunglens.data.manifest import load_table
    from lunglens.data.splits import train_test_split_by_patient
    from lunglens.data.synthetic import make_cohort
    from lunglens.pipeline import Predictor, train_and_evaluate
    from lunglens.qa import answer
    from lunglens.report.builder import build_report

    root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="lunglens-demo-"))
    manifest = make_cohort(root / "data", a.patients, 13)
    results = train_and_evaluate(manifest, ["prior", "size", "radiomics"], root / "models", seed=13, n_boot=500)
    print("SYNTHETIC DATA - these numbers say nothing about real CT scans.\n")
    for r in results:
        n = r.metrics["nodule"]
        print(f"{r.kind:10s} nodule AUROC {n['auroc']:.3f} [{n['auroc_ci'][0]:.3f}, {n['auroc_ci'][1]:.3f}]  "
              f"Brier {n['brier']:.3f}  ECE {n['ece']:.3f}  patient AUROC {r.metrics['patient']['auroc']:.3f}")
    df, patches = load_table(manifest)
    _, te = train_test_split_by_patient(df, seed=13)
    pred = Predictor(root / "models" / "lunglens-radiomics.joblib")
    res = pred.predict(patches[te[0]])
    print(f"\nTest nodule {df['nodule_id'].iloc[te[0]]}: {json.dumps(res)}\n")
    rep = build_report(band=Band(res["band"]), probability=res["probability"], diameter_mm=res["diameter_mm"],
                       position="patch centre", low=0.15, high=0.50)
    print(rep.text())
    tag = TagTranslator()
    rep_es = build_report(band=Band(res["band"]), probability=res["probability"], diameter_mm=res["diameter_mm"],
                          position="patch centre", low=0.15, high=0.50, lang="spa_Latn", translator=tag)
    print("\nFake translation (TagTranslator, shows the flow only):\n" + rep_es.lines[3][1])
    print("\nQ&A: " + json.dumps(answer("what does the BAND mean??"), ensure_ascii=False))
    print(f"\n{NOTICE}\n(files in {root})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lunglens", description=NOTICE)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic nodule cohort (patches + manifest)")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--patients", type=int, default=120)
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(fn=_synth)

    s = sub.add_parser("extract-lidc", help="extract nodule patches from LIDC-IDRI (needs the lidc extra)")
    s.add_argument("--out", default="data/lidc")
    s.add_argument("--patients", help="comma-separated patient ids, for example LIDC-IDRI-0001,LIDC-IDRI-0002")
    s.add_argument("--min-readers", type=int, default=1)
    s.set_defaults(fn=_extract)

    s = sub.add_parser("train", help="train and evaluate models on one patient-level split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--models", default="prior,size,radiomics", help="comma list of prior,size,radiomics,cnn3d")
    s.add_argument("--out")
    s.add_argument("--seed", type=int)
    s.add_argument("--test-size", type=float, default=0.25)
    s.add_argument("--min-readers", type=int, default=1)
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(fn=_train)

    s = sub.add_parser("predict", help="estimate one nodule and print the report")
    s.add_argument("--model", required=True)
    s.add_argument("--patch", help="a 32x32x32 HU patch (.npy, 1 mm voxels)")
    s.add_argument("--meta", help="JSON with the DICOM header fields of the series (for --patch)")
    s.add_argument("--dicom", help="folder with one CT series (.dcm), needs the dicom extra")
    s.add_argument("--center", help="nodule centre as z,y,x voxel indices (for --dicom)")
    s.add_argument("--lang", default=SOURCE_LANG)
    s.add_argument("--json", action="store_true")
    s.add_argument("--pdf", help="also write a PDF (needs the pdf extra)")
    s.set_defaults(fn=_predict)

    s = sub.add_parser("ask", help="answer a follow-up question from the FAQ")
    s.add_argument("question")
    s.add_argument("--lang", default=SOURCE_LANG)
    s.set_defaults(fn=_ask)

    s = sub.add_parser("languages", help="list the supported NLLB-200 language codes")
    s.set_defaults(fn=_languages)

    s = sub.add_parser("i18n-build", help="translate all fixed texts once into the cache (needs NLLB)")
    s.add_argument("--langs", help="comma list of codes (default: all)")
    s.set_defaults(fn=_i18n_build)

    s = sub.add_parser("demo", help="offline demo on synthetic data")
    s.add_argument("--out")
    s.add_argument("--patients", type=int, default=120)
    s.set_defaults(fn=_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, UnknownLanguageError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
