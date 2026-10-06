"""Streamlit front end (optional extra ``app``). Run: ``streamlit run src/lunglens/app/streamlit_app.py``.

Research use only. The app accepts a 1 mm HU patch (.npy) or a zipped DICOM CT series with a
nodule centre. It does not accept screenshots or photos.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st

from lunglens.cli import NOTICE, make_translator
from lunglens.config import Settings
from lunglens.i18n.languages import LANGUAGES, SOURCE_LANG
from lunglens.pipeline import Predictor
from lunglens.preprocess import extract_patch
from lunglens.qa import answer
from lunglens.report.builder import build_report
from lunglens.risk import Band
from lunglens.validation import load_dicom_series


@st.cache_resource
def _translator():
    return make_translator(Settings.from_env())


@st.cache_resource
def _predictor(path: str, low: float, high: float):
    return Predictor(path, low, high)


def main() -> None:
    s = Settings.from_env()
    st.set_page_config(page_title="lunglens (research)")
    st.title("lunglens - research nodule analysis")
    st.warning(NOTICE)
    lang = st.sidebar.selectbox("Language", list(LANGUAGES), format_func=lambda c: f"{LANGUAGES[c]} ({c})",
                                index=list(LANGUAGES).index(SOURCE_LANG))
    model_path = st.sidebar.text_input("Model file", str(s.model_dir / "lunglens-radiomics.joblib"))
    kind = st.radio("Input", ["HU patch (.npy)", "DICOM series (.zip)"])
    patch, position, warnings = None, "unknown", []
    if kind.startswith("HU"):
        up = st.file_uploader("32x32x32 patch, 1 mm voxels", type=["npy"])
        if up:
            patch, position = np.load(io.BytesIO(up.getvalue())), "patch centre"
    else:
        up = st.file_uploader("Zipped CT series", type=["zip"])
        center = st.text_input("Nodule centre z,y,x (voxel indices)")
        if up and center:
            with tempfile.TemporaryDirectory() as tmp:
                zipfile.ZipFile(io.BytesIO(up.getvalue())).extractall(tmp)
                folder = next((p.parent for p in Path(tmp).rglob("*.dcm")), Path(tmp))
                vol, spacing, _ = load_dicom_series(folder)
                patch = extract_patch(vol, spacing, tuple(float(v) for v in center.split(",")))
                position = f"voxel {center}"
    if patch is not None and Path(model_path).exists():
        try:
            res = _predictor(model_path, s.band_low, s.band_high).predict(patch)
        except ValueError as exc:
            res, warnings = {"probability": None, "band": "not_assessed", "diameter_mm": None}, [str(exc)]
        rep = build_report(band=Band(res["band"]), probability=res["probability"], diameter_mm=res["diameter_mm"],
                           position=position, low=s.band_low, high=s.band_high, warnings=warnings, lang=lang,
                           translator=_translator())
        st.markdown(rep.text().replace("\n\n", "\n\n---\n\n"))
        st.session_state["report"] = rep
    q = st.text_input("Question")
    if q:
        st.info(answer(q, lang, _translator())["answer"])


main()
