"""PDF export of a report (optional extra ``pdf``, ReportLab)."""
from __future__ import annotations

from pathlib import Path

from lunglens.report.builder import Report


def write_pdf(report: Report, path: str | Path) -> Path:
    try:
        from reportlab.lib.pagesizes import A4  # noqa: PLC0415
        from reportlab.lib.styles import getSampleStyleSheet  # noqa: PLC0415
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError('PDF export needs the "pdf" extra: pip install -e ".[pdf]"') from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for key, text in report.lines:
        story.append(Paragraph(text, styles["Title"] if key == "title" else styles["BodyText"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path
