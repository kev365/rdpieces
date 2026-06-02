"""Reporting outputs: hOCR, session timeline, and a PDF summary.

hOCR is synthesised from the consolidated word records (so it keeps the full-
upscale OCR quality rather than re-OCRing a shrunk montage). The timeline merges
cache-file mtimes with RDP event timestamps. The PDF needs the optional reportlab
extra.
"""

from __future__ import annotations

from datetime import datetime, timezone
from xml.sax.saxutils import escape

_HOCR_HEADER = (
    "<?xml version='1.0' encoding='UTF-8'?>\n"
    "<!DOCTYPE html PUBLIC '-//W3C//DTD XHTML 1.0 Transitional//EN' "
    "'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd'>\n"
    "<html xmlns='http://www.w3.org/1999/xhtml'>\n"
    "<head><meta name='ocr-system' content='rdpieces'/>"
    "<meta name='ocr-capabilities' content='ocr_page ocrx_word'/></head>\n<body>\n"
)


def words_to_hocr(words: list[dict], page_name: str, width: int, height: int) -> str:
    """Build an hOCR (XHTML) page from consolidated word records with bboxes."""
    lines = [_HOCR_HEADER]
    lines.append(
        f"<div class='ocr_page' title='image \"{escape(page_name)}\"; bbox 0 0 {width} {height}'>"
    )
    for i, w in enumerate(words):
        x, y, ww, hh = w["bbox"]
        conf = int(round(float(w.get("confidence", 0))))
        lines.append(
            f"<span class='ocrx_word' id='word_{i}' "
            f"title='bbox {x} {y} {x + ww} {y + hh}; x_wconf {conf}'>{escape(w['text'])}</span>"
        )
    lines.append("</div>\n</body></html>\n")
    return "\n".join(lines)


def _epoch(iso: str) -> float | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def build_timeline(cache_files: list[tuple[str, float]], events: list[dict]) -> list[dict]:
    """Merge cache-file mtimes and RDP event timestamps into one sorted timeline."""
    entries = []
    for path, mtime in cache_files:
        entries.append(
            {
                "epoch": mtime,
                "time": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                "kind": "cache_file_mtime",
                "detail": path,
            }
        )
    for e in events:
        ep = _epoch(e.get("time") or "")
        if ep is None:
            continue
        entries.append(
            {"epoch": ep, "time": e["time"], "kind": "rdp_event", "detail": f"Event {e.get('event_id')}"}
        )
    entries.sort(key=lambda x: x["epoch"])
    for x in entries:
        x.pop("epoch")
    return entries


def write_pdf_report(
    path: str, summary: dict, ocr_text: str = "", final_image_path: str | None = None
) -> None:
    """Write a one/two-page PDF summary (requires the optional reportlab extra)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdfcanvas

    width, height = letter
    c = pdfcanvas.Canvas(path, pagesize=letter)
    y = height - 54
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, y, "rdpieces RDP bitmap-cache reconstruction report")
    y -= 28
    c.setFont("Helvetica", 10)
    for key, value in summary.items():
        c.drawString(54, y, f"{key}: {value}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 54
            c.setFont("Helvetica", 10)

    if final_image_path:
        try:
            img = ImageReader(final_image_path)
            iw, ih = img.getSize()
            scale = min((width - 108) / iw, (height - 108) / ih)
            c.showPage()
            c.drawImage(img, 54, height - 54 - ih * scale, width=iw * scale, height=ih * scale)
        except Exception:
            pass

    if ocr_text:
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(54, height - 54, "Recovered text (OCR)")
        c.setFont("Helvetica", 9)
        ty = height - 74
        for line in ocr_text.splitlines():
            c.drawString(54, ty, line[:110])
            ty -= 12
            if ty < 60:
                c.showPage()
                c.setFont("Helvetica", 9)
                ty = height - 54
    c.save()
