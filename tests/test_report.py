"""Tests for reporting: hOCR, session timeline, and PDF."""

from __future__ import annotations

from rdpieces.report import build_timeline, words_to_hocr, write_pdf_report


def test_words_to_hocr_emits_page_and_word_bboxes():
    words = [
        {"text": "ipconfig", "confidence": 91.0, "bbox": [10, 20, 40, 12]},
        {"text": "Ethernet0", "confidence": 88.0, "bbox": [10, 40, 60, 12]},
    ]
    hocr = words_to_hocr(words, page_name="final_reconstruction.png", width=200, height=100)

    assert "ocr_page" in hocr
    assert "bbox 0 0 200 100" in hocr  # page bbox
    assert "ocrx_word" in hocr
    assert "bbox 10 20 50 32" in hocr  # x0 y0 x1 y1 from [10,20,40,12]
    assert "ipconfig" in hocr and "Ethernet0" in hocr


def test_words_to_hocr_escapes_xml_special_chars():
    hocr = words_to_hocr([{"text": "a<b>&c", "confidence": 50, "bbox": [0, 0, 1, 1]}], "p.png", 10, 10)
    assert "a&lt;b&gt;&amp;c" in hocr
    assert "<b>" not in hocr.split("ocrx_word", 1)[1][:200]


def test_build_timeline_merges_and_sorts_by_time():
    cache_files = [("Cache0000.bin", 1590689977.0)]  # 2020-05-28
    events = [
        {"event_id": 168, "time": "2019-01-01T00:00:00Z"},
        {"event_id": 169, "time": "2021-01-01T00:00:00Z"},
    ]
    timeline = build_timeline(cache_files, events)

    assert [e["kind"] for e in timeline] == ["rdp_event", "cache_file_mtime", "rdp_event"]
    assert timeline[0]["detail"] == "Event 168"
    assert "cache_file_mtime" in {e["kind"] for e in timeline}


def test_write_pdf_report_creates_a_pdf(tmp_path):
    pdf = tmp_path / "report.pdf"
    write_pdf_report(
        str(pdf),
        summary={"tool": "rdpieces", "scenes_rendered": 3, "resolution": "1920x1080"},
        ocr_text="ipconfig\nEthernet0",
    )
    data = pdf.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 200
