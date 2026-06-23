"""Tests for OCR preprocessing and the recognition pipeline.

The recognition tests inject a fake backend, so they validate the pipeline logic
(preprocess, confidence filtering, text/coordinate assembly) without needing the
Tesseract binary installed.
"""

from __future__ import annotations

import numpy as np

from rdpieces.ocr.engine import OcrWord, ocr_image, words_to_records, words_to_text
from rdpieces.ocr.preprocess import preprocess


def test_preprocess_upscales_and_binarizes():
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., 3] = 255
    img[:2, :, :3] = 30
    img[2:, :, :3] = 200
    out = preprocess(img, scale=3)
    assert out.shape == (12, 12)
    assert set(np.unique(out)).issubset({0, 255})


def test_preprocess_fills_transparent_with_white():
    transparent = np.zeros((4, 4, 4), np.uint8)  # alpha 0, rgb 0
    out = preprocess(transparent, scale=1)
    assert (out == 255).all()


def test_ocr_image_filters_empty_and_low_confidence():
    img = np.zeros((8, 8, 4), np.uint8)
    img[..., 3] = 255

    def fake_backend(gray):
        return [
            OcrWord("PING", 95.0, 10, 20, 40, 12),
            OcrWord("", 0.0, 0, 0, 0, 0),
            OcrWord("lo", 30.0, 5, 5, 8, 8),
        ]

    words = ocr_image(img, backend=fake_backend, min_confidence=50, scale=2)
    assert [w.text for w in words] == ["PING"]


def test_words_to_text_joins():
    assert words_to_text([OcrWord("a", 90, 0, 0, 1, 1), OcrWord("b", 90, 0, 0, 1, 1)]) == "a b"


def test_words_to_records_maps_bbox_back_to_scene_coords():
    recs = words_to_records([OcrWord("x", 90.0, 12, 24, 36, 12)], scale=6)
    assert recs[0]["text"] == "x"
    assert recs[0]["bbox"] == [2, 4, 6, 2]  # upscaled coords divided back by scale
    assert recs[0]["confidence"] == 90.0
