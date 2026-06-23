"""OCR recognition over preprocessed scenes.

The backend is injectable (default: Tesseract via pytesseract) so the pipeline is
testable without the Tesseract binary. Coordinates are mapped back from the
upscaled OCR image to scene pixel coordinates.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .preprocess import preprocess

_COMMON_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def _locate_tesseract() -> None:
    """If tesseract isn't on PATH, point pytesseract at a common install location."""
    if shutil.which("tesseract"):
        return
    try:
        import pytesseract
    except Exception:
        return
    for path in _COMMON_TESSERACT_PATHS:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


@dataclass(slots=True)
class OcrWord:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


Backend = Callable[[np.ndarray], list[OcrWord]]


def tesseract_available() -> bool:
    _locate_tesseract()
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _tesseract_backend(languages: str = "eng") -> Backend:
    def backend(gray: np.ndarray) -> list[OcrWord]:
        import pytesseract
        from PIL import Image

        data = pytesseract.image_to_data(
            Image.fromarray(gray), lang=languages, output_type=pytesseract.Output.DICT
        )
        words = []
        for text, conf, x, y, w, h in zip(
            data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
        ):
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                confidence = -1.0
            words.append(OcrWord(text, confidence, int(x), int(y), int(w), int(h)))
        return words

    return backend


def ocr_image(
    image_rgba: np.ndarray,
    *,
    backend: Backend | None = None,
    languages: str = "eng",
    min_confidence: float = 0.0,
    scale: int = 6,
) -> list[OcrWord]:
    """Preprocess then recognise; returns non-empty words above the confidence floor."""
    gray = preprocess(image_rgba, scale)
    backend = backend or _tesseract_backend(languages)
    words = backend(gray)
    return [w for w in words if w.text.strip() and w.confidence >= min_confidence]


def words_to_text(words: list[OcrWord]) -> str:
    return " ".join(w.text for w in words if w.text.strip())


def words_to_records(words: list[OcrWord], scale: int = 6) -> list[dict]:
    """Word records with bounding boxes mapped back to scene coordinates."""
    return [
        {
            "text": w.text,
            "confidence": w.confidence,
            "bbox": [w.left // scale, w.top // scale, w.width // scale, w.height // scale],
        }
        for w in words
    ]
