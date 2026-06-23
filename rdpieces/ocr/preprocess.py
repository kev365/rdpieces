"""Preprocess a reconstructed scene for OCR.

Low-resolution UI text from 64px tiles needs help: composite transparent holes
onto white, integer-upscale (cubic), greyscale, then Otsu-binarise. Tunable; the
best upscale factor is validated empirically (see plan).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    weight_b = 0.0
    best_var = -1.0
    threshold = 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between > best_var:
            best_var = between
            threshold = t
    return threshold


def preprocess(image_rgba: np.ndarray, scale: int = 6) -> np.ndarray:
    """Return a binarised (0/255) greyscale image upscaled by ``scale``."""
    arr = np.asarray(image_rgba)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        composited = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    else:
        composited = arr[..., :3].astype(np.uint8)

    img = Image.fromarray(composited, "RGB")
    if scale and scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    gray = np.asarray(img.convert("L"))
    return np.where(gray > _otsu_threshold(gray), 255, 0).astype(np.uint8)
