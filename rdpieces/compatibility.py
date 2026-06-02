"""Directional compatibility between tiles (lower score = more likely adjacent).

Phase 2 baseline: sum-of-squared-differences on the shared border. Phase 3 fuses
MGC (gradient), NCC, SSIM and a context-histogram fallback, plus a weak
cache-index prior. Tiles whose shared edge has mismatched length can never be
neighbours in that direction, scoring infinity.
"""

from __future__ import annotations

import numpy as np

from .signatures import borders
from .tile import Tile

INF = float("inf")


def right_dissim(a: Tile, b: Tile) -> float:
    """Cost of placing ``b`` immediately to the right of ``a``."""
    if a.height != b.height:
        return INF
    ra = borders(a)["right"]
    lb = borders(b)["left"]
    return float(np.mean((ra - lb) ** 2))


def down_dissim(a: Tile, b: Tile) -> float:
    """Cost of placing ``b`` immediately below ``a``."""
    if a.width != b.width:
        return INF
    ba = borders(a)["bottom"]
    tb = borders(b)["top"]
    return float(np.mean((ba - tb) ** 2))
