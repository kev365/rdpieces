"""Render a placement grid ({(row, col): Tile}) into a single canvas image.

Column widths and row heights are taken from the tiles themselves so trimmed
edge tiles (partial right column / bottom row) land at the correct offsets.
Empty cells stay transparent (holes are surfaced, not hidden).
"""

from __future__ import annotations

import numpy as np

from .tile import Tile


def render(grid: dict[tuple[int, int], Tile]) -> np.ndarray:
    """Return an (H, W, 4) RGBA canvas for the placement."""
    if not grid:
        return np.zeros((0, 0, 4), dtype=np.uint8)

    rows = max(r for r, _ in grid) + 1
    cols = max(c for _, c in grid) + 1
    col_w = [0] * cols
    row_h = [0] * rows
    for (r, c), tile in grid.items():
        col_w[c] = max(col_w[c], tile.width)
        row_h[r] = max(row_h[r], tile.height)

    x = np.cumsum([0, *col_w])
    y = np.cumsum([0, *row_h])
    canvas = np.zeros((int(y[rows]), int(x[cols]), 4), dtype=np.uint8)
    for (r, c), tile in grid.items():
        canvas[y[r] : y[r] + tile.height, x[c] : x[c] + tile.width] = tile.pixels
    return canvas
