"""Tests for rendering a placement grid into a canvas image."""

from __future__ import annotations

import numpy as np

from rdpieces.assembler import montage, render
from rdpieces.tile import Tile


def solid(w: int, h: int, rgb: tuple[int, int, int]) -> Tile:
    px = np.zeros((h, w, 4), np.uint8)
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = (*rgb, 255)
    return Tile(0, 0, w, h, 32, px, "f", 0)


def test_render_places_tiles_at_grid_positions():
    grid = {
        (0, 0): solid(64, 64, (255, 0, 0)),
        (0, 1): solid(64, 64, (0, 255, 0)),
        (1, 0): solid(64, 64, (0, 0, 255)),
        (1, 1): solid(64, 64, (255, 255, 0)),
    }
    canvas = render(grid)
    assert canvas.shape == (128, 128, 4)
    assert tuple(canvas[0, 0, :3]) == (255, 0, 0)      # top-left
    assert tuple(canvas[0, 64, :3]) == (0, 255, 0)     # top-right
    assert tuple(canvas[64, 0, :3]) == (0, 0, 255)     # bottom-left
    assert tuple(canvas[64, 64, :3]) == (255, 255, 0)  # bottom-right


def test_montage_stacks_canvases_vertically():
    a = np.zeros((10, 20, 4), np.uint8); a[..., :3] = 1; a[..., 3] = 255
    b = np.zeros((5, 8, 4), np.uint8); b[..., :3] = 2; b[..., 3] = 255

    out = montage([a, b], padding=3)

    assert out.shape == (10 + 3 + 5, 20, 4)  # width = max(20, 8); height = 10 + pad + 5
    assert tuple(out[0, 0]) == (1, 1, 1, 255)    # first canvas at top
    assert tuple(out[13, 0]) == (2, 2, 2, 255)   # second below the padding gap


def test_montage_empty_returns_empty():
    assert montage([]).shape == (0, 0, 4)


def test_render_handles_partial_edge_tiles():
    # Bottom row trimmed to height 56 (1080 mod 64), right column full.
    grid = {
        (0, 0): solid(64, 64, (10, 10, 10)),
        (1, 0): solid(64, 56, (20, 20, 20)),
    }
    canvas = render(grid)
    assert canvas.shape == (120, 64, 4)  # 64 + 56 rows
