"""Tests for rendering a placement grid into a canvas image."""

from __future__ import annotations

import numpy as np

from rdpieces.assembler import render
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


def test_render_handles_partial_edge_tiles():
    # Bottom row trimmed to height 56 (1080 mod 64), right column full.
    grid = {
        (0, 0): solid(64, 64, (10, 10, 10)),
        (1, 0): solid(64, 56, (20, 20, 20)),
    }
    canvas = render(grid)
    assert canvas.shape == (120, 64, 4)  # 64 + 56 rows
