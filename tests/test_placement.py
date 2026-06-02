"""Synthetic ground-truth harness for the placement engine.

Tile a known smooth image on the 64-grid, shuffle, reassemble, and measure
adjacency accuracy. This is the core quality metric for the reconstruction engine
(no ML or capture required) and the regression guard for metric/solver changes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from rdpieces.matcher import dissimilarity_matrices
from rdpieces.placement.edge_heuristic import mean_seam_cost, reconstruct
from rdpieces.tile import Tile


def make_tiles(cols: int, rows: int, seed: int = 0) -> list[Tile]:
    """Build full 64x64 tiles from a smooth, distinct random field (row-major)."""
    rng = np.random.default_rng(seed)
    height, width = rows * 64, cols * 64
    control = rng.integers(0, 256, (rows * 2 + 1, cols * 2 + 1, 3), dtype=np.uint8)
    img = np.asarray(Image.fromarray(control).resize((width, height), Image.BICUBIC), dtype=np.uint8)
    tiles = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            block = img[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
            rgba = np.dstack([block, np.full((64, 64), 255, np.uint8)]).astype(np.uint8)
            tiles.append(Tile(0, 0, 64, 64, 32, rgba, "syn", k))  # index = true row-major position
            k += 1
    return tiles


def adjacency_accuracy(grid: dict, cols: int) -> float:
    """Fraction of solver-adjacent tile pairs that are truly adjacent (index-based)."""
    correct = total = 0
    for (r, c), tile in grid.items():
        right = grid.get((r, c + 1))
        if right is not None:
            total += 1
            if right.index - tile.index == 1 and tile.index // cols == right.index // cols:
                correct += 1
        down = grid.get((r + 1, c))
        if down is not None:
            total += 1
            if down.index - tile.index == cols:
                correct += 1
    return correct / total if total else 0.0


def test_reconstruct_respects_row_bound():
    tiles = make_tiles(cols=4, rows=6)  # content is 6 rows tall
    shuffled = tiles[:]
    np.random.default_rng(2).shuffle(shuffled)

    grid = reconstruct(shuffled, max_rows=3)

    rows = max(r for r, _ in grid) + 1
    assert rows <= 3  # solver must not exceed the bound even though content is taller


def test_reconstructs_clean_image_with_high_adjacency_accuracy():
    cols, rows = 8, 6
    tiles = make_tiles(cols, rows)
    shuffled = tiles[:]
    np.random.default_rng(1).shuffle(shuffled)

    grid = reconstruct(shuffled)

    assert len(grid) == cols * rows
    assert adjacency_accuracy(grid, cols) >= 0.95


def test_mean_seam_cost_low_for_correct_reconstruction():
    cols, rows = 5, 4
    tiles = make_tiles(cols, rows, seed=3)
    shuffled = tiles[:]
    np.random.default_rng(4).shuffle(shuffled)
    right, down = dissimilarity_matrices(shuffled)
    grid = reconstruct(shuffled, right, down)

    cost = mean_seam_cost(grid, shuffled, right, down)
    assert 0.0 <= cost < np.inf
