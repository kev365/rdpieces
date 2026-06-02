"""Synthetic ground-truth harness for the placement engine.

Tile a known smooth image on the 64-grid, shuffle, reassemble, and measure
adjacency accuracy. This is the core quality metric for the reconstruction engine
(no ML or capture required) and the regression guard for metric/solver changes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from rdpieces.matcher import dissimilarity_matrices
from rdpieces.placement.edge_heuristic import attach_bottom_partials, mean_seam_cost, reconstruct
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


def test_attach_bottom_partials_places_partials_in_bottom_row():
    rng = np.random.default_rng(7)
    height, width = 2 * 64 + 56, 2 * 64  # 2x2 full tiles + a 56px-tall bottom row
    ctrl = rng.integers(0, 256, (8, 8, 3), np.uint8)
    img = np.asarray(Image.fromarray(ctrl).resize((width, height), Image.BICUBIC), np.uint8)

    def tile(y, x, h, w, idx):
        block = img[y : y + h, x : x + w]
        rgba = np.dstack([block, np.full((h, w), 255, np.uint8)]).astype(np.uint8)
        return Tile(0, 0, w, h, 32, rgba, "syn", idx)

    full = [tile(r * 64, c * 64, 64, 64, r * 2 + c) for r in range(2) for c in range(2)]
    partials = [tile(128, c * 64, 56, 64, 100 + c) for c in range(2)]

    right, down = dissimilarity_matrices(full)
    grid = reconstruct(full, right, down)
    out = attach_bottom_partials(grid, partials, max_cost=1e9)

    new_cells = set(out) - set(grid)
    bottom = max(r for r, _ in grid)
    assert len(new_cells) == 2
    assert all(r == bottom + 1 for r, _ in new_cells)
    assert {c for _, c in new_cells} == {0, 1}


def test_mean_seam_cost_low_for_correct_reconstruction():
    cols, rows = 5, 4
    tiles = make_tiles(cols, rows, seed=3)
    shuffled = tiles[:]
    np.random.default_rng(4).shuffle(shuffled)
    right, down = dissimilarity_matrices(shuffled)
    grid = reconstruct(shuffled, right, down)

    cost = mean_seam_cost(grid, shuffled, right, down)
    assert 0.0 <= cost < np.inf
