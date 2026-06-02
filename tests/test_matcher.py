"""Tests for vectorised dissimilarity matrices and best-buddy edges."""

from __future__ import annotations

import numpy as np

from rdpieces.compatibility import right_dissim
from rdpieces.matcher import best_buddy_edges, dissimilarity_matrices
from rdpieces.tile import Tile


def tiles_from_image(img: np.ndarray, cols: int, rows: int) -> list[Tile]:
    out, k = [], 0
    for r in range(rows):
        for c in range(cols):
            block = img[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
            rgba = np.dstack([block, np.full((64, 64), 255, np.uint8)]).astype(np.uint8)
            out.append(Tile(0, 0, 64, 64, 32, rgba, "f", k))
            k += 1
    return out


def _gradient_strip(cols: int) -> list[Tile]:
    grad = np.linspace(0, 255, cols * 64, dtype=np.uint8)
    img = np.repeat(grad[None, :], 64, axis=0)
    img = np.dstack([img, (255 - img), (img // 2).astype(np.uint8)])
    return tiles_from_image(img, cols, 1)


def test_dissimilarity_matrix_matches_pairwise_compatibility():
    tiles = _gradient_strip(3)
    right, _ = dissimilarity_matrices(tiles)
    for i in range(3):
        for j in range(3):
            if i != j:
                # float64 matmul vs float32 pairwise differ only in rounding
                assert np.isclose(right[i, j], right_dissim(tiles[i], tiles[j]), rtol=1e-4)
    assert np.isinf(right[0, 0])  # self has no adjacency


def test_best_buddies_link_true_horizontal_neighbours():
    tiles = _gradient_strip(3)  # true order 0|1|2 left-to-right
    right, down = dissimilarity_matrices(tiles)
    edges = {frozenset(e) for e in best_buddy_edges(right, down)}
    assert frozenset((0, 1)) in edges
    assert frozenset((1, 2)) in edges
