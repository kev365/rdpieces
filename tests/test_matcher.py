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
    right, _ = dissimilarity_matrices(tiles, metric="ssd")
    for i in range(3):
        for j in range(3):
            if i != j:
                # float64 matmul vs float32 pairwise differ only in rounding
                assert np.isclose(right[i, j], right_dissim(tiles[i], tiles[j]), rtol=1e-4)
    assert np.isinf(right[0, 0])  # self has no adjacency


def test_mgc_metric_ranks_true_neighbour_and_links_buddies():
    tiles = _gradient_strip(4)  # true order 0|1|2|3
    right, down = dissimilarity_matrices(tiles, metric="mgc")
    # tile 0's best right successor under MGC is its true neighbour 1
    assert int(np.argmin(right[0])) == 1
    edges = {frozenset(e) for e in best_buddy_edges(right, down)}
    assert frozenset((0, 1)) in edges and frozenset((2, 3)) in edges


def test_best_buddies_link_true_horizontal_neighbours():
    tiles = _gradient_strip(3)  # true order 0|1|2 left-to-right
    right, down = dissimilarity_matrices(tiles)
    edges = {frozenset(e) for e in best_buddy_edges(right, down)}
    assert frozenset((0, 1)) in edges
    assert frozenset((1, 2)) in edges


def test_flat_threshold_excludes_solid_tiles_from_best_buddy_graph():
    # 3 distinct gradient tiles + 3 identical solid-grey tiles. Without masking the
    # grey tiles would mutually best-buddy and merge; with masking they must not link.
    structured = _gradient_strip(3)
    grey = []
    for k in range(3):
        px = np.full((64, 64, 4), 128, np.uint8)
        px[..., 3] = 255
        grey.append(Tile(0, 0, 64, 64, 32, px, "f", 3 + k))
    tiles = structured + grey

    right, down = dissimilarity_matrices(tiles, flat_threshold=10.0)
    edges = best_buddy_edges(right, down)
    linked = {x for e in edges for x in e}

    assert linked.isdisjoint({3, 4, 5})  # solid tiles excluded
    assert frozenset((0, 1)) in {frozenset(e) for e in edges}  # structured still linked
