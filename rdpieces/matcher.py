"""Vectorised tile matching and candidate pruning.

Dissimilarity is computed with a single matmul per direction using the identity
mean((a-b)^2) = (||a||^2 + ||b||^2 - 2 a.b) / d, which scales to thousands of
tiles far better than pairwise Python loops. Operates on a homogeneous list of
equal-dimension tiles (callers group by (width, height) first).
"""

from __future__ import annotations

import numpy as np

from .signatures import borders
from .tile import Tile


def _edge_stack(tiles: list[Tile], side: str) -> np.ndarray:
    return np.stack([borders(t)[side].reshape(-1) for t in tiles]).astype(np.float64)


def _dissim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise mean-squared-difference between rows of a and rows of b."""
    d = a.shape[1]
    sa = np.sum(a * a, axis=1)[:, None]
    sb = np.sum(b * b, axis=1)[None, :]
    out = (sa + sb - 2.0 * (a @ b.T)) / d
    np.fill_diagonal(out, np.inf)
    return np.maximum(out, 0.0)  # guard tiny negatives from float error (diagonal stays inf)


def dissimilarity_matrices(tiles: list[Tile]) -> tuple[np.ndarray, np.ndarray]:
    """Return (right, down) cost matrices; [i, j] = cost of j right-of-i / below-i."""
    right = _dissim(_edge_stack(tiles, "right"), _edge_stack(tiles, "left"))
    down = _dissim(_edge_stack(tiles, "bottom"), _edge_stack(tiles, "top"))
    return right, down


def _mutual_best(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Pairs (i, j) where j is i's best successor and i is j's best predecessor."""
    best_succ = np.argmin(matrix, axis=1)  # for row i, best j
    best_pred = np.argmin(matrix, axis=0)  # for col j, best i
    edges = []
    for i, j in enumerate(best_succ):
        if best_pred[j] == i and np.isfinite(matrix[i, j]):
            edges.append((int(i), int(j)))
    return edges


def best_buddy_edges(right: np.ndarray, down: np.ndarray) -> list[tuple[int, int]]:
    """Undirected edges between tiles that are mutual best buddies in any direction."""
    seen: set[frozenset] = set()
    edges: list[tuple[int, int]] = []
    for i, j in _mutual_best(right) + _mutual_best(down):
        key = frozenset((i, j))
        if key not in seen:
            seen.add(key)
            edges.append((i, j))
    return edges
