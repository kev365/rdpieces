"""Vectorised tile matching and candidate pruning.

Dissimilarity is computed with a single matmul per direction using the identity
mean((a-b)^2) = (||a||^2 + ||b||^2 - 2 a.b) / d, which scales to thousands of
tiles far better than pairwise Python loops. Operates on a homogeneous list of
equal-dimension tiles (callers group by (width, height) first).
"""

from __future__ import annotations

import numpy as np

from .signatures import border_variance, borders, inner_borders
from .tile import Tile


def _edge_stack(tiles: list[Tile], side: str) -> np.ndarray:
    return np.stack([borders(t)[side].reshape(-1) for t in tiles]).astype(np.float64)


def _inner_stack(tiles: list[Tile], side: str) -> np.ndarray:
    return np.stack([inner_borders(t)[side].reshape(-1) for t in tiles]).astype(np.float64)


def _flat_mask(tiles: list[Tile], side: str, threshold: float) -> np.ndarray:
    return np.array([border_variance(t)[side] < threshold for t in tiles], dtype=bool)


def _dissim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise mean-squared-difference between rows of a and rows of b."""
    d = a.shape[1]
    sa = np.sum(a * a, axis=1)[:, None]
    sb = np.sum(b * b, axis=1)[None, :]
    out = (sa + sb - 2.0 * (a @ b.T)) / d
    np.fill_diagonal(out, np.inf)
    return np.maximum(out, 0.0)  # guard tiny negatives from float error (diagonal stays inf)


def _mgc_pair(outer_a, inner_a, outer_b, inner_b):
    """Symmetric gradient-prediction cost: each side extrapolates across the seam
    and is scored against the other's actual border. Sharper than raw SSD on
    gradients (the true neighbour continues the slope)."""
    pred_a = 2.0 * outer_a - inner_a  # A predicts B's border
    pred_b = 2.0 * outer_b - inner_b  # B predicts A's border
    return _dissim(pred_a, outer_b) + _dissim(outer_a, pred_b)


def dissimilarity_matrices(
    tiles: list[Tile], flat_threshold: float | None = None, metric: str = "blend"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (right, down) cost matrices; [i, j] = cost of j right-of-i / below-i.

    metric: "ssd" (border difference), "mgc" (gradient-prediction), or "blend" (sum).
    If ``flat_threshold`` is given, edges touching a flat (solid-colour) border are
    set to infinity so background tiles don't form spurious confident matches.
    """
    r_out, l_out = _edge_stack(tiles, "right"), _edge_stack(tiles, "left")
    b_out, t_out = _edge_stack(tiles, "bottom"), _edge_stack(tiles, "top")

    if metric in ("ssd", "blend"):
        right = _dissim(r_out, l_out)
        down = _dissim(b_out, t_out)
    if metric in ("mgc", "blend"):
        r_in, l_in = _inner_stack(tiles, "right"), _inner_stack(tiles, "left")
        b_in, t_in = _inner_stack(tiles, "bottom"), _inner_stack(tiles, "top")
        right_mgc = _mgc_pair(r_out, r_in, l_out, l_in)
        down_mgc = _mgc_pair(b_out, b_in, t_out, t_in)
        if metric == "mgc":
            right, down = right_mgc, down_mgc
        else:
            right = right + right_mgc
            down = down + down_mgc
    if metric not in ("ssd", "mgc", "blend"):
        raise ValueError(f"unknown metric {metric!r}")

    if flat_threshold is not None:
        right[_flat_mask(tiles, "right", flat_threshold), :] = np.inf
        right[:, _flat_mask(tiles, "left", flat_threshold)] = np.inf
        down[_flat_mask(tiles, "bottom", flat_threshold), :] = np.inf
        down[:, _flat_mask(tiles, "top", flat_threshold)] = np.inf
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
