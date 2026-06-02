"""Classical edge-matching placement (Phase 2 baseline).

Greedy growth from a seed tile: repeatedly place the unused tile that best fits an
empty cell adjacent to already-placed tiles (lowest mean border dissimilarity to
its placed neighbours). Candidate scoring is vectorised over unused tiles, and
precomputed dissimilarity matrices can be supplied (so a caller can compute them
once with the vectorised matcher and slice per scene). Phase 3 upgrades this to
MST -> loopy-BP/MRF -> CSP with anchor seeding and confidence scoring.
"""

from __future__ import annotations

import numpy as np

from ..matcher import dissimilarity_matrices
from ..tile import Tile

_NEIGHBOUR_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def reconstruct(
    tiles: list[Tile],
    right: np.ndarray | None = None,
    down: np.ndarray | None = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> dict[tuple[int, int], Tile]:
    """Arrange tiles into a grid; returns {(row, col): Tile} normalised to (0, 0).

    ``right[i, j]`` / ``down[i, j]`` are the cost of placing tile j to the right of
    / below tile i (indexed by position in ``tiles``). Computed if not supplied.
    ``max_rows`` / ``max_cols`` bound the grid span (from the resolution constraint),
    so the solver can't sprawl past one screen.
    """
    n = len(tiles)
    if n == 0:
        return {}
    if right is None or down is None:
        right, down = dissimilarity_matrices(tiles)

    placed: dict[tuple[int, int], int] = {(0, 0): 0}
    used = np.zeros(n, dtype=bool)
    used[0] = True

    def within_bounds(cell: tuple[int, int]) -> bool:
        rows = [r for r, _ in placed] + [cell[0]]
        cols = [c for _, c in placed] + [cell[1]]
        if max_rows is not None and (max(rows) - min(rows) + 1) > max_rows:
            return False
        if max_cols is not None and (max(cols) - min(cols) + 1) > max_cols:
            return False
        return True

    while not used.all():
        unused = np.flatnonzero(~used)
        frontier: set[tuple[int, int]] = set()
        for (r, c) in placed:
            for dr, dc in _NEIGHBOUR_OFFSETS:
                cell = (r + dr, c + dc)
                if cell not in placed and within_bounds(cell):
                    frontier.add(cell)
        if not frontier:
            break  # bounded grid is full; remaining tiles belong to another scene

        best_score, best_cell, best_idx = np.inf, None, None
        for (r, c) in frontier:
            left = placed.get((r, c - 1))
            right_n = placed.get((r, c + 1))
            up = placed.get((r - 1, c))
            down_n = placed.get((r + 1, c))

            cost = np.zeros(unused.shape)
            count = 0
            if left is not None:
                cost = cost + right[left, unused]
                count += 1
            if right_n is not None:
                cost = cost + right[unused, right_n]
                count += 1
            if up is not None:
                cost = cost + down[up, unused]
                count += 1
            if down_n is not None:
                cost = cost + down[unused, down_n]
                count += 1
            if count == 0:
                continue
            avg = cost / count  # inf where any neighbour edge clashes on dimensions
            j = int(np.argmin(avg))
            if avg[j] < best_score:
                best_score, best_cell, best_idx = float(avg[j]), (r, c), int(unused[j])

        if best_cell is None or not np.isfinite(best_score):
            break  # nothing placeable (all remaining tiles clash on dimensions)
        placed[best_cell] = best_idx
        used[best_idx] = True

    min_r = min(r for r, _ in placed)
    min_c = min(c for _, c in placed)
    return {(r - min_r, c - min_c): tiles[idx] for (r, c), idx in placed.items()}


def mean_seam_cost(
    grid: dict[tuple[int, int], Tile],
    tiles: list[Tile],
    right: np.ndarray,
    down: np.ndarray,
) -> float:
    """Average dissimilarity across placed adjacent seams (lower = more confident)."""
    pos = {id(t): i for i, t in enumerate(tiles)}
    cells = {cell: pos[id(t)] for cell, t in grid.items()}
    costs = []
    for (r, c), i in cells.items():
        j = cells.get((r, c + 1))
        if j is not None and np.isfinite(right[i, j]):
            costs.append(right[i, j])
        k = cells.get((r + 1, c))
        if k is not None and np.isfinite(down[i, k]):
            costs.append(down[i, k])
    return float(np.mean(costs)) if costs else float("inf")
