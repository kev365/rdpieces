"""Tests for TileStore: loading and content-key deduplication."""

from __future__ import annotations

import numpy as np

from rdpieces.tile import Tile
from rdpieces.tile_store import TileStore


def mk(key, idx=0):
    return Tile(key[0], key[1], 64, 64, 32, np.zeros((64, 64, 4), np.uint8), "f", idx)


def test_deduplicated_collapses_identical_keys():
    store = TileStore([mk((1, 1), 0), mk((1, 1), 1), mk((2, 2), 2)])
    unique = store.deduplicated()
    assert len(unique) == 2
    assert {t.key for t in unique} == {(1, 1), (2, 2)}


def test_key_multiplicity_counts_occurrences():
    store = TileStore([mk((1, 1), 0), mk((1, 1), 1), mk((2, 2), 2)])
    mult = store.key_multiplicity()
    assert mult[(1, 1)] == 2
    assert mult[(2, 2)] == 1
