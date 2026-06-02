"""Tests for the cache-intrinsic ResolutionOracle (resolution-from-trim)."""

from __future__ import annotations

import numpy as np

from rdpieces.constraints.resolution import ResolutionOracle, anchor_for_tile
from rdpieces.tile import Tile


def mk(w: int, h: int) -> Tile:
    return Tile(0, 0, w, h, 32, np.zeros((h, w, 4), np.uint8), "f", 0)


def test_modulo_from_trimmed_tiles():
    oracle = ResolutionOracle.from_tiles([mk(64, 64), mk(64, 64), mk(64, 56)])
    assert oracle.width_mod64 == 0
    assert oracle.height_mod64 == 56


def test_candidates_include_1920x1080_for_0_56():
    c = ResolutionOracle.from_tiles([mk(64, 64), mk(64, 56)]).constraint()
    assert c.width_mod64 == 0 and c.height_mod64 == 56
    assert (1920, 1080) in c.candidates


def test_user_resolution_overrides_and_sets_grid():
    c = ResolutionOracle.from_tiles([mk(64, 64), mk(64, 56)]).constraint(resolution=(1920, 1080))
    assert (c.width, c.height) == (1920, 1080)
    assert (c.cols, c.rows) == (30, 17)  # ceil(1920/64), ceil(1080/64)
    assert c.source == "user"
    assert c.modulo_ok is True


def test_user_resolution_modulo_mismatch_is_flagged():
    # Cache shows height_mod64 == 56, but the analyst claims 1920x1024 (1024 % 64 == 0).
    c = ResolutionOracle.from_tiles([mk(64, 64), mk(64, 56)]).constraint(resolution=(1920, 1024))
    assert c.modulo_ok is False
    assert any("mismatch" in n.lower() for n in c.notes)


def test_partial_height_tile_is_bottom_anchor():
    a = anchor_for_tile(mk(64, 56), cols=30, rows=17)
    assert a is not None and a.kind == "bottom_edge"
    assert a.row == 16 and a.col is None


def test_partial_both_dims_is_corner_anchor():
    a = anchor_for_tile(mk(40, 56), cols=30, rows=17)
    assert a is not None and a.kind == "corner"
    assert (a.col, a.row) == (29, 16)


def test_full_tile_has_no_anchor():
    assert anchor_for_tile(mk(64, 64), cols=30, rows=17) is None
