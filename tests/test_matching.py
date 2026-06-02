"""Tests for per-tile border signatures and directional compatibility."""

from __future__ import annotations

import numpy as np

from rdpieces.compatibility import INF, down_dissim, right_dissim
from rdpieces.signatures import border_variance, borders
from rdpieces.tile import Tile


def tile_from_rgb(rgb: np.ndarray) -> Tile:
    h, w, _ = rgb.shape
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)]).astype(np.uint8)
    return Tile(0, 0, w, h, 32, rgba, "f", 0)


def test_borders_have_expected_shapes_and_values():
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[0, :, 0] = 10  # top row, red channel
    rgb[:, -1, 1] = 20  # right column, green channel
    b = borders(tile_from_rgb(rgb))
    assert b["top"].shape == (64, 3)
    assert b["left"].shape == (64, 3)
    assert b["top"][0, 0] == 10
    assert b["right"][0, 1] == 20


def test_right_dissim_lower_for_true_horizontal_neighbour():
    # A 64x128 horizontal gradient split into two adjacent 64x64 tiles.
    grad = np.linspace(0, 255, 128, dtype=np.uint8)
    img = np.repeat(grad[None, :], 64, axis=0)
    rgb = np.dstack([img, img, img])
    left = tile_from_rgb(rgb[:, :64])
    right = tile_from_rgb(rgb[:, 64:])
    reversed_tile = tile_from_rgb(rgb[:, 64:][:, ::-1].copy())

    assert right_dissim(left, right) < right_dissim(left, reversed_tile)


def test_border_variance_zero_for_solid_and_positive_for_gradient():
    solid = tile_from_rgb(np.full((64, 64, 3), 100, np.uint8))
    assert border_variance(solid)["top"] == 0.0

    grad = np.repeat(np.linspace(0, 255, 64, dtype=np.uint8)[None, :], 64, axis=0)
    textured = tile_from_rgb(np.dstack([grad, grad, grad]))
    assert border_variance(textured)["top"] > 0.0


def test_dissim_is_inf_on_dimension_mismatch():
    a = tile_from_rgb(np.zeros((64, 64, 3), np.uint8))
    short = tile_from_rgb(np.zeros((56, 64, 3), np.uint8))
    assert right_dissim(a, short) == INF  # different heights can't be left/right neighbours
    narrow = tile_from_rgb(np.zeros((64, 40, 3), np.uint8))
    assert down_dissim(a, narrow) == INF  # different widths can't be top/bottom neighbours
