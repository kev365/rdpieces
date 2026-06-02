"""The Tile data structure: one decoded entry from an RDP bitmap cache.

The on-disk cache stores only the bitmap pixels, a 64-bit content key (key1/key2),
and the tile dimensions. It carries NO screen coordinates — placement must be
inferred downstream. A Tile therefore records exactly what the cache holds plus
provenance (which file/index it came from) for chain-of-custody.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Tile:
    """A single decoded cache tile.

    pixels is an (H, W, 4) uint8 array in RGBA order, top row first.
    """

    key1: int
    key2: int
    width: int
    height: int
    bpp: int
    pixels: np.ndarray
    source_file: str
    index: int
    mtime: float | None = None

    @property
    def key(self) -> tuple[int, int]:
        """The 64-bit persistent cache key as a (key1, key2) tuple."""
        return (self.key1, self.key2)

    @property
    def is_full(self) -> bool:
        """True for a full 64x64 tile (not an edge-trimmed partial)."""
        return self.width == 64 and self.height == 64

    @property
    def is_flat(self) -> bool:
        """True if every pixel is identical (solid colour) — a weak match signal."""
        flat = self.pixels.reshape(-1, self.pixels.shape[-1])
        return bool(np.all(flat == flat[0]))
