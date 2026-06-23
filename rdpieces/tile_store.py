"""Load tiles from cache files and deduplicate by content key.

key1/key2 is a content hash, so identical keys mean identical pixels. Dedup
shrinks the n^2 matching problem; multiplicity is retained because a tile reused
across the session is evidence of a recurring UI element.
"""

from __future__ import annotations

from collections import Counter

from .cache_parser import discover_cache_files, parse_cache_file
from .tile import Tile


class TileStore:
    def __init__(self, tiles: list[Tile]) -> None:
        self.tiles = tiles

    @classmethod
    def load(cls, source: str) -> "TileStore":
        tiles: list[Tile] = []
        for cache_file in discover_cache_files(source):
            tiles.extend(parse_cache_file(cache_file))
        return cls(tiles)

    def key_multiplicity(self) -> dict[tuple[int, int], int]:
        return dict(Counter(t.key for t in self.tiles))

    def deduplicated(self) -> list[Tile]:
        """Return the first tile seen for each distinct content key, in order."""
        seen: set[tuple[int, int]] = set()
        unique: list[Tile] = []
        for tile in self.tiles:
            if tile.key not in seen:
                seen.add(tile.key)
                unique.append(tile)
        return unique
