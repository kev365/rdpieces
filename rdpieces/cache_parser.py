"""Clean-room parser for RDP persistent bitmap cache files.

Two on-disk formats (knowledge from [MS-RDPEGDI]/[MS-RDPBCGR] + byte inspection):

* New (Windows 7+): ``Cache000X.bin`` — 12-byte file header ``b"RDP8bmp\\x00"`` +
  uint32 version, then a stream of tiles. Each tile = 12-byte header
  (key1:u32, key2:u32, width:u16, height:u16) followed by width*height*4 bytes of
  32bpp pixel data stored as B,G,R,X rows top-to-bottom.
* Old (pre-Win7): ``bcacheN.bmc`` — 20-byte tile headers, variable bpp. (Added later.)

This module does not copy ANSSI bmc-tools; it is implemented from the spec and
from inspecting sample caches.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator

import numpy as np

from .tile import Tile

BIN_MAGIC = b"RDP8bmp\x00"
_BIN_FILE_HEADER_SIZE = 12  # 8-byte magic + uint32 version
_BIN_TILE_HEADER = struct.Struct("<IIHH")  # key1, key2, width, height


def discover_cache_files(path: str) -> list[str]:
    """Return cache files under ``path``.

    If ``path`` is a single file, return ``[path]``. If a directory, return its
    ``Cache*.bin`` files (sorted) followed by its ``bcache*.bmc`` files (sorted).
    """
    if os.path.isfile(path):
        return [path]
    entries = os.listdir(path)
    bins = sorted(e for e in entries if e.lower().startswith("cache") and e.lower().endswith(".bin"))
    bmcs = sorted(e for e in entries if e.lower().startswith("bcache") and e.lower().endswith(".bmc"))
    return [os.path.join(path, e) for e in (*bins, *bmcs)]


def parse_cache_file(path: str) -> Iterator[Tile]:
    """Yield Tiles from a cache file, dispatching on the file format."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[: len(BIN_MAGIC)] == BIN_MAGIC:
        yield from _parse_bin(data, path)
    else:
        raise ValueError(f"Unrecognised cache format (no RDP8bmp magic): {path}")


def _parse_bin(data: bytes, path: str) -> Iterator[Tile]:
    off = _BIN_FILE_HEADER_SIZE
    n = len(data)
    index = 0
    while off + _BIN_TILE_HEADER.size <= n:
        key1, key2, width, height = _BIN_TILE_HEADER.unpack_from(data, off)
        off += _BIN_TILE_HEADER.size
        body_len = width * height * 4
        if off + body_len > n:
            break  # truncated final tile — tolerated (carved-input handling refined later)
        bgrx = np.frombuffer(data, dtype=np.uint8, count=body_len, offset=off).reshape(height, width, 4)
        off += body_len
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = bgrx[..., 2]  # R
        rgba[..., 1] = bgrx[..., 1]  # G
        rgba[..., 2] = bgrx[..., 0]  # B
        rgba[..., 3] = 255
        yield Tile(key1, key2, width, height, 32, rgba, path, index)
        index += 1
