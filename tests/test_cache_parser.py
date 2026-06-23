"""Tests for the clean-room RDP bitmap-cache parser.

Synthetic fixtures are built byte-by-byte from the documented format so the
assertions are non-circular (we hand-pack the bytes, the parser must recover
the structure and decode channels correctly).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from rdpieces.cache_parser import BIN_MAGIC, discover_cache_files, parse_cache_file

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE0000 = REPO_ROOT / "Test-Cache-Files" / "Cache" / "Cache0000.bin"


def _bin_tile(key1: int, key2: int, width: int, height: int, bgr: tuple[int, int, int]) -> bytes:
    """A single .bin tile: 12-byte header + solid-colour BGRX pixel body."""
    b, g, r = bgr
    header = struct.pack("<IIHH", key1, key2, width, height)
    body = bytes([b, g, r, 0xFF]) * (width * height)
    return header + body


def _bin_file(version: int, tiles: list[bytes]) -> bytes:
    return BIN_MAGIC + struct.pack("<I", version) + b"".join(tiles)


def test_parses_bin_tile_count_keys_and_dims(tmp_path):
    # Two tiles: a full 64x64 and a bottom-row 64x56 partial.
    data = _bin_file(
        6,
        [
            _bin_tile(0xBFB017F6, 0xA9CE5F6E, 64, 64, (0, 0, 0)),
            _bin_tile(0x11111111, 0x22222222, 64, 56, (0, 0, 0)),
        ],
    )
    p = tmp_path / "Cache0000.bin"
    p.write_bytes(data)

    tiles = list(parse_cache_file(str(p)))

    assert len(tiles) == 2
    assert tiles[0].key == (0xBFB017F6, 0xA9CE5F6E)
    assert (tiles[0].width, tiles[0].height) == (64, 64)
    assert (tiles[1].width, tiles[1].height) == (64, 56)
    assert tiles[0].index == 0 and tiles[1].index == 1
    assert tiles[0].bpp == 32


def test_decodes_bgrx_to_rgba_channel_order(tmp_path):
    # Stored bytes are B,G,R,X. An orange pixel (R=255,G=128,B=0) is packed as
    # B=0,G=128,R=255 and must decode to RGBA (255,128,0,255).
    data = _bin_file(6, [_bin_tile(1, 2, 64, 64, (0, 128, 255))])
    p = tmp_path / "Cache0000.bin"
    p.write_bytes(data)

    tile = next(iter(parse_cache_file(str(p))))

    assert tile.pixels.shape == (64, 64, 4)
    assert tile.pixels.dtype == np.uint8
    assert tuple(tile.pixels[0, 0]) == (255, 128, 0, 255)


def test_discover_finds_cache_files_only(tmp_path):
    (tmp_path / "Cache0001.bin").write_bytes(b"")
    (tmp_path / "Cache0000.bin").write_bytes(b"")
    (tmp_path / "bcache24.bmc").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"x")

    found = discover_cache_files(str(tmp_path))

    names = [Path(p).name for p in found]
    assert names == ["Cache0000.bin", "Cache0001.bin", "bcache24.bmc"]


def test_discover_accepts_single_file(tmp_path):
    f = tmp_path / "Cache0000.bin"
    f.write_bytes(b"")
    assert discover_cache_files(str(f)) == [str(f)]


def test_tolerates_truncated_final_tile(tmp_path):
    # A valid full tile followed by a tile whose body is cut short (carved/partial
    # recovery). The parser must yield the good tile and not crash.
    good = _bin_tile(1, 1, 64, 64, (0, 0, 0))
    truncated = struct.pack("<IIHH", 2, 2, 64, 64) + b"\x00" * 100  # body far too short
    p = tmp_path / "Cache0000.bin"
    p.write_bytes(_bin_file(6, [good]) + truncated)

    tiles = list(parse_cache_file(str(p)))

    assert len(tiles) == 1
    assert tiles[0].key == (1, 1)


@pytest.mark.skipif(not CACHE0000.exists(), reason="Test-Cache-Files not present")
def test_real_cache0000_matches_measured_ground_truth():
    tiles = list(parse_cache_file(str(CACHE0000)))

    # Ground truth measured by direct byte inspection of the sample.
    assert len(tiles) == 3303
    assert tiles[0].key == (0xBFB017F6, 0xA9CE5F6E)
    assert (tiles[0].width, tiles[0].height) == (64, 64)

    dims = {}
    for t in tiles:
        dims[(t.width, t.height)] = dims.get((t.width, t.height), 0) + 1
    assert dims == {(64, 64): 3190, (64, 56): 113}
    # The 64x56 partials are the bottom-row edge tiles (1080 mod 64 == 56).
    assert all(t.width == 64 for t in tiles)
