"""Tests for the `rdpieces extract` command (bmc-tools-style tile extraction)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from click.testing import CliRunner

from rdpieces.cli import main


def _bin_file(tiles):
    out = b"RDP8bmp\x00" + struct.pack("<I", 6)
    for key1, key2, w, h, (b, g, r) in tiles:
        out += struct.pack("<IIHH", key1, key2, w, h) + bytes([b, g, r, 0xFF]) * (w * h)
    return out


def test_extract_writes_one_image_per_tile_and_manifest(tmp_path):
    src = tmp_path / "Cache"
    src.mkdir()
    (src / "Cache0000.bin").write_bytes(
        _bin_file([(1, 2, 64, 64, (0, 0, 0)), (3, 4, 64, 56, (0, 128, 255))])
    )
    out = tmp_path / "out"

    result = CliRunner().invoke(main, ["extract", "--source", str(src), "--output", str(out)])

    assert result.exit_code == 0, result.output
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["Cache0000.bin_0000.png", "Cache0000.bin_0001.png"]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["command"] == "extract"
    assert manifest["tile_count"] == 2
    assert manifest["sources"][0]["path"].endswith("Cache0000.bin")


def test_extract_image_has_correct_size_and_pixels(tmp_path):
    from PIL import Image

    src = tmp_path / "Cache"
    src.mkdir()
    (src / "Cache0000.bin").write_bytes(_bin_file([(1, 2, 64, 56, (0, 128, 255))]))
    out = tmp_path / "out"

    CliRunner().invoke(main, ["extract", "--source", str(src), "--output", str(out)])

    img = Image.open(out / "Cache0000.bin_0000.png").convert("RGBA")
    assert img.size == (64, 56)  # (width, height)
    assert img.getpixel((0, 0)) == (255, 128, 0, 255)
