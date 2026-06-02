"""Integration test for `rdpieces reconstruct` on a synthetic cache."""

from __future__ import annotations

import json
import struct

import numpy as np
from click.testing import CliRunner
from PIL import Image

from rdpieces.cli import main


def write_synthetic_bin(path, cols, rows, seed=0):
    rng = np.random.default_rng(seed)
    height, width = rows * 64, cols * 64
    control = rng.integers(0, 256, (rows * 2 + 1, cols * 2 + 1, 3), dtype=np.uint8)
    img = np.asarray(Image.fromarray(control).resize((width, height), Image.BICUBIC), dtype=np.uint8)
    out = b"RDP8bmp\x00" + struct.pack("<I", 6)
    for r in range(rows):
        for c in range(cols):
            block = img[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64]
            bgrx = np.dstack(
                [block[..., 2], block[..., 1], block[..., 0], np.full((64, 64), 255, np.uint8)]
            ).astype(np.uint8)
            out += struct.pack("<IIHH", r * cols + c, 0, 64, 64) + bgrx.tobytes()
    path.write_bytes(out)


def test_reconstruct_produces_scene_images_and_manifest(tmp_path):
    src = tmp_path / "Cache"
    src.mkdir()
    write_synthetic_bin(src / "Cache0000.bin", cols=6, rows=5)
    out = tmp_path / "out"

    result = CliRunner().invoke(main, ["reconstruct", "--source", str(src), "--output", str(out)])

    assert result.exit_code == 0, result.output
    scenes = list(out.glob("scene_*.png"))
    assert len(scenes) >= 1

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["command"] == "reconstruct"
    assert manifest["scenes_rendered"] >= 1
    assert manifest["unique_tiles"] == 30
