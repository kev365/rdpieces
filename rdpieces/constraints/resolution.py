"""Cache-intrinsic resolution inference (resolution-from-trim).

mstsc tiles the framebuffer on a fixed 64x64 grid and clips edge tiles to the
screen remainder. So a tile narrower than 64 is a right-edge piece whose width is
``screen_width mod 64``; a tile shorter than 64 is a bottom-edge piece whose
height is ``screen_height mod 64``; a tile trimmed on both axes is the bottom-right
corner. These partial tiles are placement anchors, and the modulo pair narrows the
likely resolution.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..tile import Tile

# Common desktop/RDP resolutions used to rank candidates from a modulo pair.
_COMMON_RESOLUTIONS = [
    (800, 600), (1024, 768), (1152, 864), (1280, 720), (1280, 800), (1280, 960),
    (1280, 1024), (1360, 768), (1366, 768), (1440, 900), (1600, 900), (1600, 1200),
    (1680, 1050), (1920, 1080), (1920, 1200), (2048, 1152), (2560, 1080),
    (2560, 1440), (2560, 1600), (3440, 1440), (3840, 1080), (3840, 2160),
]


@dataclass(slots=True)
class Anchor:
    kind: str  # "right_edge" | "bottom_edge" | "corner"
    col: int | None
    row: int | None


@dataclass(slots=True)
class CanvasConstraint:
    width: int | None
    height: int | None
    width_mod64: int
    height_mod64: int
    cols: int | None
    rows: int | None
    candidates: list[tuple[int, int]]
    source: str
    confidence: float
    modulo_ok: bool = True
    notes: list[str] = field(default_factory=list)


class ResolutionOracle:
    def __init__(self, width_mod64: int, height_mod64: int) -> None:
        self.width_mod64 = width_mod64
        self.height_mod64 = height_mod64

    @classmethod
    def from_tiles(cls, tiles: Iterable[Tile]) -> "ResolutionOracle":
        partial_w = Counter(t.width for t in tiles if t.width < 64)
        partial_h = Counter(t.height for t in tiles if t.height < 64)
        width_mod = partial_w.most_common(1)[0][0] if partial_w else 0
        height_mod = partial_h.most_common(1)[0][0] if partial_h else 0
        return cls(width_mod, height_mod)

    def _candidates(self) -> list[tuple[int, int]]:
        return [
            (w, h)
            for (w, h) in _COMMON_RESOLUTIONS
            if w % 64 == self.width_mod64 and h % 64 == self.height_mod64
        ]

    def constraint(
        self, resolution: tuple[int, int] | None = None, source: str = "user"
    ) -> CanvasConstraint:
        wmod, hmod = self.width_mod64, self.height_mod64
        candidates = self._candidates()

        if resolution is not None:
            width, height = resolution
            modulo_ok = (width % 64 == wmod) and (height % 64 == hmod)
            notes: list[str] = []
            if not modulo_ok:
                notes.append(
                    f"{source} resolution {width}x{height} MISMATCH with cache edge-tile "
                    f"geometry: expected (W,H) mod 64 == ({wmod},{hmod}) but got "
                    f"({width % 64},{height % 64}). Pass --force to accept."
                )
            return CanvasConstraint(
                width=width,
                height=height,
                width_mod64=wmod,
                height_mod64=hmod,
                cols=math.ceil(width / 64),
                rows=math.ceil(height / 64),
                candidates=candidates,
                source=source,
                confidence=1.0 if modulo_ok else 0.5,
                modulo_ok=modulo_ok,
                notes=notes,
            )

        # No explicit resolution: only commit to a grid if the modulo uniquely
        # identifies one common resolution; otherwise leave it open (candidates only).
        if len(candidates) == 1:
            width, height = candidates[0]
            return CanvasConstraint(
                width=width,
                height=height,
                width_mod64=wmod,
                height_mod64=hmod,
                cols=math.ceil(width / 64),
                rows=math.ceil(height / 64),
                candidates=candidates,
                source="cache_geometry",
                confidence=0.6,
            )
        return CanvasConstraint(
            width=None,
            height=None,
            width_mod64=wmod,
            height_mod64=hmod,
            cols=None,
            rows=None,
            candidates=candidates,
            source="cache_geometry",
            confidence=0.3,
        )


def anchor_for_tile(tile: Tile, cols: int, rows: int) -> Anchor | None:
    """Classify a tile as a screen-edge anchor based on its trim, or None if full."""
    partial_w = tile.width < 64
    partial_h = tile.height < 64
    if partial_w and partial_h:
        return Anchor("corner", cols - 1, rows - 1)
    if partial_h:
        return Anchor("bottom_edge", None, rows - 1)
    if partial_w:
        return Anchor("right_edge", cols - 1, None)
    return None
