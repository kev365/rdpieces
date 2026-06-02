"""Per-tile signatures used for edge matching.

For now: the four 1-pixel borders as float RGB arrays. Phase 3 extends this with
Sobel/Prewitt gradient profiles and wider context histograms for robustness on
flat/identical-tile regions.
"""

from __future__ import annotations

import numpy as np

from .tile import Tile


def borders(tile: Tile) -> dict[str, np.ndarray]:
    """Return top/bottom/left/right border rows/cols as float32 RGB arrays."""
    rgb = tile.pixels[..., :3].astype(np.float32)
    return {
        "top": rgb[0, :, :],      # (W, 3)
        "bottom": rgb[-1, :, :],  # (W, 3)
        "left": rgb[:, 0, :],     # (H, 3)
        "right": rgb[:, -1, :],   # (H, 3)
    }


def border_variance(tile: Tile) -> dict[str, float]:
    """Per-side pixel-value variance. ~0 means a flat (solid-colour) border, which
    matches any other same-colour border and therefore makes an unreliable edge."""
    return {side: float(np.var(arr)) for side, arr in borders(tile).items()}
