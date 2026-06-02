"""Command-line interface for rdpieces."""

from __future__ import annotations

import json
import os
import re

import click
import numpy as np
from PIL import Image

from .assembler import render
from .cache_parser import discover_cache_files, parse_cache_file
from .constraints.resolution import ResolutionOracle
from .evidence import build_manifest
from .matcher import best_buddy_edges, dissimilarity_matrices
from .placement.edge_heuristic import reconstruct as place_tiles
from .segmentation import connected_components
from .tile_store import TileStore


def _parse_resolution(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", text)
    if not match:
        raise click.BadParameter(f"Expected WIDTHxHEIGHT (e.g. 1920x1080), got {text!r}")
    return int(match.group(1)), int(match.group(2))


@click.group()
@click.version_option()
def main() -> None:
    """rdpieces — automated RDP bitmap-cache reconstruction + OCR."""


@main.command()
@click.option("--source", required=True, help="Cache file or directory of cache files.")
@click.option("--output", required=True, help="Output directory for extracted tile images.")
def extract(source: str, output: str) -> None:
    """Extract every cache tile to an image (bmc-tools-style) + a manifest."""
    cache_files = discover_cache_files(source)
    if not cache_files:
        raise click.ClickException(f"No cache files found under {source!r}")
    os.makedirs(output, exist_ok=True)

    tile_count = 0
    for cache_file in cache_files:
        basename = os.path.basename(cache_file)
        for tile in parse_cache_file(cache_file):
            img = Image.fromarray(tile.pixels, "RGBA")
            img.save(os.path.join(output, f"{basename}_{tile.index:04d}.png"))
            tile_count += 1

    manifest = build_manifest(cache_files, command="extract")
    manifest["tile_count"] = tile_count
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    click.echo(f"Extracted {tile_count} tiles from {len(cache_files)} cache file(s) to {output}")


@main.command()
@click.option("--source", required=True, help="Cache file or directory of cache files.")
@click.option("--output", required=True, help="Output directory for reconstructed scenes.")
@click.option("--resolution", default=None, help="Known session resolution WIDTHxHEIGHT, e.g. 1920x1080.")
@click.option("--force", is_flag=True, help="Accept --resolution even if it conflicts with cache geometry.")
@click.option("--min-scene", default=6, show_default=True, help="Minimum tiles for a scene to be rendered.")
@click.option("--max-scene", default=1500, show_default=True, help="Skip scenes larger than this (logged).")
@click.option("--max-scenes", default=50, show_default=True, help="Cap rendered scenes (logged).")
def reconstruct(source, output, resolution, force, min_scene, max_scene, max_scenes):
    """Reconstruct screen regions from a cache by edge-matching tiles into scenes."""
    store = TileStore.load(source)
    unique = store.deduplicated()
    if not unique:
        raise click.ClickException(f"No tiles found under {source!r}")

    # Resolution constraint (cache geometry, optionally overridden by --resolution).
    oracle = ResolutionOracle.from_tiles(unique)
    user_res = _parse_resolution(resolution) if resolution else None
    constraint = oracle.constraint(user_res)
    if user_res is not None and not constraint.modulo_ok and not force:
        raise click.ClickException(constraint.notes[0] if constraint.notes else "Resolution conflict; use --force.")

    # MVP matches full 64x64 tiles only; partial edge tiles are anchored in Phase 3.
    full = [t for t in unique if t.is_full]
    partials_excluded = len(unique) - len(full)
    if len(full) < min_scene:
        raise click.ClickException(f"Only {len(full)} full tiles — nothing to reconstruct.")

    right, down = dissimilarity_matrices(full)
    components = connected_components(len(full), best_buddy_edges(right, down))

    os.makedirs(output, exist_ok=True)
    candidate_scenes = [c for c in components if len(c) >= min_scene]
    rendered = 0
    skipped_large = 0
    scene_records = []
    for comp in candidate_scenes:
        if rendered >= max_scenes:
            break
        if len(comp) > max_scene:
            skipped_large += 1
            click.echo(f"[skip] scene of {len(comp)} tiles exceeds --max-scene {max_scene}")
            continue
        idx = np.array(comp)
        sub = [full[i] for i in comp]
        grid = place_tiles(sub, right[np.ix_(idx, idx)], down[np.ix_(idx, idx)])
        canvas = render(grid)
        rows = max(r for r, _ in grid) + 1
        cols = max(c for _, c in grid) + 1
        name = f"scene_{rendered:03d}_{len(comp)}tiles_{cols}x{rows}.png"
        Image.fromarray(canvas, "RGBA").save(os.path.join(output, name))
        scene_records.append({"image": name, "tiles": len(comp), "grid_cols": cols, "grid_rows": rows})
        rendered += 1

    manifest = build_manifest(discover_cache_files(source), command="reconstruct")
    manifest.update(
        {
            "total_tiles": len(store.tiles),
            "unique_tiles": len(unique),
            "full_tiles_matched": len(full),
            "partial_tiles_excluded": partials_excluded,
            "components": len(components),
            "scenes_candidate": len(candidate_scenes),
            "scenes_rendered": rendered,
            "scenes_skipped_too_large": skipped_large,
            "resolution": {
                "source": constraint.source,
                "candidates": constraint.candidates,
                "width": constraint.width,
                "height": constraint.height,
                "width_mod64": constraint.width_mod64,
                "height_mod64": constraint.height_mod64,
                "modulo_ok": constraint.modulo_ok,
                "notes": constraint.notes,
            },
            "scenes": scene_records,
        }
    )
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    click.echo(
        f"Reconstructed {rendered} scene(s) from {len(full)} full tiles "
        f"({len(unique)} unique, {partials_excluded} partial excluded); "
        f"resolution candidates: {constraint.candidates or 'unknown'}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
