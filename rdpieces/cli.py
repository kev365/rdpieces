"""Command-line interface for rdpieces."""

from __future__ import annotations

import json
import math
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
from .placement.edge_heuristic import mean_seam_cost
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
@click.option("--flat-threshold", default=10.0, show_default=True, help="Border variance below which an edge is ignored (flat-tile suppression). 0 disables.")
def reconstruct(source, output, resolution, force, min_scene, max_scene, max_scenes, flat_threshold):
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

    right, down = dissimilarity_matrices(full, flat_threshold=flat_threshold or None)
    components = connected_components(len(full), best_buddy_edges(right, down))

    # Bound the scene grid from the resolution constraint so the solver can't sprawl
    # past one screen. Rows are confident when all candidates share a height.
    if constraint.rows is not None:
        rows_bound, cols_bound = constraint.rows, constraint.cols
    elif constraint.candidates:
        cand_rows = {math.ceil(h / 64) for _, h in constraint.candidates}
        cand_cols = {math.ceil(w / 64) for w, _ in constraint.candidates}
        rows_bound = cand_rows.pop() if len(cand_rows) == 1 else max(cand_rows)
        cols_bound = max(cand_cols)
    else:
        rows_bound = cols_bound = None

    os.makedirs(output, exist_ok=True)
    candidate_scenes = [c for c in components if len(c) >= min_scene]

    # Reconstruct each candidate scene and score its seam confidence, then render
    # the most-confident first (most-trustworthy reconstructions surface at the top).
    reconstructed = []
    skipped_large = 0
    for comp in candidate_scenes:
        if len(comp) > max_scene:
            skipped_large += 1
            click.echo(f"[skip] scene of {len(comp)} tiles exceeds --max-scene {max_scene}")
            continue
        idx = np.array(comp)
        sub = [full[i] for i in comp]
        sub_right, sub_down = right[np.ix_(idx, idx)], down[np.ix_(idx, idx)]
        grid = place_tiles(sub, sub_right, sub_down, max_rows=rows_bound, max_cols=cols_bound)
        cost = mean_seam_cost(grid, sub, sub_right, sub_down)
        reconstructed.append((cost, len(comp), grid))

    reconstructed.sort(key=lambda r: r[0])  # ascending cost = most confident first
    scene_records = []
    for cost, n_tiles, grid in reconstructed[:max_scenes]:
        canvas = render(grid)
        rows = max(r for r, _ in grid) + 1
        cols = max(c for _, c in grid) + 1
        name = f"scene_{len(scene_records):03d}_{n_tiles}tiles_{cols}x{rows}_conf{cost:.1f}.png"
        Image.fromarray(canvas, "RGBA").save(os.path.join(output, name))
        scene_records.append(
            {"image": name, "tiles": n_tiles, "grid_cols": cols, "grid_rows": rows, "mean_seam_cost": round(cost, 2)}
        )
    rendered = len(scene_records)

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
