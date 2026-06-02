"""Command-line interface for rdpieces."""

from __future__ import annotations

import json
import math
import os
import re

import click
import numpy as np
from PIL import Image

from .assembler import montage, render
from .cache_parser import discover_cache_files, parse_cache_file
from .constraints.artifacts import collect_artifacts
from .constraints.resolution import ResolutionOracle
from .evidence import build_manifest
from .matcher import best_buddy_edges, dissimilarity_matrices
from .ocr.engine import ocr_image, tesseract_available, words_to_records, words_to_text
from .placement.edge_heuristic import attach_bottom_partials, mean_seam_cost
from .placement.edge_heuristic import reconstruct as place_tiles
from .segmentation import connected_components
from .tile_store import TileStore


MONTAGE_PADDING = 8  # transparent gap between stacked scenes in the final reconstruction


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
@click.option("--artifacts", default=None, help="File or dir of host artifacts (.rdp / NTUSER.DAT / RdpCoreTS .evtx) to infer resolution/host/OS.")
@click.option("--min-scene", default=6, show_default=True, help="Minimum tiles for a scene to be rendered.")
@click.option("--max-scene", default=1500, show_default=True, help="Skip scenes larger than this (logged).")
@click.option("--max-scenes", default=50, show_default=True, help="Cap rendered scenes (logged).")
@click.option("--flat-threshold", default=10.0, show_default=True, help="Border variance below which an edge is ignored (flat-tile suppression). 0 disables.")
@click.option("--ocr", is_flag=True, help="Run OCR on each reconstructed scene (requires the Tesseract binary).")
@click.option("--ocr-lang", default="eng", show_default=True, help="Tesseract language(s), e.g. eng or eng+deu.")
@click.option("--ocr-scale", default=6, show_default=True, help="Upscale factor before OCR.")
@click.option("--ocr-min-conf", default=40.0, show_default=True, help="Drop OCR words below this confidence.")
def reconstruct(
    source, output, resolution, force, artifacts, min_scene, max_scene, max_scenes, flat_threshold,
    ocr, ocr_lang, ocr_scale, ocr_min_conf,
):
    """Reconstruct screen regions from a cache by edge-matching tiles into scenes."""
    store = TileStore.load(source)
    unique = store.deduplicated()
    if not unique:
        raise click.ClickException(f"No tiles found under {source!r}")

    # Resolution constraint: cache geometry, optionally overridden by --resolution,
    # else inferred from host artifacts (--artifacts) when compatible with the cache.
    oracle = ResolutionOracle.from_tiles(unique)
    bundle = collect_artifacts(artifacts) if artifacts else None
    user_res = _parse_resolution(resolution) if resolution else None

    chosen_res, chosen_source = user_res, "user"
    if user_res is None and bundle is not None and bundle.width and bundle.height:
        if bundle.width % 64 == oracle.width_mod64 and bundle.height % 64 == oracle.height_mod64:
            chosen_res, chosen_source = (bundle.width, bundle.height), bundle.resolution_source
        else:
            click.echo(
                f"[artifacts] {bundle.resolution_source} resolution {bundle.width}x{bundle.height} "
                f"conflicts with cache edge geometry; ignoring for grid bounds."
            )

    constraint = oracle.constraint(chosen_res, source=chosen_source)
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

    do_ocr = ocr
    if ocr and not tesseract_available():
        do_ocr = False
        click.echo(
            "[ocr] Tesseract binary not found - skipping OCR. Install it (Windows: "
            "winget install UB-Mannheim.TesseractOCR) and ensure it's on PATH."
        )

    scene_records = []
    scene_canvases = []
    final_text_parts = []
    final_words = []
    partials_pool = [t for t in unique if not t.is_full]  # bottom/right edge tiles
    partials_attached = 0
    y_offset = 0  # running top edge of each scene within the montage
    for cost, n_tiles, grid in reconstructed[:max_scenes]:
        # Anchor partial (bottom-edge) tiles below this scene, consuming the pool.
        if partials_pool:
            grown = attach_bottom_partials(grid, partials_pool)
            new_cells = set(grown) - set(grid)
            if new_cells:
                used_ids = {id(grown[cell]) for cell in new_cells}
                partials_pool = [p for p in partials_pool if id(p) not in used_ids]
                partials_attached += len(new_cells)
                grid = grown
        canvas = render(grid)
        rows = max(r for r, _ in grid) + 1
        cols = max(c for _, c in grid) + 1
        idx = len(scene_records)
        name = f"scene_{idx:03d}_{n_tiles}tiles_{cols}x{rows}_conf{cost:.1f}.png"
        Image.fromarray(canvas, "RGBA").save(os.path.join(output, name))
        scene_records.append(
            {"image": name, "tiles": n_tiles, "grid_cols": cols, "grid_rows": rows, "mean_seam_cost": round(cost, 2)}
        )
        scene_canvases.append(canvas)

        # OCR the final reconstruction in its natural per-scene strips at full upscale
        # (so quality isn't lost shrinking one huge image), consolidated into one output.
        if do_ocr:
            words = ocr_image(canvas, languages=ocr_lang, min_confidence=ocr_min_conf, scale=ocr_scale)
            text = words_to_text(words)
            if text:
                final_text_parts.append(text)
                for rec in words_to_records(words, scale=ocr_scale):
                    rec["bbox"][1] += y_offset  # map y into montage coordinates
                    rec["scene"] = name
                    final_words.append(rec)
        y_offset += canvas.shape[0] + MONTAGE_PADDING
    rendered = len(scene_records)

    # Final reconstruction: one combined canvas of all scenes (most-confident first).
    final_record = None
    final_canvas = montage(scene_canvases, padding=MONTAGE_PADDING)
    if final_canvas.size:
        Image.fromarray(final_canvas, "RGBA").save(os.path.join(output, "final_reconstruction.png"))
        final_record = {
            "image": "final_reconstruction.png",
            "width": int(final_canvas.shape[1]),
            "height": int(final_canvas.shape[0]),
        }
        if do_ocr:
            final_text = "\n".join(final_text_parts)
            final_record["ocr_scale"] = ocr_scale
            final_record["ocr_text"] = final_text
            final_record["ocr_words"] = final_words
            with open(os.path.join(output, "final_reconstruction.txt"), "w", encoding="utf-8") as fh:
                fh.write(final_text)

    manifest = build_manifest(discover_cache_files(source), command="reconstruct")
    manifest.update(
        {
            "total_tiles": len(store.tiles),
            "unique_tiles": len(unique),
            "full_tiles_matched": len(full),
            "partial_tiles": partials_excluded,
            "partial_tiles_attached": partials_attached,
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
            "artifacts": (
                None
                if bundle is None
                else {
                    "resolution_source": bundle.resolution_source,
                    "width": bundle.width,
                    "height": bundle.height,
                    "multimon": bundle.multimon,
                    "os_major": bundle.os_major,
                    "hosts": bundle.hosts,
                    "event_count": len(bundle.events),
                    "sources": bundle.sources,
                }
            ),
            "final_reconstruction": final_record,
            "scenes": scene_records,
        }
    )
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    click.echo(
        f"Reconstructed {rendered} scene(s) from {len(full)} full tiles "
        f"({len(unique)} unique; {partials_attached}/{partials_excluded} partial edge tiles attached) "
        f"-> final_reconstruction.png; resolution candidates: {constraint.candidates or 'unknown'}"
    )
    if do_ocr and final_record:
        click.echo(f"OCR -> final_reconstruction.txt ({len(final_record.get('ocr_words', []))} words)")


if __name__ == "__main__":  # pragma: no cover
    main()
