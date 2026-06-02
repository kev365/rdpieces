"""Chain-of-custody evidence handling.

Inputs are treated as read-only evidence. This module hashes source files and
builds a deterministic run manifest so reconstructions are auditable and
reproducible. Deliberately excludes wall-clock time from the manifest body so
identical inputs + config yield byte-identical output.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable

from . import __version__


def sha256_file(path: str, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read-only and streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source_files: Iterable[str], *, command: str | None = None) -> dict:
    """Build a deterministic manifest describing the tool run and its inputs.

    Sources are sorted by path so the output is independent of discovery order.
    """
    sources = []
    for path in sorted(source_files):
        stat = os.stat(path)
        sources.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {
        "tool": "rdpieces",
        "version": __version__,
        "command": command,
        "sources": sources,
    }
