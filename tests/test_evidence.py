"""Tests for chain-of-custody evidence handling."""

from __future__ import annotations

import hashlib

from rdpieces.evidence import build_manifest, sha256_file


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    assert sha256_file(str(p)) == hashlib.sha256(b"hello world").hexdigest()


def test_build_manifest_records_sources_and_command(tmp_path):
    payload = b"\x00" * 10
    p = tmp_path / "Cache0000.bin"
    p.write_bytes(payload)

    manifest = build_manifest([str(p)], command="extract")

    assert manifest["tool"] == "rdpieces"
    assert "version" in manifest
    assert manifest["command"] == "extract"
    assert len(manifest["sources"]) == 1
    src = manifest["sources"][0]
    assert src["sha256"] == hashlib.sha256(payload).hexdigest()
    assert src["size"] == 10
    assert src["path"].endswith("Cache0000.bin")


def test_manifest_is_deterministic_for_same_inputs(tmp_path):
    p = tmp_path / "Cache0000.bin"
    p.write_bytes(b"abc")
    assert build_manifest([str(p)], command="extract") == build_manifest([str(p)], command="extract")
