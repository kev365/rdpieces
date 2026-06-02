"""Tests for scene segmentation via connected components of best-buddy edges."""

from __future__ import annotations

from rdpieces.segmentation import connected_components


def test_components_group_linked_tiles_and_isolate_singletons():
    comps = connected_components(8, [(0, 1), (1, 2), (5, 6)])
    as_sets = sorted((sorted(c) for c in comps), key=lambda c: (-len(c), c))
    assert as_sets[0] == [0, 1, 2]
    assert [5, 6] in as_sets
    # singletons 3, 4, 7 each form their own component
    assert {tuple(c) for c in as_sets} >= {(0, 1, 2), (3,), (4,), (5, 6), (7,)}


def test_components_sorted_by_size_descending():
    comps = connected_components(5, [(0, 1), (1, 2), (3, 4)])
    sizes = [len(c) for c in comps]
    assert sizes == sorted(sizes, reverse=True)
    assert len(comps[0]) == 3
