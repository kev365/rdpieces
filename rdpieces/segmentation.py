"""Scene segmentation.

Rather than pre-clustering a bag of tiles, scenes emerge from reconstructability:
tiles that strongly edge-match link into a component, and disconnected components
are separate screens/regions. This reuses the matcher's best-buddy edges and keeps
each placement problem small.
"""

from __future__ import annotations


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Group 0..n-1 into connected components, sorted by size (largest first)."""
    ds = _DisjointSet(n)
    for a, b in edges:
        ds.union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(ds.find(i), []).append(i)
    return sorted(groups.values(), key=lambda c: (-len(c), c[0]))
