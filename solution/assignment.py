"""
Solution representation for the Fixed Layout Register Design problem.

An Assignment maps n items to m candidate sites (one-to-one),
with validation that the mapping is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Assignment:
    """A one-to-one mapping of n items to m sites.

    Attributes:
        mapping: Array of length n where mapping[i] = site index for item i.
        n_items: Number of items.
        n_sites: Number of available sites.
    """

    mapping: np.ndarray
    n_sites: int

    def __post_init__(self):
        self.mapping = np.array(self.mapping, dtype=int)

    @property
    def n_items(self) -> int:
        return len(self.mapping)

    def site_of(self, item: int) -> int:
        """Return the site assigned to the given item."""
        return int(self.mapping[item])

    def swap(self, item_i: int, item_j: int) -> None:
        """Swap the site assignments of two items in place."""
        self.mapping[item_i], self.mapping[item_j] = (
            self.mapping[item_j],
            self.mapping[item_i],
        )

    def copy(self) -> Assignment:
        """Return a deep copy of this assignment."""
        return Assignment(
            mapping=self.mapping.copy(),
            n_sites=self.n_sites,
        )

    def is_valid(self) -> bool:
        """Check that the assignment is a valid one-to-one mapping.

        Returns:
            True if every item maps to a distinct site within [0, n_sites).
        """
        if len(self.mapping) == 0:
            return True
        if np.any(self.mapping < 0) or np.any(self.mapping >= self.n_sites):
            return False
        return len(set(self.mapping)) == len(self.mapping)

    def used_sites(self) -> set:
        """Return the set of site indices currently in use."""
        return set(self.mapping.tolist())

    def available_sites(self) -> list:
        """Return a sorted list of site indices not currently assigned."""
        used = self.used_sites()
        return [s for s in range(self.n_sites) if s not in used]
