"""
Domain models for the Fixed Layout Register Design problem.

This module defines the core value objects and data containers
that represent the problem instance: sites, hardware constraints,
the target affinity matrix, and the site layout with precomputed
interactions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class Site:
    """A candidate trapping site with fixed 2D coordinates.

    Attributes:
        id: Unique identifier for the site.
        x: Horizontal coordinate (micrometers).
        y: Vertical coordinate (micrometers).
    """

    id: int
    x: float
    y: float

    def distance_to(self, other: Site) -> float:
        """Euclidean distance to another site."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_origin(self) -> float:
        """Euclidean distance from the workspace center (origin)."""
        return math.sqrt(self.x ** 2 + self.y ** 2)


@dataclass(frozen=True)
class HardwareConstraints:
    """Physical constraints imposed by the hardware.

    Attributes:
        c6: Van der Waals interaction constant (device-specific).
        r_min: Minimum allowed distance between any pair of atoms.
        r_max: Maximum allowed distance from the workspace center.
    """

    c6: float
    r_min: float
    r_max: float

    def __post_init__(self):
        if self.c6 <= 0:
            raise ValueError(f"c6 must be positive, got {self.c6}")
        if self.r_min <= 0:
            raise ValueError(f"r_min must be positive, got {self.r_min}")
        if self.r_max <= 0:
            raise ValueError(f"r_max must be positive, got {self.r_max}")
        if self.r_min >= 2 * self.r_max:
            raise ValueError(
                f"r_min ({self.r_min}) must be less than the maximum possible "
                f"pairwise distance 2*r_max ({2 * self.r_max})"
            )

    def interaction_at_distance(self, distance: float) -> float:
        """Compute the interaction strength at a given distance.

        U = C6 / r^6

        Args:
            distance: Euclidean distance between two atoms.

        Returns:
            The interaction energy.

        Raises:
            ValueError: If distance is zero or negative.
        """
        if distance <= 0:
            raise ValueError(f"Distance must be positive, got {distance}")
        return self.c6 / distance ** 6

    @property
    def max_interaction(self) -> float:
        """Maximum achievable interaction (atoms at minimum distance)."""
        return self.interaction_at_distance(self.r_min)

    @property
    def min_interaction(self) -> float:
        """Minimum achievable interaction (atoms at maximum distance 2*r_max)."""
        return self.interaction_at_distance(2 * self.r_max)


@dataclass(frozen=True)
class TargetMatrix:
    """Symmetric target affinity matrix Q.

    Encapsulates the desired pairwise interaction strengths
    and optional per-pair weights for cost evaluation.

    Attributes:
        matrix: Symmetric n x n matrix of target interactions.
        weights: Optional n x n matrix of per-pair weights.
                 Defaults to uniform weights (all ones).
    """

    matrix: np.ndarray
    weights: np.ndarray = field(default=None)

    def __post_init__(self):
        n = self.matrix.shape[0]
        if self.matrix.shape != (n, n):
            raise ValueError(f"Target matrix must be square, got {self.matrix.shape}")
        if self.weights is None:
            object.__setattr__(self, "weights", np.ones((n, n), dtype=float))
        elif self.weights.shape != (n, n):
            raise ValueError(
                f"Weight matrix shape {self.weights.shape} must match "
                f"target matrix shape {self.matrix.shape}"
            )

    @property
    def n_items(self) -> int:
        """Number of items to assign."""
        return self.matrix.shape[0]

    def target(self, i: int, j: int) -> float:
        """Target interaction between items i and j."""
        return self.matrix[i, j]

    def weight(self, i: int, j: int) -> float:
        """Weight for the (i, j) pair."""
        return self.weights[i, j]


class SiteLayout:
    """A collection of candidate sites with precomputed pairwise interactions.

    Validates all sites against hardware constraints at construction time
    and precomputes the full interaction matrix for efficient cost evaluation.

    Attributes:
        sites: List of candidate sites.
        constraints: Hardware constraints governing the layout.
        interaction_matrix: Precomputed m x m matrix of site-to-site interactions.
        distance_matrix: Precomputed m x m matrix of site-to-site distances.
    """

    def __init__(self, sites: List[Site], constraints: HardwareConstraints):
        """
        Args:
            sites: Candidate sites. Must all lie within r_max of the origin
                   and be pairwise separated by at least r_min.
            constraints: Hardware constraints.

        Raises:
            ValueError: If any site violates a constraint.
        """
        self._sites = list(sites)
        self._constraints = constraints
        self._validate_sites()
        self._distance_matrix, self._interaction_matrix = self._precompute()

    def _validate_sites(self) -> None:
        """Verify every site satisfies hardware constraints."""
        for site in self._sites:
            dist = site.distance_to_origin()
            if dist > self._constraints.r_max + 1e-9:
                raise ValueError(
                    f"Site {site.id} at ({site.x:.4f}, {site.y:.4f}) is "
                    f"{dist:.4f} from center, exceeding r_max={self._constraints.r_max}"
                )

        for i, si in enumerate(self._sites):
            for j, sj in enumerate(self._sites):
                if j <= i:
                    continue
                d = si.distance_to(sj)
                if d < self._constraints.r_min - 1e-9:
                    raise ValueError(
                        f"Sites {si.id} and {sj.id} are {d:.4f} apart, "
                        f"violating r_min={self._constraints.r_min}"
                    )

    def _precompute(self) -> tuple:
        """Build distance and interaction matrices."""
        m = self.n_sites
        dist = np.zeros((m, m))
        interact = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                d = self._sites[i].distance_to(self._sites[j])
                dist[i, j] = d
                dist[j, i] = d
                u = self._constraints.interaction_at_distance(d)
                interact[i, j] = u
                interact[j, i] = u
        return dist, interact

    @property
    def sites(self) -> List[Site]:
        return list(self._sites)

    @property
    def constraints(self) -> HardwareConstraints:
        return self._constraints

    @property
    def n_sites(self) -> int:
        return len(self._sites)

    def interaction(self, s: int, t: int) -> float:
        """Precomputed interaction between sites s and t."""
        return self._interaction_matrix[s, t]

    def distance(self, s: int, t: int) -> float:
        """Precomputed distance between sites s and t."""
        return self._distance_matrix[s, t]

    @property
    def interaction_matrix(self) -> np.ndarray:
        return self._interaction_matrix.copy()

    @property
    def distance_matrix(self) -> np.ndarray:
        return self._distance_matrix.copy()
