"""
Cost evaluation for the Fixed Layout Register Design problem.

Defines the abstract cost function interface and a concrete
weighted-squared-error implementation with an efficient
swap-delta computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from solution.assignment import Assignment
from models.models import SiteLayout, TargetMatrix


class CostFunction(ABC):
    """Abstract interface for evaluating an assignment's quality.

    Subclasses must implement full evaluation and incremental
    swap-delta computation.
    """

    @abstractmethod
    def evaluate(self, assignment: Assignment, layout: SiteLayout) -> float:
        """Compute the total cost of an assignment.

        Args:
            assignment: Current item-to-site mapping.
            layout: Site layout with precomputed interactions.

        Returns:
            The total cost (lower is better).
        """

    @abstractmethod
    def swap_delta(
        self,
        assignment: Assignment,
        layout: SiteLayout,
        item_i: int,
        item_j: int,
    ) -> float:
        """Compute the cost change from swapping two items, without performing it.

        Args:
            assignment: Current item-to-site mapping.
            layout: Site layout with precomputed interactions.
            item_i: First item to swap.
            item_j: Second item to swap.

        Returns:
            The change in cost (negative means improvement).
        """


class WeightedSquaredError(CostFunction):
    """Weighted sum of squared errors between target and realized interactions.

    F(σ) = Σ_{i<j} w_ij · (U_{σ(i),σ(j)} − Q_ij)²
    """

    def __init__(self, target: TargetMatrix):
        """
        Args:
            target: The target affinity matrix with optional weights.
        """
        self._target = target

    @property
    def target(self) -> TargetMatrix:
        return self._target

    def evaluate(self, assignment: Assignment, layout: SiteLayout) -> float:
        n = self._target.n_items
        sigma = assignment.mapping
        cost = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                u_ij = layout.interaction(sigma[i], sigma[j])
                q_ij = self._target.target(i, j)
                w_ij = self._target.weight(i, j)
                cost += w_ij * (u_ij - q_ij) ** 2

        return cost

    def swap_delta(
        self,
        assignment: Assignment,
        layout: SiteLayout,
        item_i: int,
        item_j: int,
    ) -> float:
        """
        Only terms involving item_i or item_j change when their
        sites are swapped. The direct (i,j) term is invariant
        since U(s_i, s_j) = U(s_j, s_i).
        """
        n = self._target.n_items
        sigma = assignment.mapping
        si, sj = sigma[item_i], sigma[item_j]
        delta = 0.0

        for k in range(n):
            if k == item_i or k == item_j:
                continue
            sk = sigma[k]

            # Contribution of (item_i, k) pair
            idx = (min(item_i, k), max(item_i, k))
            w = self._target.weight(*idx)
            q = self._target.target(item_i, k)
            old_u = layout.interaction(si, sk)
            new_u = layout.interaction(sj, sk)
            delta += w * ((new_u - q) ** 2 - (old_u - q) ** 2)

            # Contribution of (item_j, k) pair
            idx = (min(item_j, k), max(item_j, k))
            w = self._target.weight(*idx)
            q = self._target.target(item_j, k)
            old_u = layout.interaction(sj, sk)
            new_u = layout.interaction(si, sk)
            delta += w * ((new_u - q) ** 2 - (old_u - q) ** 2)

        return delta
