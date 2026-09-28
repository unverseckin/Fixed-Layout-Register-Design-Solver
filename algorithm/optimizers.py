"""
Local search optimizers for improving an initial assignment.

Defines the abstract optimizer interface and a concrete 2-opt
(pairwise swap) local search implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from solution.assignment import Assignment
from algorithm.cost import CostFunction
from models.models import SiteLayout


@dataclass
class OptimizationResult:
    """Result of a local search optimization run.

    Attributes:
        assignment: The optimized assignment.
        cost: Final cost after optimization.
        iterations: Number of improvement passes performed.
        swaps_performed: Total number of improving swaps made.
    """

    assignment: Assignment
    cost: float
    iterations: int
    swaps_performed: int


class LocalSearchOptimizer(ABC):
    """Abstract interface for local search improvement."""

    @abstractmethod
    def optimize(
        self,
        assignment: Assignment,
        layout: SiteLayout,
        cost_fn: CostFunction,
        initial_cost: float,
    ) -> OptimizationResult:
        """Improve the given assignment via local search.

        Args:
            assignment: Starting assignment (modified in place).
            layout: Site layout with precomputed interactions.
            cost_fn: Cost function for evaluation.
            initial_cost: Pre-evaluated cost of the starting assignment.

        Returns:
            OptimizationResult with the improved assignment and metadata.
        """


class TwoOptSwapOptimizer(LocalSearchOptimizer):
    """2-opt local search: iteratively apply the best pairwise swap.

    Scans all item pairs in each pass. If swapping items i and j
    reduces the cost, the swap is accepted. Passes repeat until
    no improving swap exists or the iteration limit is reached.
    """

    def __init__(self, max_passes: int = 200, improvement_tolerance: float = 1e-12):
        """
        Args:
            max_passes: Maximum number of full scans over all pairs.
            improvement_tolerance: Minimum cost reduction to accept a swap.
        """
        self._max_passes = max_passes
        self._tolerance = improvement_tolerance

    def optimize(
        self,
        assignment: Assignment,
        layout: SiteLayout,
        cost_fn: CostFunction,
        initial_cost: float,
    ) -> OptimizationResult:
        current_cost = initial_cost
        n = assignment.n_items
        total_swaps = 0
        passes_done = 0

        for pass_num in range(self._max_passes):
            improved_this_pass = False

            for i in range(n):
                for j in range(i + 1, n):
                    delta = cost_fn.swap_delta(assignment, layout, i, j)
                    if delta < -self._tolerance:
                        assignment.swap(i, j)
                        current_cost += delta
                        total_swaps += 1
                        improved_this_pass = True

            passes_done = pass_num + 1
            if not improved_this_pass:
                break

        return OptimizationResult(
            assignment=assignment,
            cost=current_cost,
            iterations=passes_done,
            swaps_performed=total_swaps,
        )
