"""
Top-level solver for the Fixed Layout Register Design problem.

Orchestrates the multi-start heuristic pipeline:
  build initial solution → local search → repeat → keep best.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from solution.assignment import Assignment
from algorithm.builders import SolutionBuilder
from algorithm.cost import CostFunction
from models.models import SiteLayout, TargetMatrix
from algorithm.optimizers import LocalSearchOptimizer

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Complete result of a solver run.

    Attributes:
        best_assignment: The best assignment found across all restarts.
        best_cost: Cost of the best assignment.
        cost_history: Best cost found after each restart.
        total_time: Total wall-clock time in seconds.
        restarts_performed: Number of restarts completed.
    """

    best_assignment: Assignment
    best_cost: float
    cost_history: List[float] = field(default_factory=list)
    total_time: float = 0.0
    restarts_performed: int = 0


class FixedLayoutSolver:
    """Multi-start solver for the Fixed Layout Register Design problem.

    Combines a SolutionBuilder (for initial assignments) with a
    LocalSearchOptimizer (for improvement), running multiple
    independent restarts and returning the best solution found.

    This class follows the Strategy pattern: the builder and
    optimizer can be swapped independently without changing the
    solver's orchestration logic.
    """

    def __init__(
        self,
        builder: SolutionBuilder,
        optimizer: LocalSearchOptimizer,
        cost_fn: CostFunction,
    ):
        """
        Args:
            builder: Strategy for constructing initial assignments.
            optimizer: Strategy for improving assignments via local search.
            cost_fn: Cost function for evaluating assignments.
        """
        self._builder = builder
        self._optimizer = optimizer
        self._cost_fn = cost_fn

    def solve(
        self,
        target: TargetMatrix,
        layout: SiteLayout,
        num_restarts: int = 10,
        seed: int = 42,
    ) -> SolverResult:
        """Run the multi-start heuristic.

        Args:
            target: Target affinity matrix.
            layout: Site layout (pre-validated against constraints).
            num_restarts: Number of independent restarts.
            seed: Random seed for reproducibility.

        Returns:
            SolverResult containing the best solution and metadata.

        Raises:
            ValueError: If the problem instance is infeasible
                        (more items than sites).
        """
        self._validate_instance(target, layout)

        rng = np.random.default_rng(seed)
        best_assignment: Optional[Assignment] = None
        best_cost = float("inf")
        cost_history: List[float] = []

        t_start = time.perf_counter()

        for restart in range(num_restarts):
            t_iter = time.perf_counter()

            # Phase 1: construct initial solution
            assignment = self._builder.build(target, layout, rng)
            initial_cost = self._cost_fn.evaluate(assignment, layout)

            # Phase 2: improve via local search
            result = self._optimizer.optimize(
                assignment, layout, self._cost_fn, initial_cost
            )

            # Track best
            if result.cost < best_cost:
                best_cost = result.cost
                best_assignment = result.assignment.copy()

            cost_history.append(best_cost)

            elapsed = time.perf_counter() - t_iter
            logger.info(
                "Restart %d/%d: cost=%.6e (best=%.6e), "
                "%d swaps in %d passes, %.3fs",
                restart + 1,
                num_restarts,
                result.cost,
                best_cost,
                result.swaps_performed,
                result.iterations,
                elapsed,
            )

        total_time = time.perf_counter() - t_start

        return SolverResult(
            best_assignment=best_assignment,
            best_cost=best_cost,
            cost_history=cost_history,
            total_time=total_time,
            restarts_performed=num_restarts,
        )

    @staticmethod
    def _validate_instance(target: TargetMatrix, layout: SiteLayout) -> None:
        """Check that the instance is feasible."""
        if target.n_items > layout.n_sites:
            raise ValueError(
                f"Infeasible: {target.n_items} items cannot be assigned "
                f"to {layout.n_sites} sites (need m >= n)"
            )
