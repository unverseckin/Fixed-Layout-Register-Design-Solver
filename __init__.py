"""
Fixed Layout Register Design Solver

A modular solver for the problem of assigning n items to m
pre-specified candidate sites so that the realized pairwise
interaction matrix (governed by an inverse-sixth-power distance
law) matches a target affinity matrix as closely as possible,
subject to hardware constraints on minimum inter-item spacing
and maximum distance from the workspace center.

Quick start:

    constraints = HardwareConstraints(c6=862690, r_min=5.0, r_max=35.0)
    sites = [Site(id=i, x=..., y=...) for i in range(m)]
    layout = SiteLayout(sites, constraints)

    target = TargetMatrix(matrix=Q)
    cost_fn = WeightedSquaredError(target)

    solver = FixedLayoutSolver(
        builder=GreedyBuilder(),
        optimizer=TwoOptSwapOptimizer(),
        cost_fn=cost_fn,
    )
    result = solver.solve(target, layout, num_restarts=20)

    report = ReportGenerator.generate(result, target, layout)
    print(ReportGenerator.format_text(report))
"""

from models.models import (
    Site,
    HardwareConstraints,
    TargetMatrix,
    SiteLayout,
)
from solution.assignment import Assignment
from algorithm.cost import CostFunction, WeightedSquaredError
from algorithm.builders import SolutionBuilder, GreedyBuilder
from algorithm.optimizers import (
    LocalSearchOptimizer,
    TwoOptSwapOptimizer,
    OptimizationResult,
)
from algorithm.solver import FixedLayoutSolver, SolverResult
from results.reporting import ReportGenerator, SolutionReport, PairReport

__all__ = [
    "Site",
    "HardwareConstraints",
    "TargetMatrix",
    "SiteLayout",
    "Assignment",
    "CostFunction",
    "WeightedSquaredError",
    "SolutionBuilder",
    "GreedyBuilder",
    "LocalSearchOptimizer",
    "TwoOptSwapOptimizer",
    "OptimizationResult",
    "FixedLayoutSolver",
    "SolverResult",
    "ReportGenerator",
    "SolutionReport",
    "PairReport",
]
