"""
Reporting utilities for solution analysis.

Provides structured output of solution quality metrics,
including per-pair error breakdown and summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from models.models import SiteLayout, TargetMatrix
from algorithm.solver import SolverResult


@dataclass(frozen=True)
class PairReport:
    """Error report for a single item pair."""

    item_i: int
    item_j: int
    target_q: float
    realized_u: float
    absolute_error: float
    relative_error_pct: float


@dataclass
class SolutionReport:
    """Full quality report for a solved instance."""

    n_items: int
    n_sites: int
    total_cost: float
    total_absolute_error: float
    mean_relative_error_pct: float
    max_relative_error_pct: float
    pair_reports: List[PairReport]
    assignment_map: dict  # item_id -> (site_id, x, y)
    total_time: float
    restarts: int


class ReportGenerator:
    """Generates structured quality reports for a solution."""

    @staticmethod
    def generate(
        result: SolverResult,
        target: TargetMatrix,
        layout: SiteLayout,
    ) -> SolutionReport:
        """Produce a detailed quality report.

        Args:
            result: The solver result containing the best assignment.
            target: The target affinity matrix.
            layout: The site layout.

        Returns:
            A SolutionReport with per-pair and summary metrics.
        """
        assignment = result.best_assignment
        sigma = assignment.mapping
        n = target.n_items
        pair_reports = []
        total_abs_error = 0.0
        relative_errors = []

        for i in range(n):
            for j in range(i + 1, n):
                q = target.target(i, j)
                u = layout.interaction(sigma[i], sigma[j])
                abs_err = abs(u - q)
                rel_err = 100.0 * abs_err / abs(q) if abs(q) > 1e-12 else 0.0

                pair_reports.append(PairReport(
                    item_i=i,
                    item_j=j,
                    target_q=q,
                    realized_u=u,
                    absolute_error=abs_err,
                    relative_error_pct=rel_err,
                ))

                total_abs_error += abs_err
                relative_errors.append(rel_err)

        assignment_map = {}
        for i in range(n):
            s = sigma[i]
            site = layout.sites[s]
            assignment_map[i] = (site.id, site.x, site.y)

        return SolutionReport(
            n_items=n,
            n_sites=layout.n_sites,
            total_cost=result.best_cost,
            total_absolute_error=total_abs_error,
            mean_relative_error_pct=float(np.mean(relative_errors)) if relative_errors else 0.0,
            max_relative_error_pct=max(relative_errors) if relative_errors else 0.0,
            pair_reports=pair_reports,
            assignment_map=assignment_map,
            total_time=result.total_time,
            restarts=result.restarts_performed,
        )

    @staticmethod
    def format_text(report: SolutionReport) -> str:
        """Format a report as human-readable text."""
        lines = []
        lines.append("=" * 65)
        lines.append("SOLUTION REPORT")
        lines.append("=" * 65)
        lines.append(f"  Items: {report.n_items}    Sites: {report.n_sites}")
        lines.append(f"  Cost (weighted SSE): {report.total_cost:.6e}")
        lines.append(f"  Total absolute error: {report.total_absolute_error:.4f}")
        lines.append(f"  Mean relative error: {report.mean_relative_error_pct:.1f}%")
        lines.append(f"  Max  relative error: {report.max_relative_error_pct:.1f}%")
        lines.append(f"  Time: {report.total_time:.3f}s ({report.restarts} restarts)")
        lines.append("")
        lines.append("  Assignment:")
        for item_id in sorted(report.assignment_map):
            site_id, x, y = report.assignment_map[item_id]
            lines.append(f"    Item {item_id} -> Site {site_id} "
                         f"at ({x:.2f}, {y:.2f})")
        lines.append("")
        lines.append(f"  {'Pair':<8} {'Q_ij':>10} {'U_ij':>10} "
                     f"{'|Error|':>10} {'Rel%':>8}")
        lines.append(f"  {'-' * 50}")
        for pr in report.pair_reports:
            lines.append(
                f"  ({pr.item_i},{pr.item_j}){'':<3} "
                f"{pr.target_q:>10.4f} {pr.realized_u:>10.4f} "
                f"{pr.absolute_error:>10.4f} {pr.relative_error_pct:>7.1f}%"
            )
        lines.append("=" * 65)
        return "\n".join(lines)
