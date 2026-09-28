"""
demo.py — Fixed Layout Register Design:
=======================================

Run from the project root:
    python demo.py

This script walks through the full pipeline:
  1. Define hardware constraints (C6, Rmin, Rmax)
  2. Build a set of candidate sites on a triangular grid
  3. Create a target affinity matrix Q
  4. Solve the assignment problem with the multi-start heuristic
  5. Print a detailed quality report

"""

import logging
import math
import sys

import numpy as np

from models.models import (
    Site,
    HardwareConstraints,
    TargetMatrix,
    SiteLayout,
)
from algorithm.cost import WeightedSquaredError
from algorithm.builders import GreedyBuilder
from algorithm.optimizers import TwoOptSwapOptimizer
from algorithm.solver import FixedLayoutSolver
from results.reporting import ReportGenerator

# ── Logging ──────────────────────────────────────────────
# Show solver progress (one line per restart).
# Set to WARNING to silence solver output.
logging.basicConfig(
    level=logging.INFO,
    format="  %(message)s",
    stream=sys.stdout,
)


def build_triangular_sites(
        r_max: float, spacing: float, r_min: float
) -> list[Site]:
    """Generate candidate sites on a triangular (hex) lattice.

    A triangular lattice packs more sites into a circular
    workspace than a square grid and is a common choice
    for neutral-atom devices.

    Args:
        r_max: Workspace radius — sites must lie within this
               distance of the origin.
        spacing: Distance between nearest-neighbour sites.
                 Must be >= r_min.
        r_min: Minimum allowed inter-site distance (for
               validation).

    Returns:
        A list of Site objects, each inside the r_max disk
        and at least r_min apart from every other site.
    """
    if spacing < r_min:
        raise ValueError(
            f"Lattice spacing ({spacing}) must be >= r_min ({r_min})"
        )

    sites: list[Site] = []
    sid = 0

    # Triangular lattice: offset every other row by half a step
    row_height = spacing * math.sqrt(3) / 2
    n_rows = int(2 * r_max / row_height) + 2

    for row in range(-n_rows, n_rows + 1):
        y = row * row_height
        x_offset = (spacing / 2) if (row % 2) else 0.0
        n_cols = int(2 * r_max / spacing) + 2

        for col in range(-n_cols, n_cols + 1):
            x = col * spacing + x_offset

            # Keep only sites inside the workspace disk
            if math.sqrt(x ** 2 + y ** 2) <= r_max:
                sites.append(Site(id=sid, x=round(x, 4), y=round(y, 4)))
                sid += 1

    return sites


def build_target_matrix(n: int, seed: int = 99) -> np.ndarray:
    """Create a synthetic symmetric target matrix.

    The matrix has a mix of strong and weak interactions
    to make the assignment non-trivial.  Diagonal is zero.

    In a real application this would come from the QUBO
    formulation of the combinatorial problem being solved.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.5, 8.0, size=(n, n))
    Q = (raw + raw.T) / 2
    np.fill_diagonal(Q, 0.0)
    return Q


def main() -> None:
    # ── 1. Hardware constraints ──────────────────────────
    C6 = 862_690.0  # Van der Waals constant (typical Rb-87)
    R_MIN = 5.0  # Minimum inter-atom distance (µm)
    R_MAX = 35.0  # Maximum distance from register centre (µm)

    constraints = HardwareConstraints(c6=C6, r_min=R_MIN, r_max=R_MAX)

    print("=" * 62)
    print("  Fixed Layout Register Design — demo")
    print("=" * 62)
    print(f"\n  Hardware constraints:")
    print(f"    C6    = {C6:,.0f}")
    print(f"    R_min = {R_MIN} µm")
    print(f"    R_max = {R_MAX} µm")
    print(f"    Achievable interaction range: "
          f"[{constraints.min_interaction:.4f}, "
          f"{constraints.max_interaction:.4f}]")

    # ── 2. Build candidate sites ─────────────────────────
    SPACING = 6.0  # µm between nearest neighbours
    sites = build_triangular_sites(R_MAX, SPACING, R_MIN)

    print(f"\n  Candidate sites: {len(sites)} "
          f"(triangular lattice, spacing={SPACING} µm)")
    print(f"    First 5 sites:")
    for s in sites[:5]:
        print(f"      Site {s.id:2d}: ({s.x:7.2f}, {s.y:7.2f})")
    if len(sites) > 5:
        print(f"      ... and {len(sites) - 5} more")

    layout = SiteLayout(sites, constraints)
    print(f"    All sites passed constraint validation.")

    # ── 3. Target affinity matrix ────────────────────────
    #N_ITEMS = 8
    #Q = build_target_matrix(N_ITEMS, seed=99)
    Q = np.array([[-17, 10, 10, 10, 0, 20],
				  [10, -18, 10, 10, 10, 20],
				  [10, 10, -29, 10, 20, 20],
				  [10, 10, 10, -19, 10, 10],
				  [0, 10, 20, 10, -17, 10],
				  [20, 20, 20, 10, 10, -28]])
    N_ITEMS = Q.shape[0]
    target = TargetMatrix(matrix=Q)

    print(f"\n  Target matrix Q ({N_ITEMS}×{N_ITEMS}):")
    col_header = "        " + "".join(f"   {j:>2d}   " for j in range(N_ITEMS))
    print(col_header)
    for i in range(N_ITEMS):
        row_vals = "  ".join(f"{Q[i, j]:6.2f}" for j in range(N_ITEMS))
        print(f"    {i:2d}  {row_vals}")

    # ── 4. Solve ─────────────────────────────────────────
    NUM_RESTARTS = 10
    SEED = 42

    ALPHA = 1.0  # GRASP RCL parameter (0=greedy, 1=random)

    cost_fn = WeightedSquaredError(target)
    solver = FixedLayoutSolver(
        builder=GreedyBuilder(alpha=ALPHA),
        optimizer=TwoOptSwapOptimizer(max_passes=200),
        cost_fn=cost_fn,
    )

    print(f"\n  Solving ({NUM_RESTARTS} restarts, seed={SEED}, "
          f"alpha={ALPHA})...")
    print(f"  {'-' * 56}")

    result = solver.solve(
        target=target,
        layout=layout,
        num_restarts=NUM_RESTARTS,
        seed=SEED,
    )

    # ── 5. Report ────────────────────────────────────────
    report = ReportGenerator.generate(result, target, layout)
    print(f"\n{ReportGenerator.format_text(report)}")

    # ── 6. Convergence summary ───────────────────────────
    print("\n  Convergence (best cost after each restart):")
    history = result.cost_history
    best_found_at = history.index(min(history)) + 1

    for r, cost in enumerate(history, 1):
        tag = "  <-- best" if r == best_found_at else ""
        print(f"    Restart {r:3d}: {cost:.6e}{tag}")

    print(f"\n  Best solution found at restart {best_found_at} "
          f"of {NUM_RESTARTS}.")
    print(f"  Total wall-clock time: {result.total_time:.3f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()
