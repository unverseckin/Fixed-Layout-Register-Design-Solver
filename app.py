"""
Flask web application for the Fixed Layout Register Design Solver.

Provides a browser-based interface for:
  - Configuring hardware constraints and solver parameters
  - Generating candidate site layouts (triangular lattice)
  - Inputting or randomly generating a target Q matrix
  - Running the multi-start GRASP + 2-opt solver
  - Visualizing the solution layout and convergence history
"""

import logging
import math

import numpy as np
from flask import Flask, jsonify, render_template, request

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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_triangular_sites(
    r_max: float, spacing: float, r_min: float
) -> list[Site]:
    """Generate candidate sites on a triangular (hex) lattice within a circle.

    Args:
        r_max: Maximum distance from center.
        spacing: Distance between adjacent lattice points.
        r_min: Minimum allowed inter-site distance (for validation).

    Returns:
        List of Site objects.
    """
    coords = []
    row = 0
    y = -r_max
    while y <= r_max:
        x_offset = (spacing / 2) * (row % 2)
        x = -r_max + x_offset
        while x <= r_max:
            if math.sqrt(x**2 + y**2) <= r_max + 1e-9:
                coords.append((x, y))
            x += spacing
        y += spacing * math.sqrt(3) / 2
        row += 1

    # Filter out pairs violating r_min
    filtered = []
    for cx, cy in coords:
        ok = True
        for fx, fy in filtered:
            if math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2) < r_min - 1e-9:
                ok = False
                break
        if ok:
            filtered.append((cx, cy))

    return [Site(id=i, x=x, y=y) for i, (x, y) in enumerate(filtered)]


def build_random_q(n: int, q_min: float, q_max: float, seed: int) -> np.ndarray:
    """Generate a random symmetric target matrix."""
    rng = np.random.default_rng(seed)
    upper = rng.uniform(q_min, q_max, size=(n, n))
    Q = (upper + upper.T) / 2
    np.fill_diagonal(Q, 0)
    return Q


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate-sites", methods=["POST"])
def generate_sites():
    """Generate candidate sites and return their coordinates."""
    data = request.get_json()
    try:
        r_max = float(data["r_max"])
        spacing = float(data["spacing"])
        r_min = float(data["r_min"])

        sites = build_triangular_sites(r_max, spacing, r_min)
        return jsonify({
            "sites": [{"id": s.id, "x": round(s.x, 4), "y": round(s.y, 4)}
                       for s in sites],
            "count": len(sites),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/generate-q", methods=["POST"])
def generate_q():
    """Generate a random symmetric Q matrix."""
    data = request.get_json()
    try:
        n = int(data["n"])
        q_min = float(data.get("q_min", 0.5))
        q_max = float(data.get("q_max", 50.0))
        seed = int(data.get("seed", 42))

        Q = build_random_q(n, q_min, q_max, seed)
        return jsonify({
            "matrix": Q.round(4).tolist(),
            "n": n,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/solve", methods=["POST"])
def solve():
    """Run the solver and return results."""
    data = request.get_json()
    try:
        # Parse hardware constraints
        c6 = float(data["c6"])
        r_min = float(data["r_min"])
        r_max = float(data["r_max"])
        constraints = HardwareConstraints(c6=c6, r_min=r_min, r_max=r_max)

        # Parse sites
        sites_data = data["sites"]
        sites = [Site(id=s["id"], x=s["x"], y=s["y"]) for s in sites_data]
        layout = SiteLayout(sites, constraints)

        # Parse Q matrix
        Q = np.array(data["q_matrix"], dtype=float)
        n = Q.shape[0]

        if n > layout.n_sites:
            return jsonify({
                "error": f"Need at least {n} sites for {n} items, "
                         f"but only {layout.n_sites} sites available."
            }), 400

        target = TargetMatrix(matrix=Q)

        # Solver parameters
        alpha = float(data.get("alpha", 0.3))
        num_restarts = int(data.get("num_restarts", 20))
        max_passes = int(data.get("max_passes", 200))
        seed = int(data.get("seed", 42))

        # Build and run solver
        cost_fn = WeightedSquaredError(target)
        solver = FixedLayoutSolver(
            builder=GreedyBuilder(alpha=alpha),
            optimizer=TwoOptSwapOptimizer(max_passes=max_passes),
            cost_fn=cost_fn,
        )

        result = solver.solve(
            target=target,
            layout=layout,
            num_restarts=num_restarts,
            seed=seed,
        )

        # Generate report
        report = ReportGenerator.generate(result, target, layout)

        # Build response
        sigma = result.best_assignment.mapping

        # Assigned items with positions
        assigned_items = []
        for i in range(n):
            s = int(sigma[i])
            site = sites[s]
            assigned_items.append({
                "item": i,
                "site_id": site.id,
                "x": round(site.x, 4),
                "y": round(site.y, 4),
            })

        # Pair-wise comparison
        pairs = []
        for pr in report.pair_reports:
            pairs.append({
                "i": pr.item_i,
                "j": pr.item_j,
                "target": round(pr.target_q, 6),
                "realized": round(pr.realized_u, 6),
                "abs_error": round(pr.absolute_error, 6),
                "rel_error_pct": round(pr.relative_error_pct, 2),
            })

        return jsonify({
            "success": True,
            "summary": {
                "n_items": report.n_items,
                "n_sites": report.n_sites,
                "total_cost": report.total_cost,
                "total_absolute_error": round(report.total_absolute_error, 4),
                "mean_relative_error_pct": round(report.mean_relative_error_pct, 2),
                "max_relative_error_pct": round(report.max_relative_error_pct, 2),
                "time_seconds": round(report.total_time, 3),
                "restarts": report.restarts,
            },
            "assigned_items": assigned_items,
            "pairs": pairs,
            "convergence": result.cost_history,
            "all_sites": [{"id": s.id, "x": round(s.x, 4), "y": round(s.y, 4)}
                          for s in sites],
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.exception("Solver error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
