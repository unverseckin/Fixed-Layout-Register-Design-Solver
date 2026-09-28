"""
Initial solution construction strategies.

Defines the abstract builder interface and a concrete GRASP-style
randomized greedy construction heuristic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from solution.assignment import Assignment
from models.models import SiteLayout, TargetMatrix


class SolutionBuilder(ABC):
	"""Abstract interface for constructing an initial assignment."""

	@abstractmethod
	def build(
			self,
			target: TargetMatrix,
			layout: SiteLayout,
			rng: np.random.Generator,
	) -> Assignment:
		"""Construct an initial feasible assignment.

        Args:
            target: The target affinity matrix.
            layout: The site layout with precomputed interactions.
            rng: Random number generator for randomized decisions.

        Returns:
            A feasible Assignment.
        """


class GreedyBuilder(SolutionBuilder):
	"""GRASP-style randomized greedy construction.

    Instead of always assigning items to the single best site pair,
    this builder maintains a Restricted Candidate List (RCL) of
    near-best options and selects randomly from it. This injects
    controlled diversity across restarts while preserving solution
    quality.

    The RCL is controlled by the alpha parameter:
      - alpha = 0.0: purely greedy (always pick the best) — deterministic
      - alpha = 1.0: purely random (all candidates equally likely)
      - alpha in (0, 1): GRASP — RCL includes candidates within
        alpha * (worst - best) of the best cost

    Strategy:
      1. Rank all item pairs (i,j) by |Q_ij| descending.
      2. For each pair, evaluate all available site assignments,
         build the RCL, and pick one randomly.
      3. Assign any remaining items to remaining sites randomly.
    """

	def __init__(self, alpha: float = 0.3):
		"""
        Args:
            alpha: RCL tolerance parameter in [0, 1].
                   0.0 = purely greedy (deterministic),
                   1.0 = purely random.
                   Values around 0.2–0.4 give good diversity
                   while maintaining solution quality.

        Raises:
            ValueError: If alpha is outside [0, 1].
        """
		if not 0.0 <= alpha <= 1.0:
			raise ValueError(f"alpha must be in [0, 1], got {alpha}")
		self._alpha = alpha

	@property
	def alpha(self) -> float:
		return self._alpha

	def build(
			self,
			target: TargetMatrix,
			layout: SiteLayout,
			rng: np.random.Generator,
	) -> Assignment:
		n = target.n_items
		m = layout.n_sites

		if n > m:
			raise ValueError(
				f"Cannot assign {n} items to {m} sites (need m >= n)"
			)

		mapping = np.full(n, -1, dtype=int)
		item_assigned = np.zeros(n, dtype=bool)
		site_used = np.zeros(m, dtype=bool)

		# Sort pairs by target interaction strength (strongest first)
		pairs = []
		for i in range(n):
			for j in range(i + 1, n):
				pairs.append((abs(target.target(i, j)), i, j))
		pairs.sort(reverse=True, key=lambda x: x[0])

		for _, i, j in pairs:
			if item_assigned[i] and item_assigned[j]:
				continue

			available = np.where(~site_used)[0]
			if len(available) == 0:
				break

			if not item_assigned[i] and not item_assigned[j]:
				self._assign_pair(
					i, j, target, layout, available,
					mapping, item_assigned, site_used, rng,
				)
			elif not item_assigned[i]:
				self._assign_single(
					i, j, target, layout, available,
					mapping, item_assigned, site_used, rng,
				)
			elif not item_assigned[j]:
				self._assign_single(
					j, i, target, layout, available,
					mapping, item_assigned, site_used, rng,
				)

		# Assign any remaining items to remaining sites randomly
		unassigned = np.where(~item_assigned)[0]
		remaining = np.where(~site_used)[0]
		rng.shuffle(remaining)
		for idx, item in enumerate(unassigned):
			mapping[item] = remaining[idx]
			item_assigned[item] = True
			site_used[remaining[idx]] = True

		return Assignment(mapping=mapping, n_sites=m)

	def _select_from_rcl(
			self,
			candidates: list[tuple[float, tuple]],
			rng: np.random.Generator,
	) -> tuple:
		"""Build the Restricted Candidate List and select one entry.

        Args:
            candidates: List of (cost, data) tuples, where lower
                        cost is better and data is whatever the
                        caller needs back (site indices).
            rng: Random generator.

        Returns:
            The data tuple of the selected candidate.
        """
		if not candidates:
			return None

		costs = [c for c, _ in candidates]
		best = min(costs)
		worst = max(costs)
		threshold = best + self._alpha * (worst - best)

		rcl = [data for cost, data in candidates if cost <= threshold]
		idx = rng.integers(0, len(rcl))
		return rcl[idx]

	def _assign_pair(
			self,
			i: int,
			j: int,
			target: TargetMatrix,
			layout: SiteLayout,
			available: np.ndarray,
			mapping: np.ndarray,
			item_assigned: np.ndarray,
			site_used: np.ndarray,
			rng: np.random.Generator,
	) -> None:
		"""Assign both items i and j using RCL selection over site pairs."""
		q_ij = target.target(i, j)

		# Evaluate all available site pairs
		candidates = []
		for s in available:
			for t in available:
				if s == t:
					continue
				cost = (layout.interaction(s, t) - q_ij) ** 2
				candidates.append((cost, (int(s), int(t))))

		selected = self._select_from_rcl(candidates, rng)
		if selected is not None:
			best_s, best_t = selected
			mapping[i] = best_s
			mapping[j] = best_t
			item_assigned[i] = True
			item_assigned[j] = True
			site_used[best_s] = True
			site_used[best_t] = True

	def _assign_single(
			self,
			unassigned_item: int,
			assigned_partner: int,
			target: TargetMatrix,
			layout: SiteLayout,
			available: np.ndarray,
			mapping: np.ndarray,
			item_assigned: np.ndarray,
			site_used: np.ndarray,
			rng: np.random.Generator,
	) -> None:
		"""Assign one item using RCL selection over single sites."""
		partner_site = mapping[assigned_partner]
		q = target.target(unassigned_item, assigned_partner)

		candidates = []
		for s in available:
			cost = (layout.interaction(s, partner_site) - q) ** 2
			candidates.append((cost, (int(s),)))

		selected = self._select_from_rcl(candidates, rng)
		if selected is not None:
			(best_site,) = selected
			mapping[unassigned_item] = best_site
			item_assigned[unassigned_item] = True
			site_used[best_site] = True