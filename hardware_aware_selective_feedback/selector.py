"""Request-level joint selection of decoding action and observer exits.

The Selector chooses both the speculative decoding configuration (depth,
draft length) and which intermediate layers to observe for each request.
It learns from executed-cost measurements, not from counterfactual rollouts.

The CostBandit is an independent-arm lower-confidence-bound baseline that
uses only executed costs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log, sqrt
from statistics import mean, stdev
from typing import Optional


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """A fully-specified decoding action: depth, draft length, and observer exits.

    depth=0, length=0 means ordinary (non-speculative) greedy decoding.
    depth>0 activates LayerSkip with `length` draft tokens per cycle.
    `observations` lists intermediate layer depths to hook for feedback.
    """
    depth: int = 0
    length: int = 0
    observations: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (self.depth == 0) != (self.length == 0):
            raise ValueError("Ordinary decoding requires depth=0 and length=0.")
        if self.depth < 0 or self.length < 0:
            raise ValueError("Depth and draft length must be nonnegative.")
        if len(set(self.observations)) != len(self.observations):
            raise ValueError("Observer exit depths must be unique.")

    @property
    def key(self) -> tuple[int, int]:
        """Hashable (depth, length) identifier used as dict key."""
        return (self.depth, self.length)


# ---------------------------------------------------------------------------
# Selector: cost-aware heuristic with optional observer value-of-information
# ---------------------------------------------------------------------------

class Selector:
    """Cost-aware heuristic action selector with calibrated observer scheduling.

    Tracks per-token executed cost for each arm.  Observer exits are scheduled
    only when their measured overhead is justified by the uncertainty reduction
    they provide.

    Note: No formal safety or value-of-information guarantee.  The calibration
    uses repeated same-prompt pairs, which leaves thermal and ordering noise.
    """

    def __init__(
        self,
        depths: tuple[int, ...] = (4, 8, 12),
        lengths: tuple[int, ...] = (1, 3),
        objective: str = "latency",
    ) -> None:
        if objective not in ("latency", "energy"):
            raise ValueError("Objective must be 'latency' or 'energy'.")
        self.depths = tuple(depths)
        self.arms: tuple[Action, ...] = (Action(),) + tuple(
            Action(d, k) for d in depths for k in lengths
        )
        self.objective = objective

        # Per-arm executed costs (J or s) per token, without observers.
        self._costs: dict[tuple, list[float]] = defaultdict(list)
        # Same, but runs that included paid observers.
        self._observed_costs: dict[tuple, list[float]] = defaultdict(list)
        # One-step acceptance evidence: [successes, total] per depth.
        self._matches: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        # Measured per-token overhead of paid observers (paired deltas).
        self._observer_cost: dict[str, list[float]] = defaultdict(list)
        self.decisions = 0

    # ── Public interface ────────────────────────────────────────────────

    def update(self, action: Action, measured: dict, output: dict) -> None:
        """Record the cost and feedback from an executed request."""
        key = "energy_j" if self.objective == "energy" else "latency_s"
        value = measured[key]
        if value is None:
            raise ValueError("Selected objective has no sensor reading.")
        tokens = len(output["tokens"])
        if tokens < 1:
            raise ValueError("No generated tokens to score.")

        cost_per_token = value / tokens
        (self._observed_costs if action.observations else self._costs)[action.key].append(
            cost_per_token
        )

        # Update one-step acceptance statistics from cycle observations.
        for cycle in output["cycles"]:
            draft    = cycle.get("draft", [])
            accepted = cycle.get("accepted", 0)
            if action.depth and draft:
                bucket = self._matches[action.depth]
                bucket[0] += accepted
                bucket[1] += accepted + int(accepted < len(draft))
            for item in cycle["observations"]:
                if item["valid"]:
                    bucket = self._matches[item["depth"]]
                    bucket[0] += int(item["match"])
                    bucket[1] += 1

    def record_observer_pair(
        self,
        base: tuple,
        observed: tuple,
        mode: str,
    ) -> None:
        """Record the measured overhead of one paired base/observed run.

        Args:
            base:     (output, measured) from the run without observers.
            observed: (output, measured) from the run with observers.
            mode:     'selected' or 'all'.
        """
        if mode not in ("selected", "all"):
            raise ValueError(f"Unknown observer mode: {mode!r}")
        key = "energy_j" if self.objective == "energy" else "latency_s"
        base_cost, obs_cost = base[1][key], observed[1][key]
        if base_cost is not None and obs_cost is not None:
            self._observer_cost[mode].append(
                obs_cost / len(observed[0]["tokens"])
                - base_cost / len(base[0]["tokens"])
            )

    def choose(self, remaining_requests: int = 1) -> Action:
        """Select the action (and optional paid exits) for the next request."""
        arm = min(self.arms, key=self._arm_score)
        self.decisions += 1

        # Observer exits are only offered when running depth=8, length=3 and
        # the overhead has been calibrated (>= 3 pairs) and is cost-justified.
        if arm.key != (8, 3) or remaining_requests <= 1:
            return arm

        selected_exits = (4,)
        all_exits      = (4, 12)
        if not set(all_exits).issubset(self.depths):
            return arm

        best_cost = min((mean(v) for v in self._costs.values() if v), default=1.0)

        def _justified(exits: tuple, mode: str) -> bool:
            overhead = self._observer_cost[mode]
            if len(overhead) < 3:
                return False
            value = best_cost * max(self._uncertainty(d) for d in exits) * min(remaining_requests, 8)
            cost  = mean(overhead)
            return value > max(0.0, cost) * 2 and cost < best_cost * 0.25

        if _justified(all_exits, "all"):
            return Action(arm.depth, arm.length, all_exits)
        if _justified(selected_exits, "selected"):
            return Action(arm.depth, arm.length, selected_exits)
        return arm

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of accumulated statistics."""
        return {
            "objective":  self.objective,
            "decisions":  self.decisions,
            "executed_costs_per_token": {
                str(k): {"mean": mean(v), "n": len(v)}
                for k, v in self._costs.items() if v
            },
            "observed_request_costs_per_token": {
                str(k): {"mean": mean(v), "n": len(v)}
                for k, v in self._observed_costs.items() if v
            },
            "valid_one_step": {
                str(d): {"matches": m, "observations": n, "rate": m / n}
                for d, (m, n) in self._matches.items() if n
            },
            "measured_observer_deltas_per_token": {
                mode: {"mean": mean(v), "n": len(v)}
                for mode, v in self._observer_cost.items() if v
            },
            "decision_rule": (
                "Heuristic: minimum observed cost plus uncertainty penalty. "
                "Paid exits scheduled only at depth=8 length=3 after ≥3 calibration pairs. "
                "No formal safety guarantee."
            ),
        }

    # ── Private helpers ─────────────────────────────────────────────────

    def _uncertainty(self, depth: int) -> float:
        """Bayesian uncertainty on acceptance rate at `depth` (Laplace smoothing)."""
        successes, n = self._matches[depth]
        p = (successes + 1) / (n + 2)
        return sqrt(p * (1 - p) / (n + 3))

    def _arm_score(self, arm: Action) -> float:
        """Lower score = preferred arm.  Returns -inf for unexecuted arms."""
        values = self._costs[arm.key]
        if not values:
            return float("-inf")  # force each arm to execute at least once
        avg   = mean(values)
        noise = stdev(values) / sqrt(len(values)) if len(values) > 1 else avg * 0.10
        uncertainty_penalty = avg * self._uncertainty(arm.depth) * 0.10 if arm.depth else 0.0
        return avg + noise + uncertainty_penalty


# ---------------------------------------------------------------------------
# CostBandit: independent-arm LCB baseline
# ---------------------------------------------------------------------------

class CostBandit:
    """Lower-confidence-bound bandit over a fixed arm set.

    Only executed costs are observed; there is no observer overhead or
    acceptance-rate tracking.  Used as a simpler comparison baseline.
    """

    def __init__(
        self,
        arms: tuple[Action, ...],
        objective: str = "latency",
        exploration: float = 0.5,
    ) -> None:
        self.arms = tuple(arms)
        self.objective = objective
        self.exploration = exploration
        self._costs: dict[tuple, list[float]] = defaultdict(list)

    def update(self, action: Action, measured: dict, output: dict) -> None:
        """Record executed cost for the chosen arm."""
        key = "energy_j" if self.objective == "energy" else "latency_s"
        value = measured[key]
        if value is None or not output["tokens"]:
            raise ValueError("CostBandit requires measured cost and generated tokens.")
        self._costs[action.key].append(value / len(output["tokens"]))

    def choose(self) -> Action:
        """Return the arm with the lowest LCB score; unvisited arms go first."""
        for arm in self.arms:
            if not self._costs[arm.key]:
                return arm
        total = sum(len(v) for v in self._costs.values())
        scale = mean(mean(v) for v in self._costs.values())
        return min(
            self.arms,
            key=lambda arm: (
                mean(self._costs[arm.key])
                - self.exploration * scale * sqrt(log(total + 1) / len(self._costs[arm.key]))
            ),
        )

    def summary(self) -> dict:
        return {str(k): {"mean": mean(v), "n": len(v)} for k, v in self._costs.items() if v}
