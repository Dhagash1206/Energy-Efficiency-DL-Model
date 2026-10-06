"""Request-level joint selection of decoding action and paid exit observations.

Only executed requests yield cost measurements. Intermediate exit predictions
update one-step acceptance statistics, never imaginary alternative rollouts.
"""
from collections import defaultdict
from dataclasses import dataclass
from math import log, sqrt
from statistics import mean, stdev


@dataclass(frozen=True)
class Action:
    depth: int = 0
    length: int = 0
    observations: tuple[int, ...] = ()

    def __post_init__(self):
        if (self.depth == 0) != (self.length == 0):
            raise ValueError("Ordinary decoding requires depth=length=0")
        if self.depth < 0 or self.length < 0:
            raise ValueError("Depth and draft length must be nonnegative")
        if len(set(self.observations)) != len(self.observations):
            raise ValueError("Observation exits must be unique")

    @property
    def key(self):
        return (self.depth, self.length)


class Selector:
    """Cost-aware heuristic; no formal safety or value-of-information guarantee."""

    def __init__(self, depths=(4, 8, 12), lengths=(1, 3), objective="latency"):
        if objective not in ("latency", "energy"):
            raise ValueError("Objective must be latency or energy")
        self.depths = tuple(depths)
        self.arms = (Action(),) + tuple(Action(d, k) for d in depths for k in lengths)
        self.objective = objective
        self.costs = defaultdict(list)  # executed costs without paid exit observers
        self.observed_costs = defaultdict(list)  # executed costs with observers
        self.matches = defaultdict(lambda: [0, 0])  # valid one-step evidence
        self.observer_cost = defaultdict(list)  # paired, measured per-token deltas
        self.decisions = 0

    def update(self, action, measured, output):
        value = measured["energy_j"] if self.objective == "energy" else measured["latency_s"]
        if value is None:
            raise ValueError("Selected objective has no sensor reading")
        tokens = len(output["tokens"])
        if tokens < 1:
            raise ValueError("No generated tokens to score")
        (self.observed_costs if action.observations else self.costs)[action.key].append(value / tokens)
        for cycle in output["cycles"]:
            # The chosen depth reveals a valid sequential prefix and at most
            # its first mismatch. Later drafted tokens are not evidence.
            draft = cycle.get("draft", [])
            accepted = cycle.get("accepted", 0)
            if action.depth and draft:
                counts = self.matches[action.depth]
                counts[0] += accepted
                counts[1] += accepted + int(accepted < len(draft))
            for item in cycle["observations"]:
                if item["valid"]:
                    counts = self.matches[item["depth"]]
                    counts[0] += int(item["match"])
                    counts[1] += 1

    def record_observer_pair(self, base, observed, mode):
        """Same prompt and decoding arm; no invented counterfactual cost."""
        if mode not in ("selected", "all"):
            raise ValueError(mode)
        key = "energy_j" if self.objective == "energy" else "latency_s"
        a, b = base[1][key], observed[1][key]
        if a is not None and b is not None:
            self.observer_cost[mode].append(
                b / len(observed[0]["tokens"]) - a / len(base[0]["tokens"])
            )

    def uncertainty(self, depth):
        successes, n = self.matches[depth]
        p = (successes + 1) / (n + 2)
        return sqrt(p * (1 - p) / (n + 3))

    def action_score(self, arm):
        values = self.costs[arm.key]
        if not values:
            return float("-inf")  # execute each arm at least once
        average = mean(values)
        noise = stdev(values) / sqrt(len(values)) if len(values) > 1 else average * 0.10
        return average + noise + (average * self.uncertainty(arm.depth) * 0.10
                                   if arm.depth else 0)

    def choose(self, remaining_requests):
        """Choose an executed-cost arm and a paid-observation set for one request."""
        arm = min(self.arms, key=self.action_score)
        self.decisions += 1
        # Observer overhead was calibrated only for this executed arm and
        # these exact exit sets. Never transfer that cost to another arm/exit.
        if arm.key != (8, 3) or remaining_requests <= 1:
            return arm
        selected_exits = (4,)
        all_exits = (4, 12)
        if not set(all_exits).issubset(self.depths):
            return arm
        best_cost = min((mean(v) for v in self.costs.values() if v), default=1.0)
        selected_value = best_cost * self.uncertainty(4) * min(remaining_requests, 8)
        all_value = best_cost * max(self.uncertainty(4), self.uncertainty(12)) * min(remaining_requests, 8)
        selected_cost = mean(self.observer_cost["selected"]) if self.observer_cost["selected"] else float("inf")
        all_cost = mean(self.observer_cost["all"]) if self.observer_cost["all"] else float("inf")
        if len(self.observer_cost["all"]) >= 3 and all_value > max(0.0, all_cost) * 2 and all_cost < best_cost * 0.25:
            exits = all_exits
        elif len(self.observer_cost["selected"]) >= 3 and selected_value > max(0.0, selected_cost) and selected_cost < best_cost * 0.25:
            exits = selected_exits
        else:
            exits = ()
        return Action(arm.depth, arm.length, exits)

    def summary(self):
        return {
            "objective": self.objective,
            "decisions": self.decisions,
            "executed_costs_per_token": {
                str(k): {"mean": mean(v), "n": len(v)}
                for k, v in self.costs.items() if v
            },
            "observed_request_costs_per_token": {
                str(k): {"mean": mean(v), "n": len(v)}
                for k, v in self.observed_costs.items() if v
            },
            "valid_one_step": {
                str(d): {"matches": m, "observations": n, "rate": m / n}
                for d, (m, n) in self.matches.items() if n
            },
            "measured_observer_deltas_per_token": {
                mode: {"mean": mean(v), "n": len(v)}
                for mode, v in self.observer_cost.items() if v
            },
            "decision_rule": "heuristic measured base cost plus uncertainty; paid exits only at calibrated depth 8 length 3 after three overhead pairs; no safety guarantee",
        }


class CostBandit:
    """Independent-arm LCB baseline: only executed cost is observed."""

    def __init__(self, arms, objective="latency", exploration=0.5):
        self.arms = tuple(arms)
        self.objective = objective
        self.exploration = exploration
        self.costs = defaultdict(list)

    def update(self, action, measured, output):
        key = "energy_j" if self.objective == "energy" else "latency_s"
        value = measured[key]
        if value is None or not output["tokens"]:
            raise ValueError("Bandit requires measured cost and generated tokens")
        self.costs[action.key].append(value / len(output["tokens"]))

    def choose(self):
        for arm in self.arms:
            if not self.costs[arm.key]:
                return arm
        total = sum(len(v) for v in self.costs.values())
        scale = mean(mean(v) for v in self.costs.values())
        return min(self.arms, key=lambda arm: mean(self.costs[arm.key]) -
                   self.exploration * scale * sqrt(log(total + 1) / len(self.costs[arm.key])))

    def summary(self):
        return {str(k): {"mean": mean(v), "n": len(v)}
                for k, v in self.costs.items() if v}
