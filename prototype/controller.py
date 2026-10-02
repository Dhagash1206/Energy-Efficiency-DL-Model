"""Action selection using illustrative energy costs, never GPU measurements."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Action:
    name: str
    exit_depth: int
    draft_length: int
    energy_j: float
    latency_ms: float
    relative_margin: float
    context_cost: float
    acceptance_prior: float

    def __post_init__(self):
        numbers = (self.energy_j, self.latency_ms, self.relative_margin,
                   self.context_cost, self.acceptance_prior)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Action parameters must be finite")
        if self.energy_j <= 0 or self.latency_ms <= 0 or self.context_cost < 0:
            raise ValueError("Costs must be positive and context cost nonnegative")
        if not 0 <= self.relative_margin < 1 or not 0 <= self.acceptance_prior <= 1:
            raise ValueError("Invalid uncertainty margin or acceptance prior")
        if self.draft_length < 0 or self.exit_depth < 0:
            raise ValueError("Depth and draft length cannot be negative")
        if (self.draft_length == 0) != (self.exit_depth == 0):
            raise ValueError("Ordinary decoding uses depth=0 and draft_length=0")

    @property
    def ordinary(self):
        return self.draft_length == 0

    def cycle_costs(self, context_length):
        factor = 1 + self.context_cost * context_length
        return self.energy_j * factor, self.latency_ms * factor


@dataclass
class State:
    context_length: int
    remaining_tokens: int
    history: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=16)))

    def acceptance(self, action):
        observations = self.history[action.exit_depth]
        # Two prior pseudo-observations prevent one rejection from dominating.
        return (sum(observations) + 2 * action.acceptance_prior) / (len(observations) + 2)

    def observe(self, action, checks):
        if not action.ordinary:
            self.history[action.exit_depth].extend(checks)


def expected_committed_tokens(probability, draft_length):
    """Accepted prefix plus correction/bonus; assumes independent acceptance."""
    if not 0 <= probability <= 1 or draft_length < 0:
        raise ValueError("Invalid acceptance probability or draft length")
    return sum(probability ** index for index in range(draft_length + 1))


class Controller:
    def __init__(self, actions, overhead_j=0.025, overhead_ms=0.15):
        self.actions = tuple(actions)
        if not all(math.isfinite(value) and value >= 0 for value in (overhead_j, overhead_ms)):
            raise ValueError("Controller overhead must be finite and nonnegative")
        self.overhead_j = overhead_j
        self.overhead_ms = overhead_ms
        ordinary = [action for action in self.actions if action.ordinary]
        if len(ordinary) != 1:
            raise ValueError("Exactly one ordinary-decoding action is required")
        self.ordinary = ordinary[0]

    def estimates(self, state):
        if state.context_length < 0 or state.remaining_tokens <= 0:
            raise ValueError("Context must be nonnegative and token budget positive")
        rows = []
        for action in self.actions:
            # Reserve one position for the correction or bonus token.
            if action.draft_length >= state.remaining_tokens:
                continue
            acceptance = 1.0 if action.ordinary else state.acceptance(action)
            committed = expected_committed_tokens(acceptance, action.draft_length)
            energy, latency = action.cycle_costs(state.context_length)
            rows.append({
                "action": action,
                "acceptance_estimate": acceptance,
                "expected_committed_tokens": committed,
                "energy_per_token": (energy + self.overhead_j) / committed,
                "energy_upper": (energy * (1 + action.relative_margin) + self.overhead_j) / committed,
                "energy_lower": (energy * (1 - action.relative_margin) + self.overhead_j) / committed,
                "latency_per_token_ms": (latency + self.overhead_ms) / committed,
            })
        return rows

    def choose(self, state, objective="energy"):
        rows = self.estimates(state)
        ordinary = next(row for row in rows if row["action"].ordinary)
        if objective == "latency":
            chosen = min(rows, key=lambda row: row["latency_per_token_ms"])
            return chosen, "lowest predicted latency per committed token"
        if objective != "energy":
            raise ValueError("Objective must be energy or latency")
        candidates = [row for row in rows if not row["action"].ordinary
                      and row["energy_upper"] < ordinary["energy_lower"]]
        if not candidates:
            return ordinary, "ordinary fallback: no margin-separated predicted saving"
        return min(candidates, key=lambda row: row["energy_upper"]), "predicted saving exceeds illustrative energy margin"
