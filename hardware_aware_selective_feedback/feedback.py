"""Feedback validation and per-request match-rate accounting.

Observations are one-step teacher-forced predictions from an intermediate
layer.  Only positions that fall within the committed prefix of a cycle are
valid evidence; rejected or post-stop positions are marked invalid.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def validate_feedback(
    context: list[int],
    draft: list[int],
    target: list[int],
    observations: dict[int, list[int]],
    accepted: int,
    committed_count: int,
) -> list[dict[str, Any]]:
    """Build feedback rows for all observer predictions in one cycle.

    For each (depth, position) pair, a row records:
      - whether the position is valid evidence (within the committed prefix)
      - whether the observer's prediction matched the target
      - the full prefix context used to produce the prediction

    Args:
        context:         Prompt + all previously committed tokens.
        draft:           Draft token IDs proposed this cycle.
        target:          Target token IDs produced by the verifier.
        observations:    {depth: [predicted_token_id, ...]} from observer hooks.
        accepted:        Number of draft tokens accepted by the verifier.
        committed_count: Total tokens committed this cycle (accepted + 1).

    Returns:
        List of row dicts, one per (depth, position) pair.
    """
    rows = []
    for depth, predictions in observations.items():
        if len(predictions) != len(target):
            raise ValueError(
                f"Observer at depth {depth} produced {len(predictions)} predictions "
                f"but target has {len(target)} positions."
            )
        for position, (prediction, truth) in enumerate(zip(predictions, target)):
            valid = position <= accepted and position < committed_count
            rows.append({
                "depth":      depth,
                "position":   position,
                "prefix_ids": context + draft[:position],
                "prediction": prediction,
                "target":     truth,
                "valid":      valid,
                "match":      (prediction == truth) if valid else None,
                "reason":     "committed_context" if valid else "rejected_or_after_stop",
                "evidence":   "one_step_only",
                "counterfactual_energy_j": None,
            })
    return rows


class LearningState:
    """Accumulate one-step observer match statistics across cycles.

    Tracks (matches, observations) per exit depth.  Only valid feedback rows
    (those within the committed prefix) are counted.

    This is a descriptive statistic only — not a rollout or energy estimator.
    """

    def __init__(self) -> None:
        self._counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    def update(self, rows: list[dict[str, Any]]) -> None:
        """Add valid rows from one cycle to the running totals."""
        for row in rows:
            if row["valid"]:
                bucket = self._counts[row["depth"]]
                bucket[0] += int(row["match"])
                bucket[1] += 1

    def export(self) -> dict[str, dict[str, Any]]:
        """Return per-depth statistics as a JSON-serialisable dict."""
        return {
            str(depth): {"matches": m, "observations": n, "rate": m / n}
            for depth, (m, n) in self._counts.items()
            if n > 0
        }
