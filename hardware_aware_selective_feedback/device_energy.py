"""Device-local GPU joules for a speculative depth and draft length.

The profile is fit on the GPU that produced the rows. A rejected draft is
wasted work only when its coefficient is positive and the separated fit beats
a token-count fit by more than the repeat noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTIONS = ((0, 0), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4))


@dataclass(frozen=True)
class DeviceRow:
    verify_steps: int
    accepted: int
    rejected: int
    joules: float
    setting: str

    def __post_init__(self) -> None:
        if self.verify_steps < 1 or self.accepted < 0 or self.rejected < 0:
            raise ValueError("verify steps must be positive and draft counts nonnegative")
        if not np.isfinite(self.joules) or self.joules < 0:
            raise ValueError("joules must be finite and nonnegative")


def _fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((target - predicted) ** 2)))
    return coefficients, rmse


def repeat_noise_j(rows: list[DeviceRow]) -> float:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row.setting, []).append(row.joules)
    deviations = [float(np.std(values, ddof=1)) for values in groups.values() if len(values) >= 2]
    if not deviations:
        raise ValueError("repeat noise needs two measurements of one setting")
    return float(np.median(deviations))


def fit_device(rows: list[DeviceRow]) -> dict:
    """Fit joules = a + b*verify_steps + c*accepted + d*rejected.

    The token-count model uses those three counts as one total.
    """
    if len(rows) < 8:
        raise ValueError("need at least 8 rows")
    if not any(row.rejected > 0 for row in rows) or not any(row.accepted > 0 for row in rows):
        raise ValueError("need both accepted and rejected draft tokens")
    target = np.asarray([row.joules for row in rows], dtype=np.float64)
    verify = np.asarray([row.verify_steps for row in rows], dtype=np.float64)
    accepted = np.asarray([row.accepted for row in rows], dtype=np.float64)
    rejected = np.asarray([row.rejected for row in rows], dtype=np.float64)
    full = np.column_stack([np.ones(len(rows)), verify, accepted, rejected])
    counted = np.column_stack([np.ones(len(rows)), verify + accepted + rejected])
    full_coef, full_rmse = _fit(full, target)
    _count_coef, count_rmse = _fit(counted, target)
    noise = repeat_noise_j(rows)
    rejected_j = float(full_coef[3])
    return {
        "intercept": float(full_coef[0]),
        "verify_j": float(full_coef[1]),
        "accepted_j": float(full_coef[2]),
        "rejected_j": rejected_j,
        "full_rmse_j": full_rmse,
        "token_count_rmse_j": count_rmse,
        "repeat_noise_j": noise,
        "rejected_is_waste": rejected_j > 0 and (count_rmse - full_rmse) > noise,
    }


def cycle_joules_per_token(profile: dict, length: int, accepted: int) -> float:
    """Joules per committed token for one decode cycle of this draft length."""
    if length == 0:
        if accepted != 0:
            raise ValueError("greedy accepts no draft tokens")
        return float(profile["verify_j"])
    if not 0 <= accepted <= length:
        raise ValueError("accepted count must fall inside the draft")
    joules = (
        profile["verify_j"]
        + profile["accepted_j"] * accepted
        + profile["rejected_j"] * (length - accepted)
    )
    return float(joules / (accepted + 1))


def step_winners(profile: dict, depth: int, accepted_counts: list[int]) -> list[tuple[int, int]]:
    """Best exact-output action at each full draft cycle, using this GPU's costs.

    A length-4 cycle's accepted prefix is the acceptance a shorter draft would
    have had. Greedy is always one of the options.
    """
    if depth <= 0:
        raise ValueError("step comparison needs the draft depth")
    winners = []
    for accepted in accepted_counts:
        if not 0 <= accepted <= 4:
            raise ValueError("length-4 cycles accept between 0 and 4 tokens")
        scores = {(0, 0): cycle_joules_per_token(profile, 0, 0)}
        for length in (1, 2, 4):
            scores[(depth, length)] = cycle_joules_per_token(profile, length, min(length, accepted))
        winners.append(min(scores, key=lambda action: (scores[action], action)))
    return winners


def startup_choice(measured: list[dict]) -> dict:
    """Lowest mean joules among actions that matched greedy on every prompt.

    measured rows need depth, length, joules, and matched.
    """
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in measured:
        grouped.setdefault((int(row["depth"]), int(row["length"])), []).append(row)
    eligible = []
    for action, rows in grouped.items():
        if rows and all(row["matched"] for row in rows):
            mean_j = float(np.mean([row["joules"] for row in rows]))
            eligible.append({"depth": action[0], "length": action[1], "mean_joules": mean_j, "runs": len(rows)})
    if not eligible:
        raise ValueError("no action matched greedy on every prompt")
    eligible.sort(key=lambda item: (item["mean_joules"], item["depth"], item["length"]))
    return eligible[0]


def fit_exit_profiles(measured: list[dict]) -> dict[int, dict]:
    """Fit verify, kept, and thrown-away joules separately at each draft exit.

    `measured` rows are the startup sweep records. Greedy rows are skipped.
    A row needs depth, verify_steps, accepted, rejected, and joules.
    """
    grouped: dict[int, list[DeviceRow]] = {}
    for row in measured:
        depth = int(row["depth"])
        if depth <= 0:
            continue
        grouped.setdefault(depth, []).append(
            DeviceRow(
                int(row["verify_steps"]),
                int(row["accepted"]),
                int(row["rejected"]),
                float(row["joules"]),
                f"{depth}/{int(row['length'])}",
            )
        )
    if not grouped:
        raise ValueError("the sweep has no draft exits to fit")
    return {depth: fit_device(rows) for depth, rows in sorted(grouped.items())}


def acceptance_steps(histogram: dict, depth: int) -> list[int]:
    """Expand a depth's acceptance histogram into one count per measured cycle."""
    steps = []
    for accepted in range(5):
        count = int(histogram.get(f"depth-{depth}-accepted-{accepted}", 0))
        steps.extend([accepted] * count)
    if not steps:
        raise ValueError(f"no acceptance counts for depth {depth}")
    return steps


def cheapest_changes(profile: dict, counts: list[int], depth: int) -> bool:
    """True when the lowest-joule action is not the same at every recorded step."""
    winners = step_winners(profile, depth, counts)
    return len(set(winners)) > 1


def train_head_decision(profile: dict, winners: list[tuple[int, int]]) -> dict:
    distinct = sorted(set(winners))
    varies = len(distinct) > 1
    train = bool(profile["rejected_is_waste"] and varies and winners)
    if train:
        reason = "rejected drafts are wasted work and the cheapest action changes across steps"
    elif not profile["rejected_is_waste"]:
        reason = "a rejected draft is not positive wasted work above the repeat noise"
    elif not varies:
        reason = "one action wins every decode step"
    else:
        reason = "no decode steps were available"
    return {
        "train_head": train,
        "actions_seen": [{"depth": depth, "length": length} for depth, length in distinct],
        "reason": reason,
    }
