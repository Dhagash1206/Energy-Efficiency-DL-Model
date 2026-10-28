"""Separate GPU joules of committed tokens from rejected draft tokens.

A reject term that does not beat a token-count fit is a length penalty.
The draft head is not an energy model in that case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RejectRow:
    committed: int
    rejected: int
    joules: float
    setting: str

    def __post_init__(self) -> None:
        if self.committed < 1 or self.rejected < 0:
            raise ValueError("committed must be positive and rejected nonnegative")
        if not np.isfinite(self.joules) or self.joules < 0:
            raise ValueError("joules must be finite and nonnegative")


def _fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((target - predicted) ** 2)))
    return coefficients, rmse


def repeat_noise_j(rows: list[RejectRow]) -> float:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row.setting, []).append(row.joules)
    deviations = [float(np.std(values, ddof=1)) for values in groups.values() if len(values) >= 2]
    if not deviations:
        raise ValueError("repeat noise needs two measurements of one setting")
    return float(np.median(deviations))


def fit_reject(rows: list[RejectRow]) -> dict:
    """Fit joules = a + b*committed + c*rejected against a token-count model.

    The token-count model uses committed + rejected as one count, so c is
    forced equal to b. The reject term is real when it cuts RMSE by more
    than the repeat noise.
    """
    if len(rows) < 8:
        raise ValueError("need at least 8 rows")
    if not any(row.rejected > 0 for row in rows):
        raise ValueError("need at least one rejected draft token")
    committed_values = {row.committed for row in rows}
    if len(committed_values) < 2:
        raise ValueError("committed length does not vary, so the reject term is not identified")
    target = np.asarray([row.joules for row in rows], dtype=np.float64)
    committed = np.asarray([row.committed for row in rows], dtype=np.float64)
    rejected = np.asarray([row.rejected for row in rows], dtype=np.float64)
    full = np.column_stack([np.ones(len(rows)), committed, rejected])
    counted = np.column_stack([np.ones(len(rows)), committed + rejected])
    full_coef, full_rmse = _fit(full, target)
    _count_coef, count_rmse = _fit(counted, target)
    noise = repeat_noise_j(rows)
    return {
        "intercept": float(full_coef[0]),
        "committed_j": float(full_coef[1]),
        "rejected_j": float(full_coef[2]),
        "full_rmse_j": full_rmse,
        "token_count_rmse_j": count_rmse,
        "repeat_noise_j": noise,
        "reject_is_real": (count_rmse - full_rmse) > noise,
    }
