"""Separate GPU joules into a depth term and the leftover used by the gate.

Joules are fit as

    joules = a + b * depth + c * log(seq_len) + d * is_decode

The depth-only fit uses a and b. The gate feature is the difference between
the full prediction and the depth-only prediction. If length and phase do
not reduce error beyond repeat noise, that feature is only measurement noise
and the study stops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeRow:
    depth: int
    seq_len: int
    phase: str
    joules: float

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be prefill or decode, got {self.phase}")
        if self.depth < 1 or self.seq_len < 1:
            raise ValueError("depth and seq_len must be positive")
        if not math.isfinite(self.joules) or self.joules < 0:
            raise ValueError("joules must be finite and nonnegative")


@dataclass(frozen=True)
class EnergyProfile:
    """Frozen device profile. Coefficients are joules per unit of each input."""

    intercept: float
    depth_slope: float
    log_length_slope: float
    decode_offset: float
    depth_only_intercept: float
    depth_only_slope: float
    depth_only_rmse: float
    full_rmse: float
    repeat_noise_j: float

    def predicted_joules(self, depth: int, seq_len: int, phase: str) -> float:
        decode = 1.0 if phase == "decode" else 0.0
        return (
            self.intercept
            + self.depth_slope * depth
            + self.log_length_slope * math.log(seq_len)
            + self.decode_offset * decode
        )

    def depth_only_joules(self, depth: int) -> float:
        return self.depth_only_intercept + self.depth_only_slope * depth

    def residual(self, depth: int, seq_len: int, phase: str) -> float:
        """Joules predicted beyond an affine depth penalty."""
        return self.predicted_joules(depth, seq_len, phase) - self.depth_only_joules(depth)

    @property
    def residual_is_real(self) -> bool:
        """True when length and phase explain more error than repeat noise."""
        gain = self.depth_only_rmse - self.full_rmse
        return gain > self.repeat_noise_j


def _design(rows: list[ProbeRow], full: bool) -> np.ndarray:
    columns = []
    for row in rows:
        features = [1.0, float(row.depth)]
        if full:
            features.extend([math.log(row.seq_len), 1.0 if row.phase == "decode" else 0.0])
        columns.append(features)
    return np.asarray(columns, dtype=np.float64)


def _fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((target - predicted) ** 2)))
    return coefficients, rmse


def repeat_noise_j(rows: list[ProbeRow]) -> float:
    """Median within-cell standard deviation. A cell is one depth, length, and phase."""
    groups: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        groups.setdefault((row.depth, row.seq_len, row.phase), []).append(row.joules)
    deviations = []
    for values in groups.values():
        if len(values) >= 2:
            deviations.append(float(np.std(values, ddof=1)))
    if not deviations:
        raise ValueError("repeat noise needs at least two measurements in one cell")
    return float(np.median(deviations))


def fit_profile(rows: list[ProbeRow]) -> EnergyProfile:
    if len(rows) < 8:
        raise ValueError("need at least 8 probe rows to separate depth from length and phase")
    target = np.asarray([row.joules for row in rows], dtype=np.float64)
    full_coef, full_rmse = _fit(_design(rows, full=True), target)
    depth_coef, depth_rmse = _fit(_design(rows, full=False), target)
    return EnergyProfile(
        intercept=float(full_coef[0]),
        depth_slope=float(full_coef[1]),
        log_length_slope=float(full_coef[2]),
        decode_offset=float(full_coef[3]),
        depth_only_intercept=float(depth_coef[0]),
        depth_only_slope=float(depth_coef[1]),
        depth_only_rmse=depth_rmse,
        full_rmse=full_rmse,
        repeat_noise_j=repeat_noise_j(rows),
    )
