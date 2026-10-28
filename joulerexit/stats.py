"""Paired bootstrap for joules per correct answer."""

from __future__ import annotations

import numpy as np


def joules_per_correct(joules: np.ndarray, correct: np.ndarray) -> float | None:
    total = float(correct.sum())
    if total <= 0:
        return None
    return float(joules.sum() / total)


def paired_savings(baseline_j: np.ndarray, baseline_c: np.ndarray, other_j: np.ndarray, other_c: np.ndarray, seed: int = 0, draws: int = 2000) -> dict:
    """Bootstrap baseline_ratio - other_ratio. Positive means `other` uses fewer joules per correct."""
    if len(baseline_j) != len(other_j) or len(baseline_j) == 0:
        raise ValueError("paired bootstrap needs equal nonempty example lists")
    rng = np.random.default_rng(seed)
    count = len(baseline_j)
    samples = []
    for _ in range(draws):
        index = rng.integers(0, count, size=count)
        base = joules_per_correct(baseline_j[index], baseline_c[index])
        other = joules_per_correct(other_j[index], other_c[index])
        if base is None or other is None:
            continue
        samples.append(base - other)
    if len(samples) < draws // 2:
        return {"difference": None, "low": None, "high": None, "draws": len(samples)}
    values = np.sort(np.asarray(samples))
    return {
        "difference": float(np.mean(values)),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "draws": len(samples),
    }
