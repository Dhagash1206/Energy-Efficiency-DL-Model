"""Logged exit controller that is not part of the checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OutsideController:
    bias: float
    entropy: float
    residual: float
    depth: float

    def probability(self, entropy: float, residual: float, depth: float) -> float:
        score = self.bias + self.entropy * entropy + self.residual * residual + self.depth * depth
        return 1.0 / (1.0 + math.exp(-score))

    def halt(self, entropy: float, residual: float, depth: float) -> bool:
        return self.probability(entropy, residual, depth) > 0.5

    def to_dict(self) -> dict[str, float]:
        return {"bias": self.bias, "entropy": self.entropy, "residual": self.residual, "depth": self.depth}


def fit_controller(features: list[tuple[float, float, float]], labels: list[int]) -> OutsideController:
    """Fit halt = sigmoid(bias + a*entropy + b*residual + c*depth) on logged exits."""
    if len(features) < 8:
        raise ValueError("need at least 8 logged exits to fit the outside controller")
    inputs = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32)
    inputs = (inputs - inputs.mean(dim=0)) / inputs.std(dim=0).clamp(min=1e-4)
    # Keep the standardization inside the returned weights by folding it in after fitting.
    mean = torch.tensor(features, dtype=torch.float32).mean(dim=0)
    std = torch.tensor(features, dtype=torch.float32).std(dim=0).clamp(min=1e-4)
    weight = torch.zeros(4, requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=0.05)
    for _ in range(400):
        score = weight[0] + inputs[:, 0] * weight[1] + inputs[:, 1] * weight[2] + inputs[:, 2] * weight[3]
        loss = F.binary_cross_entropy_with_logits(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    raw = weight.detach()
    # score = b + w · ((x - mean) / std) = (b - sum w*mean/std) + sum (w/std) x
    bias = float(raw[0] - (raw[1:] * mean / std).sum())
    scaled = raw[1:] / std
    return OutsideController(bias, float(scaled[0]), float(scaled[1]), float(scaled[2]))
