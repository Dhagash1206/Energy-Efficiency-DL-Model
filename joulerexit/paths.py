"""Soft and sampled exit paths.

A path is a distribution over exit depths for each token. Prompt tokens are
forced to the last exit so the question is encoded at every depth. Answer
tokens may leave early. Later blocks attend only to tokens that reached them.
"""

from __future__ import annotations

import torch


def soft_paths(early_probs: torch.Tensor) -> torch.Tensor:
    """Convert per-exit halting probabilities into a path over every exit.

    early_probs: [batch, time, n_early], each the probability of halting now
    given that the token is still active. Returns [batch, time, n_exits].
    """
    if early_probs.ndim != 3:
        raise ValueError("early_probs must be [batch, time, n_early]")
    remain = torch.ones(early_probs.shape[:2], device=early_probs.device, dtype=early_probs.dtype)
    steps = []
    for index in range(early_probs.shape[-1]):
        take = remain * early_probs[:, :, index]
        steps.append(take)
        remain = remain * (1.0 - early_probs[:, :, index])
    steps.append(remain)
    return torch.stack(steps, dim=-1)


def hard_paths(early_probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one exit per token. Probabilities are not differentiated here."""
    remain = torch.ones(early_probs.shape[:2], device=early_probs.device, dtype=early_probs.dtype)
    steps = []
    for index in range(early_probs.shape[-1]):
        draw = torch.rand(early_probs.shape[:2], device=early_probs.device, generator=generator)
        take_now = (draw < early_probs[:, :, index]).to(early_probs.dtype)
        steps.append(remain * take_now)
        remain = remain * (1.0 - take_now)
    steps.append(remain)
    return torch.stack(steps, dim=-1)


def straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Hard sample in the forward pass, soft probabilities in the backward pass."""
    return hard.detach() - soft.detach() + soft


def reached_at(paths: torch.Tensor, exit_index: int) -> torch.Tensor:
    """True where the token has not halted before this exit. Shape [batch, time]."""
    return paths[:, :, exit_index:].sum(dim=-1) > 0.5
