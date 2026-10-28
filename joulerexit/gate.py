"""Per-depth exit gates.

Each gate reads the current token's hidden state, the next-token entropy,
and the frozen energy residual of stopping at this depth. The last depth
always exits.
"""

from __future__ import annotations

import torch
from torch import nn


class ExitGate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 128)
        self.out = nn.Linear(130, 1)
        # Start reluctant to exit. The energy term has to earn an early exit.
        nn.init.constant_(self.out.bias, -1.5)

    def forward(self, hidden: torch.Tensor, entropy: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden must be [batch, hidden_size] for the current token")
        if entropy.shape != residual.shape or entropy.shape[0] != hidden.shape[0]:
            raise ValueError("entropy and residual must be one value per batch row")
        features = torch.cat(
            [torch.nn.functional.gelu(self.proj(hidden)), entropy[:, None], residual[:, None]],
            dim=-1,
        )
        return torch.sigmoid(self.out(features)).squeeze(-1)


class ExitGates(nn.Module):
    """One trainable gate per early exit. The final block has no gate."""

    def __init__(self, hidden_size: int, n_exits: int):
        super().__init__()
        if n_exits < 2:
            raise ValueError("need at least one early exit and the final block")
        self.gates = nn.ModuleList(ExitGate(hidden_size) for _ in range(n_exits - 1))

    def probability(self, exit_index: int, hidden: torch.Tensor, entropy: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if exit_index < 0 or exit_index > len(self.gates):
            raise IndexError(f"exit index {exit_index} is outside 0..{len(self.gates)}")
        if exit_index == len(self.gates):
            return torch.ones(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        return self.gates[exit_index](hidden, entropy, residual)
