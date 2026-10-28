"""Draft-config head. The backbone stays frozen.

The head reads the layer-4 hidden state of the token about to be extended
and picks a speculative action. The label is the action with the lowest
device-profile joules given how many tokens of a length-4 draft were accepted.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hardware_aware_selective_feedback.device_energy import cycle_joules_per_token
from hardware_aware_selective_feedback.vendor.self_speculation.llama_model_utils import forward_early

ACTIONS = ((0, 0), (4, 1), (4, 2), (4, 4))


class ConfigHead(nn.Module):
    def __init__(self, hidden_size: int, actions: tuple[tuple[int, int], ...] = ACTIONS):
        super().__init__()
        if len(actions) < 2:
            raise ValueError("the head needs at least two actions")
        self.actions = tuple(actions)
        self.proj = nn.Linear(hidden_size, 128)
        self.out = nn.Linear(128, len(self.actions))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden must be [batch, hidden_size]")
        return self.out(F.gelu(self.proj(hidden)))

    def action_at(self, hidden: torch.Tensor) -> tuple[int, int]:
        index = int(self.forward(hidden.view(1, -1)).argmax(dim=-1))
        return self.actions[index]


def action_costs(profile: dict, accepted: int, actions: tuple[tuple[int, int], ...] = ACTIONS) -> list[float]:
    costs = []
    for depth, length in actions:
        if length == 0:
            costs.append(cycle_joules_per_token(profile, 0, 0))
        else:
            costs.append(cycle_joules_per_token(profile, length, min(length, accepted)))
    return costs


def head_loss(logits: torch.Tensor, labels: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Action cross-entropy plus the expected joules of the predicted action.

    The backbone is frozen, so next-token loss on the committed tokens does
    not reach this head. The acceptance prefix is what selects the label, and
    the cost row is the device profile's joules for each action at that step.
    """
    if costs.shape != logits.shape or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels and costs must match the action logits")
    expected = (logits.softmax(dim=-1) * costs).sum(dim=-1).mean()
    return F.cross_entropy(logits, labels) + expected


def probe_hidden(model, pending: torch.Tensor, cache, depth: int = 4) -> torch.Tensor:
    """Layer-`depth` hidden state of the pending token, without keeping that forward."""
    early = forward_early(model, pending, cache, depth, None)
    return early.exit_query_cache[0, -1].detach()
