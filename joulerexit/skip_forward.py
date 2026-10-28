"""Teacher-forced forward with the inference attention mask.

Every token is still run through later blocks so each exit has a real
next-token distribution for the loss. Keys of a token that sampled an
earlier exit are hidden from later blocks, matching generation, where a
deeper token cannot attend to a token that never reached that depth.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from joulerexit.paths import soft_paths, straight_through
from joulerexit.residual import EnergyProfile


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    log_p = logits.float() - log_z
    return -(log_p.exp() * log_p).sum(dim=-1)


def _costs(profile: EnergyProfile, depth: int, seq_len: torch.Tensor, is_decode: torch.Tensor):
    length = seq_len.clamp(min=1).float()
    decode = is_decode.float()
    predicted = (
        profile.intercept
        + profile.depth_slope * depth
        + profile.log_length_slope * torch.log(length)
        + profile.decode_offset * decode
    )
    depth_only = torch.full_like(
        predicted,
        profile.depth_only_intercept + profile.depth_only_slope * depth,
    )
    return predicted, predicted - depth_only, depth_only


def _attention_mask(reached: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask. Active tokens see only keys that reached this depth.

    A token that already exited is not a key. Its own query attends to itself
    so the softmax stays finite; that output is discarded.
    """
    time = reached.shape[1]
    causal = torch.tril(torch.ones(time, time, device=reached.device, dtype=torch.bool))
    eye = torch.eye(time, device=reached.device, dtype=torch.bool)
    allowed = (causal.unsqueeze(0) & reached[:, None, :]) | ((~reached)[:, :, None] & eye.unsqueeze(0))
    mask = torch.zeros(reached.shape[0], 1, time, time, device=reached.device, dtype=dtype)
    blocked = -1.0e4 if dtype in (torch.float16, torch.bfloat16) else torch.finfo(dtype).min
    return mask.masked_fill(~allowed.unsqueeze(1), blocked)


def _gate_feature(mode: str, residual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    if mode == "residual":
        return residual
    if mode == "raw":
        return predicted
    if mode in ("depth", "entropy"):
        return torch.zeros_like(residual)
    raise ValueError(f"Unknown gate mode {mode}")


def forward_exits(
    model,
    gates,
    profile: EnergyProfile,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    exits: tuple[int, ...],
    mode: str,
    lambda_energy: float,
    generator: torch.Generator | None = None,
    entropy_tau: float = 1.5,
    detach_backbone: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the training loss and detached statistics.

    `labels` use -100 on prompt tokens. Those tokens always run every block.
    Answer tokens sample an exit and later blocks do not attend to them.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, time]")
    if mode not in ("residual", "depth", "raw", "entropy"):
        raise ValueError(f"Unknown gate mode {mode}")
    base = model.model
    device = input_ids.device
    batch, time = input_ids.shape
    position_ids = torch.arange(time, device=device).unsqueeze(0).expand(batch, -1)
    hidden = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(hidden, position_ids=position_ids)
    is_prompt = labels < 0
    seq_len = position_ids + 1
    is_decode = ~is_prompt

    exit_logits: list[torch.Tensor] = []
    early_probs: list[torch.Tensor] = []
    hard_steps: list[torch.Tensor] = []
    cost_steps: list[torch.Tensor] = []
    remain = torch.ones(batch, time, device=device, dtype=torch.float32)
    previous = 0

    for index, depth in enumerate(exits):
        reached = remain > 0.5
        mask = _attention_mask(reached, hidden.dtype)
        block_ctx = torch.no_grad() if detach_backbone else nullcontext()
        with block_ctx:
            for layer in base.layers[previous:depth]:
                updated = layer(
                    hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                updated = updated[0] if isinstance(updated, tuple) else updated
                hidden = updated
            logits = model.lm_head(base.norm(hidden))
        if detach_backbone:
            hidden = hidden.detach()
            logits = logits.detach()
        exit_logits.append(logits)
        previous = depth
        predicted, residual, depth_only = _costs(profile, depth, seq_len, is_decode)
        if mode == "entropy":
            cost_steps.append(torch.zeros_like(predicted))
        elif mode == "depth":
            cost_steps.append(depth_only)
        else:
            cost_steps.append(predicted)
        if index == len(exits) - 1:
            hard_steps.append(remain)
            break
        ent = _entropy(logits)
        feature = _gate_feature(mode, residual, predicted)
        probs = gates.probability(
            index,
            hidden.reshape(batch * time, hidden.shape[-1]).float(),
            ent.reshape(batch * time).float(),
            feature.reshape(batch * time).float(),
        ).view(batch, time)
        probs = torch.where(is_prompt, torch.zeros_like(probs), probs)
        early_probs.append(probs)
        draw = torch.rand(batch, time, device=device, generator=generator)
        take_now = (draw < probs.detach()).to(torch.float32)
        hard_steps.append(remain * take_now)
        remain = remain * (1.0 - take_now)

    soft = soft_paths(torch.stack(early_probs, dim=-1))
    hard = torch.stack(hard_steps, dim=-1)
    mix = straight_through(hard, soft)
    costs = torch.stack(cost_steps, dim=-1)
    energy = (mix * costs).sum(dim=-1)
    mixed_logits = (mix.unsqueeze(-1) * torch.stack(exit_logits, dim=-2)).sum(dim=-2)

    shift_logits = mixed_logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    supervised = shift_labels != -100
    if not bool(supervised.any()):
        raise ValueError("batch has no answer tokens")
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(batch, time - 1)
    ce = token_loss[supervised].mean()
    if mode == "entropy":
        # Supervise the gate with the entropy rule. Prompt tokens stay at the last block.
        entropies = torch.stack([_entropy(logits).detach() for logits in exit_logits[:-1]], dim=-1)
        target = (entropies < entropy_tau).to(soft.dtype)
        live = torch.stack(early_probs, dim=-1).clamp(1e-4, 1.0 - 1e-4)
        token_weights = (~is_prompt).unsqueeze(-1).expand_as(live).to(live.dtype)
        bce = F.binary_cross_entropy(live, target, reduction="none")
        loss = ce + (bce * token_weights).sum() / token_weights.sum().clamp(min=1.0)
        mean_energy = 0.0
    else:
        mean_energy_t = energy[:, :-1][supervised].mean()
        loss = ce + lambda_energy * mean_energy_t
        mean_energy = float(mean_energy_t.detach())

    with torch.no_grad():
        chosen = hard.argmax(dim=-1)
        pred = shift_logits.argmax(dim=-1)
        correct = (pred == shift_labels) & supervised
        depth_index = torch.tensor(exits, device=device, dtype=torch.float32)
        mean_block = depth_index[chosen[:, :-1][supervised]].mean()

    stats = {
        "loss": float(loss.detach()),
        "ce": float(ce.detach()),
        "energy": mean_energy,
        "token_acc": float(correct.float().sum() / supervised.float().sum()),
        "mean_exit_block": float(mean_block),
    }
    return loss, stats
