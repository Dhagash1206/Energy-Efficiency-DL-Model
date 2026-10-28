"""Generate tokens, physically skipping blocks after the chosen exit.

Prompt tokens are encoded at every depth so the question remains visible.
Each generated token runs only the blocks up to its exit and writes keys
only for those blocks. A later token that goes deeper attends only to
earlier tokens that reached that depth.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from joulerexit.residual import EnergyProfile
from joulerexit.skip_forward import _costs, _entropy, _gate_feature


def _layer_forward(layer, hidden, position_ids, position_embeddings, cache, attention_mask):
    updated = layer(
        hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        position_embeddings=position_embeddings,
    )
    return updated[0] if isinstance(updated, tuple) else updated


def _logits(model, hidden: torch.Tensor) -> torch.Tensor:
    return model.lm_head(model.model.norm(hidden))


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor, exits: tuple[int, ...]):
    """Run the prompt through every block and keep the hidden state at each exit."""
    base = model.model
    device = input_ids.device
    cache = DynamicCache()
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    hidden = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(hidden, position_ids=position_ids)
    time = input_ids.shape[1]
    causal = torch.tril(torch.ones(time, time, device=device, dtype=torch.bool))
    mask = torch.zeros(1, 1, time, time, device=device, dtype=hidden.dtype)
    mask = mask.masked_fill(~causal.view(1, 1, time, time), torch.finfo(hidden.dtype).min)
    snapshots = []
    previous = 0
    for depth in exits:
        for layer in base.layers[previous:depth]:
            hidden = _layer_forward(layer, hidden, position_ids, position_embeddings, cache, mask)
        snapshots.append(hidden[:, -1, :])
        previous = depth
    return cache, snapshots


def _halt(
    mode: str,
    index: int,
    exits: tuple[int, ...],
    hidden: torch.Tensor,
    logits: torch.Tensor,
    gates,
    profile: EnergyProfile,
    seq_len: int,
    tau: float,
    controller,
) -> bool:
    if index == len(exits) - 1:
        return True
    if mode == "full":
        return False
    if mode == "fixed":
        return index == 1
    entropy = _entropy(logits)
    depth = exits[index]
    length = torch.tensor([seq_len], device=hidden.device)
    decode = torch.ones(1, device=hidden.device)
    predicted, residual, _depth_only = _costs(profile, depth, length, decode)
    if mode == "entropy" and gates is None:
        return bool(entropy.item() < tau)
    if mode == "controller":
        return controller.halt(float(entropy.item()), float(residual.item()), depth)
    feature = _gate_feature(mode, residual, predicted)
    probability = gates.probability(index, hidden.float(), entropy.float(), feature.float())
    return bool(probability.item() > 0.5)


def choose_token(model, mode, exits, snapshots, gates, profile, seq_len, tau, controller):
    for index, hidden in enumerate(snapshots):
        logits = _logits(model, hidden.unsqueeze(1))[:, -1, :]
        if _halt(mode, index, exits, hidden, logits, gates, profile, seq_len, tau, controller):
            token = int(logits.argmax(dim=-1).item())
            return token, index
    raise RuntimeError("the last exit did not halt")


@torch.inference_mode()
def generate(
    model,
    gates,
    profile: EnergyProfile,
    input_ids: torch.Tensor,
    exits: tuple[int, ...],
    mode: str,
    max_new_tokens: int,
    tau: float = 1.5,
    controller=None,
    eos_token_id: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return generated token ids and the exit index of each of them.

    The first token is read from the prompt's exit snapshots. Later tokens
    run only the blocks up to the exit that accepts them.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("generation uses batch size 1")
    cache, snapshots = prefill(model, input_ids, exits)
    prompt_len = input_ids.shape[1]
    token, exit_index = choose_token(
        model, mode, exits, snapshots, gates, profile, prompt_len, tau, controller
    )
    tokens = [token]
    exits_used = [exit_index]
    base = model.model

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and token == eos_token_id:
            break
        position = prompt_len + len(tokens) - 1
        position_ids = torch.tensor([[position]], device=input_ids.device)
        hidden = base.embed_tokens(torch.tensor([[token]], device=input_ids.device))
        position_embeddings = base.rotary_emb(hidden, position_ids=position_ids)
        previous = 0
        chosen = None
        for index, depth in enumerate(exits):
            for layer in base.layers[previous:depth]:
                hidden = _layer_forward(layer, hidden, position_ids, position_embeddings, cache, None)
            logits = _logits(model, hidden)[:, -1, :]
            if _halt(mode, index, exits, hidden[:, -1, :], logits, gates, profile, position + 1, tau, controller):
                token = int(logits.argmax(dim=-1).item())
                chosen = index
                break
            previous = depth
        if chosen is None:
            raise RuntimeError("decode did not exit")
        tokens.append(token)
        exits_used.append(chosen)
        if eos_token_id is not None and token == eos_token_id:
            break
    return tokens, exits_used


@torch.inference_mode()
def all_exit_logits(model, input_ids: torch.Tensor, exits: tuple[int, ...]) -> list[torch.Tensor]:
    """Logits at every exit for a full sequence. Every token reaches every block."""
    base = model.model
    device = input_ids.device
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    hidden = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(hidden, position_ids=position_ids)
    time = input_ids.shape[1]
    causal = torch.tril(torch.ones(time, time, device=device, dtype=torch.bool))
    mask = torch.zeros(1, 1, time, time, device=device, dtype=hidden.dtype)
    mask = mask.masked_fill(~causal.view(1, 1, time, time), torch.finfo(hidden.dtype).min)
    logits = []
    previous = 0
    for depth in exits:
        for layer in base.layers[previous:depth]:
            updated = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden = updated[0] if isinstance(updated, tuple) else updated
        logits.append(_logits(model, hidden))
        previous = depth
    return logits

