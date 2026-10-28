"""Run the first N transformer blocks and the shared vocabulary head.

Keys and values are written only for the blocks that run. This is the
compute the energy probe measures at a forced exit.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def forward_depth(model, input_ids: torch.Tensor, depth: int, past_key_values=None):
    """Return logits and the cache after `depth` blocks (1-based count)."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    base = model.model
    n_layers = len(base.layers)
    if depth > n_layers:
        raise ValueError(f"depth {depth} exceeds {n_layers} blocks")

    inputs_embeds = base.embed_tokens(input_ids)
    if past_key_values is None:
        past_key_values = DynamicCache()
    past_seen = past_key_values.get_seq_length()
    cache_position = torch.arange(
        past_seen,
        past_seen + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )
    position_ids = cache_position.unsqueeze(0)
    causal_mask = _causal_mask(base, inputs_embeds, cache_position, past_key_values)
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)

    hidden = inputs_embeds
    for layer in base.layers[:depth]:
        out = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden = out[0] if isinstance(out, tuple) else out

    # Every exit uses the final norm and the one shared vocabulary head.
    hidden = base.norm(hidden)
    logits = model.lm_head(hidden)
    return logits, past_key_values


def _causal_mask(base, inputs_embeds, cache_position, past_key_values):
    if not hasattr(base, "_update_causal_mask"):
        return None
    try:
        return base._update_causal_mask(
            None, inputs_embeds, cache_position, past_key_values, False
        )
    except TypeError:
        return None
