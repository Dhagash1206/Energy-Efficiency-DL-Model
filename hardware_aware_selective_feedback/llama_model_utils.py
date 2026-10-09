# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Low-level Transformer forward utilities for the LayerSkip speculative engine.

Three public entry points
-------------------------
forward()           -- Full model pass (all layers). Used by native_greedy and
                       generate() for ordinary decoding.
forward_early()     -- Shallow pass (layers 0..exit_layer-1). Produces a cheap
                       draft token and accumulates an exit-query cache.
forward_remainder() -- Deep pass (layers exit_layer..N-1) over a block of
                       [pending, *draft] tokens. Verifies and extends the draft.

Both forward_early and forward_remainder use the legacy tuple KV cache so that
crop_past_key_values can trim it in O(1) with a slice. forward() wraps
DynamicCache internally to stay consistent with the HuggingFace model contract.

NOTE: forward_early / forward_remainder should be migrated to DynamicCache
once the crop-in-place API stabilises in transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import transformers


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ForwardResult:
    """Output of every forward variant."""
    logits: torch.Tensor
    past_key_values: Optional[KVCache]
    # Accumulated pre-exit hidden states used to reconstruct the full sequence
    # context when forward_remainder transitions from early to full layers.
    exit_query_cache: Optional[torch.Tensor] = field(default=None)


# ---------------------------------------------------------------------------
# Attention mask helpers (vendored from transformers BART; unchanged)
# ---------------------------------------------------------------------------

def _make_causal_mask(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_length: int = 0,
) -> torch.Tensor:
    """Lower-triangular causal mask of shape [bsz, 1, tgt, tgt + past]."""
    bsz, tgt = shape
    mask = torch.full((tgt, tgt), torch.finfo(dtype).min, device=device)
    cond = torch.arange(tgt, device=device)
    mask.masked_fill_(cond < (cond + 1).view(tgt, 1), 0)
    mask = mask.to(dtype)
    if past_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt, past_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt, tgt + past_length)


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
) -> torch.Tensor:
    """Expand [bsz, src] padding mask to [bsz, 1, tgt, src] additive mask."""
    bsz, src = mask.size()
    tgt = tgt_len or src
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt, src).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)


def _causal_mask(
    model: transformers.LlamaForCausalLM,
    batch_size: int,
    seq_len: int,
    past_length: int,
    inputs_embeds: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build the combined causal + padding mask for a decoder block."""
    mask = None
    if seq_len > 1:
        mask = _make_causal_mask(
            (batch_size, seq_len),
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_length=past_length,
        )
    return mask


# ---------------------------------------------------------------------------
# Decoder layer runner
# ---------------------------------------------------------------------------

def _run_decoder(
    model: transformers.LlamaForCausalLM,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, transformers.cache_utils.DynamicCache]:
    """Run one transformer decoder layer, handling the transformers ≥4.50 API.

    transformers 4.50 requires explicit rotary embeddings and cache_position;
    older versions accept neither. We inject both unconditionally and strip the
    deprecated padding_mask kwarg.
    """
    cache = kwargs["past_key_value"]
    kwargs.pop("padding_mask", None)
    positions = kwargs["position_ids"]
    if "position_embeddings" not in kwargs:
        kwargs["position_embeddings"] = model.model.rotary_emb(hidden_states, positions)
    kwargs["cache_position"] = positions[0]
    outputs = layer(hidden_states, **kwargs)
    return outputs[0], cache


# ---------------------------------------------------------------------------
# KV-cache utilities
# ---------------------------------------------------------------------------

def crop_past_key_values(
    past_key_values: KVCache,
    maximum_length: int,
) -> KVCache:
    """Slice every layer's K/V tensors to `maximum_length` sequence positions.

    Called after each speculative cycle to discard rejected draft tokens from
    the cache before the next draft step begins.
    """
    cropped: KVCache = []
    for k, v in past_key_values:
        if k is None or k == []:
            break
        cropped.append((k[:, :, :maximum_length, :], v[:, :, :maximum_length, :]))
    return tuple(cropped)


# ---------------------------------------------------------------------------
# Forward: full model (all layers)
# ---------------------------------------------------------------------------

def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[KVCache],
    logits_to_keep: int = 0,
) -> ForwardResult:
    """Run all N transformer layers and return logits + updated KV cache.

    Args:
        input_ids:      [batch, seq] token IDs.
        past_key_values: Legacy tuple KV cache from a previous call, or None
                         for the first call (prompt prefill).
        logits_to_keep: If > 0, only project the last `logits_to_keep` hidden
                        states to vocabulary space (saves memory on long prompts).

    Returns:
        ForwardResult with logits [batch, seq_or_1, vocab] and updated cache.
    """
    if logits_to_keep < 0:
        raise ValueError("logits_to_keep must be nonnegative")

    device = input_ids.device
    batch, seq = input_ids.shape
    past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
    position_ids = torch.arange(past_len, seq + past_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    hidden = model.model.embed_tokens(input_ids)
    attn_mask = _causal_mask(model, batch, seq, past_len, hidden)
    pos_emb = model.model.rotary_emb(hidden, position_ids)

    for layer in model.model.layers:
        hidden, cache = _run_decoder(
            model, layer, hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    cache = cache.to_legacy_cache()
    if logits_to_keep:
        hidden = hidden[:, -logits_to_keep:]
    logits = model.lm_head(model.model.norm(hidden))
    return ForwardResult(logits=logits, past_key_values=cache)


# ---------------------------------------------------------------------------
# Forward: early layers only (draft step)
# ---------------------------------------------------------------------------

def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[KVCache],
    exit_layer: int,
    exit_query_cache: Optional[torch.Tensor],
) -> ForwardResult:
    """Run layers 0..exit_layer-1 to produce a cheap draft token.

    The hidden states at the exit boundary are accumulated in
    `exit_query_cache`, which forward_remainder uses to reconstruct the full
    sequence context for the upper layers.

    Args:
        input_ids:        [batch, 1] token ID of the pending token.
        past_key_values:  KV cache covering layers 0..exit_layer-1.
        exit_layer:       Number of layers to run (depth of the early exit).
        exit_query_cache: Accumulated pre-exit hidden states from previous
                          draft steps, or None on the first call.

    Returns:
        ForwardResult with draft logits and updated exit_query_cache.
    """
    device = input_ids.device
    batch, seq = input_ids.shape
    past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
    position_ids = torch.arange(past_len, seq + past_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)

    hidden = model.model.embed_tokens(input_ids)
    attn_mask = _causal_mask(model, batch, seq, past_len, hidden)
    pos_emb = model.model.rotary_emb(hidden, position_ids)

    for layer in model.model.layers[:exit_layer]:
        hidden, cache = _run_decoder(
            model, layer, hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
            past_key_value=cache,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    cache = cache.to_legacy_cache()

    # Accumulate pre-exit hidden states for the remainder block.
    exit_query_cache = (
        hidden if exit_query_cache is None
        else torch.cat([exit_query_cache, hidden], dim=1)
    )

    logits = model.lm_head(model.model.norm(hidden))
    return ForwardResult(logits=logits, past_key_values=cache,
                         exit_query_cache=exit_query_cache)


# ---------------------------------------------------------------------------
# Forward: remainder layers (verification step)
# ---------------------------------------------------------------------------

def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[KVCache],
    exit_layer: int,
    exit_query_cache: Optional[torch.Tensor],
) -> ForwardResult:
    """Run layers exit_layer..N-1 over [pending, *draft] tokens to verify drafts.

    The block fed to this function has shape [batch, 1 + num_drafts]:
      - Position 0  : the pending token (last committed token).
      - Positions 1+: the draft tokens proposed by forward_early.

    Position IDs must be anchored at the correct absolute sequence positions
    regardless of how many full-model passes have already occurred.

    Early layers (< exit_layer) process only the pending token using the
    draft KV cache.  Upper layers (>= exit_layer) process the full block
    [exit_query_cache | pending_hidden, *draft_hidden] using the verified KV
    cache (empty on the first call).

    Args:
        input_ids:        [batch, 1 + drafts] pending + draft token IDs.
        past_key_values:  Mixed KV cache: layers 0..exit_layer-1 are populated
                          from draft steps; layers exit_layer..N-1 may be
                          empty if no full pass has happened yet.
        exit_layer:       Boundary between draft and full layers.
        exit_query_cache: Pre-exit hidden states accumulated by forward_early.

    Returns:
        ForwardResult with verification logits [batch, 1 + drafts, vocab].
    """
    device = input_ids.device
    batch, seq = input_ids.shape        # seq = 1 + num_drafts
    num_pending = 1                     # we always verify exactly 1 pending token first

    # ── Compute sequence offsets ──────────────────────────────────────────
    # draft_past_len: how many tokens layers 0..exit_layer-1 have already seen.
    # full_past_len : how many tokens layers exit_layer..N-1 have seen. This is
    #                 0 on the first verification call and grows as cycles accumulate.
    draft_past_len = 0
    full_past_len  = 0
    if past_key_values is not None and past_key_values[0] is not None:
        draft_past_len = past_key_values[0][0].shape[2]
        if len(past_key_values) == len(model.model.layers):
            # Upper layers have a populated cache — use its length directly.
            full_past_len = past_key_values[-1][0].shape[2]
        else:
            # Upper layers have no cache yet.  Position IDs must still point
            # to the correct absolute positions: the pending token sits at
            # draft_past_len - seq + num_pending, and the draft tokens follow.
            full_past_len = draft_past_len - seq + num_pending

    total_seq = num_pending + draft_past_len   # full sequence length after this step

    cache = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)
    hidden = model.model.embed_tokens(input_ids)

    # Position IDs span [full_past_len, full_past_len + seq).
    position_ids = torch.arange(
        full_past_len, full_past_len + seq, dtype=torch.long, device=device,
    ).unsqueeze(0)
    pos_emb = model.model.rotary_emb(hidden, position_ids)

    # Early layers see only the pending token (1 position) against the draft cache.
    early_pos_ids = position_ids[:, -num_pending:]
    early_pos_emb = tuple(p[:, -num_pending:] for p in pos_emb)
    early_mask    = _causal_mask(model, batch, num_pending, draft_past_len, hidden)

    # Upper layers see the full block against the verified cache.
    full_mask = _causal_mask(model, batch, seq, full_past_len, hidden)

    full_hidden: Optional[torch.Tensor] = None
    for idx, layer in enumerate(model.model.layers):
        if idx < exit_layer:
            # Early layers: process only the pending token.
            pending_hidden = hidden[:, -num_pending:]
            hidden, cache = _run_decoder(
                model, layer, pending_hidden,
                attention_mask=early_mask,
                position_ids=early_pos_ids,
                position_embeddings=early_pos_emb,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
        else:
            # Upper layers: reconstruct the full block on first entry.
            if full_hidden is None:
                if exit_query_cache is not None:
                    # Prepend cached pre-exit states from all previous draft steps.
                    full_hidden = torch.cat(
                        [exit_query_cache, hidden[:, -num_pending:]], dim=1
                    )
                else:
                    full_hidden = hidden
            hidden, cache = _run_decoder(
                model, layer, full_hidden,
                attention_mask=full_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
            full_hidden = hidden  # subsequent upper layers reuse output directly

    cache = cache.to_legacy_cache()
    logits = model.lm_head(model.model.norm(hidden))
    return ForwardResult(logits=logits, past_key_values=cache,
                         exit_query_cache=exit_query_cache)
