"""Speculative decoding engine for the hardware-aware selective feedback system.

Public API
----------
native_greedy(model, prompt, limit, eos)
    Plain autoregressive greedy decoding. Used as the exact-output reference.

generate(model, prompt, limit, depth, drafts, exits, mode, period, eos,
         adaptive_drafts, log_margins)
    Greedy decoding with optional LayerSkip speculative decoding and observer
    feedback collection.

Both functions return token ID lists (native_greedy) or a structured dict
(generate) that includes cycle-level statistics for energy accounting.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

import torch

from .feedback import LearningState, validate_feedback
from .llama_model_utils import (
    ForwardResult,
    crop_past_key_values,
    forward,
    forward_early,
    forward_remainder,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_cache(cache, expected_layers: int, expected_length: int) -> None:
    """Assert the KV cache has exactly the right shape at a cycle boundary.

    Raises RuntimeError if any layer's key or value tensor has a sequence
    dimension other than `expected_length`, or if the number of layers
    differs from `expected_layers`.
    """
    if len(cache) != expected_layers or any(
        k.shape[2] != expected_length or v.shape[2] != expected_length
        for k, v in cache
    ):
        raise RuntimeError(
            f"Incomplete KV cache: expected {expected_layers} layers × "
            f"{expected_length} positions, got {len(cache)} layers."
        )


@contextmanager
def _observe(
    model: torch.nn.Module,
    depths: list[int],
) -> Generator[dict[int, list], None, None]:
    """Collect per-layer greedy predictions via forward hooks (context manager).

    Registers a hook on each requested layer that projects hidden states to
    token IDs and appends them to a buffer.  The buffer is yielded so that
    the caller can read collected predictions after the forward pass.

    Lower layers fire once per draft step plus once during remainder; upper
    layers fire once per cycle.  The engine concatenates the buffers in
    ``generate`` to restore token-position alignment.

    Note: projections use float32 to match the norm+lm_head precision in
    ``forward_remainder``, keeping observer predictions consistent with the
    main decoding path.
    """
    chunks: dict[int, list] = {d: [] for d in depths}
    handles = []

    def _make_hook(depth: int):
        def _hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            chunks[depth].append(
                model.lm_head(model.model.norm(hidden.float())).argmax(-1)
            )
        return _hook

    try:
        for depth in depths:
            handles.append(
                model.model.layers[depth - 1].register_forward_hook(_make_hook(depth))
            )
        yield chunks
    finally:
        for h in handles:
            h.remove()


def _adaptive_k(
    drafts: int,
    limit: int,
    tokens_so_far: int,
    accept_window: list[float],
) -> int:
    """Choose draft length for this cycle based on recent acceptance rate.

    Returns a value in [1, drafts] (capped by remaining budget):
      - >= 50% recent acceptance → use full draft length
      - < 25%                    → fall back to 1
      - otherwise                → drafts - 1  (conservative middle ground)
    """
    budget = limit - tokens_so_far - 1
    if not accept_window:
        return min(drafts, budget)
    rate = sum(accept_window) / len(accept_window)
    if rate >= 0.5:
        return min(drafts, budget)
    if rate < 0.25:
        return min(1, budget)
    return min(max(1, drafts - 1), budget)


# ---------------------------------------------------------------------------
# Public: native greedy decoding (exact-output reference)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def native_greedy(
    model: torch.nn.Module,
    prompt: list[int],
    limit: int,
    eos: tuple[int, ...] = (),
) -> list[int]:
    """Autoregressively decode up to `limit` tokens using greedy argmax.

    Tokens are kept on-device throughout; the CPU sees exactly one sync
    (the final ``.tolist()``).  EOS checking uses ``torch.isin`` on-device.

    Args:
        model:  Any HuggingFace causal LM.
        prompt: List of integer token IDs (not batched).
        limit:  Maximum number of tokens to generate.
        eos:    Optional tuple of token IDs that stop generation early.

    Returns:
        List of generated token IDs (not including the prompt).
    """
    if limit == 0:
        return []

    ids = torch.tensor([prompt], device=model.device)
    eos_t = torch.tensor(list(eos), dtype=torch.long, device=ids.device) if eos else None
    collected, cache = [], None

    for _ in range(limit):
        out = model(ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        # float() cast ensures argmax matches forward_remainder's FP32 logits,
        # preventing near-tie divergence between reference and speculative paths.
        ids = out.logits[:, -1].float().argmax(-1).reshape(1, 1)
        collected.append(ids)
        if eos_t is not None and torch.isin(ids, eos_t).item():
            break

    return torch.cat(collected, dim=1)[0].tolist()


# ---------------------------------------------------------------------------
# Public: speculative generate
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt: list[int],
    limit: int = 32,
    depth: int = 0,
    drafts: int = 0,
    exits: tuple[int, ...] = (),
    mode: str = "none",
    period: int = 4,
    eos: tuple[int, ...] = (),
    adaptive_drafts: bool = False,
    log_margins: bool = False,
) -> dict:
    """Generate tokens with optional LayerSkip speculative decoding.

    Ordinary decoding (depth=0, drafts=0)
    --------------------------------------
    Equivalent to native_greedy but returns a structured output dict with
    cycle statistics and optional observer feedback.

    Speculative decoding (depth>0, drafts>0)
    ----------------------------------------
    Each cycle:
      1. Run `drafts` cheap forward_early passes (layers 0..depth-1).
      2. Verify all draft tokens in one forward_remainder pass (all layers).
      3. Commit the longest matching prefix plus one correction token.

    Args:
        model:           HuggingFace causal LM.
        prompt:          Integer token IDs.
        limit:           Maximum tokens to generate.
        depth:           Early-exit layer for drafting (0 = ordinary decoding).
        drafts:          Number of draft tokens per cycle (0 = ordinary).
        exits:           Layer depths at which to collect observer predictions.
        mode:            Observer mode: 'none' | 'selected' | 'all' | 'periodic'.
        period:          Observe every `period` cycles (mode='periodic' only).
        eos:             Stop-token IDs.
        adaptive_drafts: Scale draft length per cycle using a rolling acceptance
                         rate window. Reduces wasted verification on hard prompts.
        log_margins:     Store top-1 vs top-2 logit margin per cycle under
                         'top2_margin' for near-tie mismatch diagnosis.

    Returns:
        dict with keys:
          tokens             – list of generated token IDs
          cycles             – per-cycle statistics (draft, target, accepted, …)
          feedback_statistics – aggregated observer match rates by depth
    """
    # ── Validation ────────────────────────────────────────────────────────
    layers = len(model.model.layers)
    if not prompt or limit < 0 or period < 1:
        raise ValueError("Require nonempty prompt, nonnegative limit, and positive period.")
    if mode not in ("none", "selected", "all", "periodic"):
        raise ValueError(f"Unknown observation mode: {mode!r}")
    if (depth == 0) != (drafts == 0) or not 0 <= depth < layers or drafts < 0:
        raise ValueError("Ordinary=(depth=0, drafts=0); speculative depth must be < layers.")
    if len(set(exits)) != len(exits) or any(not 1 <= d < layers for d in exits):
        raise ValueError("exits must be unique layer indices in [1, layers).")

    # ── State ─────────────────────────────────────────────────────────────
    tokens: list[int] = []
    cycles: list[dict] = []
    state = LearningState()

    if not limit:
        return dict(tokens=tokens, cycles=cycles, feedback_statistics={})

    eos_t = torch.tensor(list(eos), dtype=torch.long, device=model.device) if eos else None
    accept_window: list[float] = []  # rolling acceptance rates for adaptive_drafts
    WINDOW = 4

    # ── Prompt prefill ────────────────────────────────────────────────────
    # Run all layers over the full prompt; keep only the last token's logits
    # to save memory on long prompts.
    result = forward(model, torch.tensor([prompt], device=model.device), None,
                     logits_to_keep=1)
    cache = result.past_key_values
    pending = result.logits[:, -1].argmax(-1).reshape(1, 1)
    tokens.append(pending[0, 0].item())
    del result

    # ── Decode loop ───────────────────────────────────────────────────────
    while len(tokens) < limit and tokens[-1] not in eos:
        context = prompt + tokens
        _check_cache(cache, layers, len(context) - 1)

        # Number of draft tokens this cycle.
        k = (
            _adaptive_k(drafts, limit, len(tokens), accept_window)
            if adaptive_drafts and depth
            else min(drafts, limit - len(tokens) - 1)
        )

        # Observer exit depths for this cycle.
        requested = _resolve_exits(exits, k, depth, mode, len(cycles), period)

        # ── Forward passes ────────────────────────────────────────────────
        proposed_tensors: list[torch.Tensor] = []
        with _observe(model, requested) as collected:
            if k == 0:
                # Ordinary step: single full-model pass.
                result = forward(model, pending, cache)
            else:
                # Speculative step: k cheap draft passes + 1 verification pass.
                query, draft_input = None, pending
                for _ in range(k):
                    early = forward_early(model, draft_input, cache, depth, query)
                    cache = early.past_key_values
                    query = early.exit_query_cache
                    draft_input = early.logits[:, -1].argmax(-1).reshape(1, 1)
                    proposed_tensors.append(draft_input)
                    del early
                    # Early EOS: stop drafting, remainder still verifies what we have.
                    if eos_t is not None and torch.isin(draft_input, eos_t).item():
                        break
                block = torch.cat([pending, *proposed_tensors], dim=1)
                result = forward_remainder(model, block, cache, depth, query)

        # ── Cycle output: one GPU sync per cycle, not per token ───────────
        target_ids = result.logits.argmax(-1)
        target: list[int] = target_ids[0].tolist()

        margin: Optional[list[float]] = None
        if log_margins:
            top2 = result.logits[0].float().topk(2, dim=-1).values  # [seq, 2]
            margin = (top2[:, 0] - top2[:, 1]).tolist()

        proposed: list[int] = (
            torch.cat(proposed_tensors, dim=1)[0].tolist() if proposed_tensors else []
        )

        # Count the longest accepted prefix.
        matched = 0
        while matched < len(proposed) and proposed[matched] == target[matched]:
            matched += 1

        # Committed tokens = accepted prefix + first correction.
        committed = proposed[:matched] + [target[matched]]
        # Truncate at EOS if present.
        for i, tok in enumerate(committed):
            if tok in eos:
                committed = committed[: i + 1]
                break

        # ── Feedback and state update ─────────────────────────────────────
        predictions = {
            d: torch.cat(parts, dim=1)[0].tolist() for d, parts in collected.items()
        }
        feedback = validate_feedback(context, proposed, target, predictions,
                                     matched, len(committed))
        state.update(feedback)
        tokens.extend(committed)

        # Crop KV cache to committed sequence length.
        cache = crop_past_key_values(result.past_key_values, len(prompt) + len(tokens) - 1)
        del result
        _check_cache(cache, layers, len(prompt) + len(tokens) - 1)

        # Next pending token = last committed token.
        pending = target_ids[:, len(committed) - 1 : len(committed)]

        # Update acceptance window for adaptive draft length.
        if depth and proposed:
            accept_window.append(matched / len(proposed))
            if len(accept_window) > WINDOW:
                accept_window.pop(0)

        cycles.append({
            "context_ids":   context,
            "draft":         proposed,
            "target":        target,
            "accepted":      matched,
            "committed":     committed,
            "observations":  feedback,
            "top2_margin":   margin,
            "shorter_prefix_acceptance": {
                str(n): min(n, matched) for n in range(1, len(proposed) + 1)
            },
        })

    return dict(tokens=tokens, cycles=cycles, feedback_statistics=state.export())


# ---------------------------------------------------------------------------
# Internal: observer exit resolution
# ---------------------------------------------------------------------------

def _resolve_exits(
    exits: tuple[int, ...],
    k: int,
    depth: int,
    mode: str,
    cycle_index: int,
    period: int,
) -> list[int]:
    """Return the observer exit depths to collect this cycle."""
    if mode == "none":
        return []
    if mode == "periodic" and cycle_index % period:
        return []
    candidates = [d for d in exits if k == 0 or d != depth]
    if mode == "selected":
        return candidates[:1]
    return candidates  # 'all' or 'periodic' on an active cycle
