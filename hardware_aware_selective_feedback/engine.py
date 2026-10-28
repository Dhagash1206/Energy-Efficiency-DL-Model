"""Speculative decoding engine for the hardware-aware selective feedback system.

Public API
----------
reference_greedy(model, prompt, limit, eos)
    Ordinary greedy decoding through the same vendored forward code that
    speculative decoding uses. This is the exact-output reference.

native_greedy(model, prompt, limit, eos)
    Plain Hugging Face greedy decoding. An independent cross-check only: its
    attention kernels round differently in FP16, so it is not the reference.

batched_greedy(model, prompts, limit, eos, pad_id)
    Plain greedy decoding of several prompts in one batch (ordinary baseline).

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
from .vendor.self_speculation.llama_model_utils import (
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

    Projections apply the same norm + lm_head, in the model's dtype, as the
    main decoding path, so an observer prediction at the final depth equals
    the decoded token.
    """
    chunks: dict[int, list] = {d: [] for d in depths}
    handles = []

    def _make_hook(depth: int):
        def _hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            chunks[depth].append(model.lm_head(model.model.norm(hidden)).argmax(-1))
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


ADAPTIVE_WINDOW = 4        # cycles in the rolling acceptance window
SHORTEN_BELOW = 0.5        # recent acceptance below this shortens drafts
FALLBACK_BELOW = 0.3       # recent acceptance below this switches to ordinary decoding
PROBE_EVERY = 4            # ordinary cycles between 1-token re-probes while in fallback


def _adaptive_k(
    drafts: int,
    limit: int,
    tokens_so_far: int,
    accept_window: list[float],
    ordinary_streak: int = 0,
) -> int:
    """Choose draft length for this cycle based on recent acceptance rate.

    Returns a value in [0, drafts] (capped by remaining budget):
      - >= SHORTEN_BELOW acceptance   → full draft length
      - >= FALLBACK_BELOW acceptance  → drafts - 1 (at least 1)
      - <  FALLBACK_BELOW acceptance  → 0 (ordinary step), except every
        PROBE_EVERY ordinary cycles a 1-token probe refreshes the window
        so drafting can resume if acceptance recovers.
    """
    budget = limit - tokens_so_far - 1
    if not accept_window:
        return min(drafts, budget)
    rate = sum(accept_window) / len(accept_window)
    if rate >= SHORTEN_BELOW:
        return min(drafts, budget)
    if rate >= FALLBACK_BELOW:
        return min(max(1, drafts - 1), budget)
    if ordinary_streak >= PROBE_EVERY:
        return min(1, budget)
    return 0


# ---------------------------------------------------------------------------
# Public: native greedy decoding (exact-output reference)
# ---------------------------------------------------------------------------

def reference_greedy(
    model: torch.nn.Module,
    prompt: list[int],
    limit: int,
    eos: tuple[int, ...] = (),
) -> list[int]:
    """Exact-output reference: ordinary decoding via the vendored forward().

    Speculative decoding verifies drafts with the same vendored layers, so a
    mismatch against this reference reflects drafting itself, not a
    different attention kernel or rounding order (which ``native_greedy``
    introduces in FP16).
    """
    return generate(model, prompt, limit, eos=eos)["tokens"]


@torch.inference_mode()
def native_greedy(
    model: torch.nn.Module,
    prompt: list[int],
    limit: int,
    eos: tuple[int, ...] = (),
) -> list[int]:
    """Autoregressively decode up to `limit` tokens using Hugging Face's forward.

    Independent cross-check of ``reference_greedy``; in FP16 the two can
    differ on near-tied logits because the attention kernels differ.

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
        ids = out.logits[:, -1].argmax(-1).reshape(1, 1)
        collected.append(ids)
        if eos_t is not None and torch.isin(ids, eos_t).item():
            break

    return torch.cat(collected, dim=1)[0].tolist()


@torch.inference_mode()
def batched_greedy(
    model: torch.nn.Module,
    prompts: list[list[int]],
    limit: int,
    eos: tuple[int, ...] = (),
    pad_id: int = 0,
) -> list[list[int]]:
    """Greedy-decode several prompts together in one batch (ordinary decoding).

    Prompts are left-padded so every row's next token sits in the last column;
    the attention mask hides the padding and position IDs restart at each
    row's first real token, so each row should reproduce ``native_greedy``.
    Batch shapes can change floating-point rounding, so callers should still
    compare against ``native_greedy`` for exact-output claims.

    Rows that hit EOS stop contributing tokens; the batch stops when every row
    has finished or ``limit`` tokens were generated.

    Returns:
        One list of generated token IDs per prompt (EOS included when hit).
    """
    if not prompts or any(not p for p in prompts) or limit < 0:
        raise ValueError("Require nonempty prompts and a nonnegative limit.")
    if limit == 0:
        return [[] for _ in prompts]

    device = model.device
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    eos_t = torch.tensor(list(eos), dtype=torch.long, device=device) if eos else None
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    collected, cache = [], None

    for _ in range(limit):
        out = model(ids, attention_mask=mask, position_ids=positions,
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        ids = out.logits[:, -1].argmax(-1, keepdim=True)
        collected.append(ids)
        if eos_t is not None:
            finished |= torch.isin(ids[:, 0], eos_t)
            if finished.all().item():
                break
        mask = torch.cat([mask, mask.new_ones(len(prompts), 1)], dim=1)
        positions = positions[:, -1:] + 1

    rows = torch.cat(collected, dim=1).tolist()
    if eos:
        rows = [row[: next((i + 1 for i, tok in enumerate(row) if tok in eos), len(row))]
                for row in rows]
    return rows


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
    graph_drafter=None,
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
                         rate window, falling back to ordinary steps when
                         acceptance is very low. Reduces wasted draft passes.
        log_margins:     Store top-1 vs top-2 logit margin per cycle under
                         'top2_margin' for near-tie mismatch diagnosis.
        graph_drafter:   Optional GraphDrafter (graph_drafter.py) for this
                         depth; replaces the forward_early loop with a
                         static-cache step (CUDA Graph on CUDA). Requires
                         mode='none' because observer hooks do not fire.

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
    if graph_drafter is not None:
        if graph_drafter.depth != depth:
            raise ValueError("graph_drafter depth must equal the draft depth.")
        if mode != "none":
            raise ValueError("graph_drafter requires mode='none' (observer hooks do not fire).")
        graph_drafter.reset()

    # ── State ─────────────────────────────────────────────────────────────
    tokens: list[int] = []
    cycles: list[dict] = []
    state = LearningState()

    if not limit:
        return dict(tokens=tokens, cycles=cycles, feedback_statistics={})

    eos_t = torch.tensor(list(eos), dtype=torch.long, device=model.device) if eos else None
    accept_window: list[float] = []  # rolling acceptance rates for adaptive_drafts
    ordinary_streak = 0              # consecutive adaptive fallback cycles

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
            _adaptive_k(drafts, limit, len(tokens), accept_window, ordinary_streak)
            if adaptive_drafts and depth
            else min(drafts, limit - len(tokens) - 1)
        )
        ordinary_streak = ordinary_streak + 1 if (adaptive_drafts and depth and k == 0) else 0

        # Observer exit depths for this cycle.
        requested = _resolve_exits(exits, k, depth, mode, len(cycles), period)

        # ── Forward passes ────────────────────────────────────────────────
        proposed_tensors: list[torch.Tensor] = []
        with _observe(model, requested) as collected:
            if k == 0:
                # Ordinary step: single full-model pass.
                result = forward(model, pending, cache)
            elif graph_drafter is not None:
                proposed_tensors, cache, query = graph_drafter.draft(cache, pending, k, eos_t)
                block = torch.cat([pending, *proposed_tensors], dim=1)
                result = forward_remainder(model, block, cache, depth, query)
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
        if graph_drafter is not None and k:
            graph_drafter.commit(len(prompt) + len(tokens) - 1)

        # Next pending token = last committed token.
        pending = target_ids[:, len(committed) - 1 : len(committed)]

        # Update acceptance window for adaptive draft length.
        if depth and proposed:
            accept_window.append(matched / len(proposed))
            if len(accept_window) > ADAPTIVE_WINDOW:
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
