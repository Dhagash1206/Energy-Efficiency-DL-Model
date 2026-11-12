"""Committed-output stopping contract and boundary resolution for all decoders.

Phase P2: Ensures ordinary, speculative, and DEL decoders evaluate stop conditions
only on committed output, properly handle multi-token stop sequences crossing
cycles, crop cache and tokens consistently on mid-block stops, and classify
repetition loops without silently extending token limits.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StopCriteria:
    """Unified stop configuration across tasks and decoding policies."""

    eos_token_ids: Tuple[int, ...] = (128001, 128009)
    stop_sequences: Tuple[Tuple[int, ...], ...] = ()
    stop_strings: Tuple[str, ...] = ("\nQuestion:", "\n---", "```\n")
    max_tokens: int = 256
    repetition_ngram_size: int = 16
    repetition_threshold: int = 3  # consecutive repetitions of the n-gram


@dataclass
class StopDecision:
    stopped: bool
    finish_reason: Optional[str] = (
        None  # "eos", "stop_sequence", "stop_string", "length", "repetition"
    )
    matched_boundary: Optional[Any] = None
    crop_index: Optional[int] = None  # Token index at which to crop relative to evaluated sequence
    is_repetitive: bool = False
    repetition_pattern: Optional[List[int]] = None


def detect_repetition(
    tokens: Sequence[int], ngram_size: int = 16, threshold: int = 3
) -> Tuple[bool, Optional[List[int]]]:
    """Detect if a sequence of generated tokens has entered a repetitive loop."""
    if len(tokens) < ngram_size * threshold:
        return False, None

    recent = list(tokens[-ngram_size * threshold :])
    unit = recent[-ngram_size:]
    repeats = 0
    for i in range(len(recent) - ngram_size, -1, -ngram_size):
        if recent[i : i + ngram_size] == unit:
            repeats += 1
        else:
            break
    if repeats >= threshold:
        return True, unit

    return False, None


def evaluate_committed_stopping(
    committed_tokens: Sequence[int],
    criteria: StopCriteria,
    tokenizer: Optional[Any] = None,
    is_committed: bool = True,
) -> StopDecision:
    """Evaluate stopping conditions strictly on committed output tokens.

    Finds the earliest matching stop boundary across all conditions and respects max_tokens.
    A candidate/rejected draft (is_committed=False) MUST NEVER trigger stopping.
    """
    if not is_committed or not committed_tokens:
        return StopDecision(stopped=False)

    candidates: List[Tuple[int, int, str, Any]] = []
    # Candidate tuple: (crop_index, priority, finish_reason, matched_boundary)
    # Lower priority number wins ties (0: stop_sequence/eos/stop_string, 1: length, 2: repetition)

    # 1. EOS token candidates
    for idx, token_id in enumerate(committed_tokens):
        if token_id in criteria.eos_token_ids:
            candidates.append((idx + 1, 0, "eos", token_id))
            break  # first EOS is the earliest EOS

    # 2. Token-level stop sequences: find the earliest match across ALL sequences
    for seq in criteria.stop_sequences:
        if not seq:
            continue
        seq_len = len(seq)
        for i in range(len(committed_tokens) - seq_len + 1):
            if tuple(committed_tokens[i : i + seq_len]) == seq:
                candidates.append((i, 0, "stop_sequence", seq))
                break  # first match for this seq

    # 3. String-level stop sequences (if tokenizer available)
    if tokenizer is not None and criteria.stop_strings:
        try:
            full_text = tokenizer.decode(committed_tokens, skip_special_tokens=False)
            for s in criteria.stop_strings:
                pos = full_text.find(s)
                if pos != -1:
                    visible_text = full_text[:pos]
                    re_encoded = tokenizer.encode(visible_text, add_special_tokens=False)
                    crop_idx = min(len(re_encoded), len(committed_tokens))
                    candidates.append((crop_idx, 0, "stop_string", s))
        except Exception:
            pass

    # 4. Length limit candidate
    if len(committed_tokens) >= criteria.max_tokens:
        candidates.append((criteria.max_tokens, 1, "length", None))

    # 5. Repetition candidate
    is_rep, rep_unit = detect_repetition(
        committed_tokens,
        ngram_size=criteria.repetition_ngram_size,
        threshold=criteria.repetition_threshold,
    )
    if is_rep:
        candidates.append((len(committed_tokens), 2, "repetition", rep_unit))

    if not candidates:
        return StopDecision(stopped=False)

    # Sort candidates by (crop_index, priority) so the EARLIEST boundary wins
    candidates.sort(key=lambda c: (c[0], c[1]))
    best_crop, _, reason, boundary = candidates[0]

    # If the earliest stopping boundary happens after max_tokens, cap at max_tokens
    if best_crop > criteria.max_tokens:
        return StopDecision(
            stopped=True,
            finish_reason="length",
            matched_boundary=None,
            crop_index=criteria.max_tokens,
            is_repetitive=is_rep,
            repetition_pattern=rep_unit,
        )

    return StopDecision(
        stopped=True,
        finish_reason=reason,
        matched_boundary=boundary,
        crop_index=best_crop,
        is_repetitive=is_rep,
        repetition_pattern=rep_unit,
    )


def apply_stop_decision_to_block(
    prior_tokens: Sequence[int],
    newly_committed_block: Sequence[int],
    criteria: StopCriteria,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[int], StopDecision]:
    """Check stopping for a newly committed speculative block evaluated in the full context

    of all tokens generated so far, correctly detecting stop sequences spanning block boundaries
    and cropping the newly committed block mid-cycle.
    """
    all_tokens = list(prior_tokens) + list(newly_committed_block)
    decision = evaluate_committed_stopping(
        all_tokens,
        criteria,
        tokenizer=tokenizer,
        is_committed=True,
    )

    if not decision.stopped:
        return list(newly_committed_block), decision

    # Stop occurred; determine how much of newly_committed_block to keep
    crop_idx = decision.crop_index if decision.crop_index is not None else len(all_tokens)
    prior_len = len(prior_tokens)

    if crop_idx <= prior_len:
        # Stop occurred at or before the boundary of the new block
        return [], decision
    else:
        kept_count = crop_idx - prior_len
        cropped_block = list(newly_committed_block)[:kept_count]
        return cropped_block, decision


def crop_kv_cache(past_key_values: Any, maximum_length: int) -> Any:
    """Trim key and value tensors in the KV cache to maximum_length positions.

    Supports legacy tuple/list caches and HuggingFace DynamicCache.
    """
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(maximum_length)
        return past_key_values
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :maximum_length, :]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][
                :, :, :maximum_length, :
            ]
        return past_key_values
    cropped = []
    for layer in past_key_values:
        if layer is None or len(layer) != 2:
            continue
        k, v = layer
        if k is None or v is None:
            continue
        cropped.append((k[:, :, :maximum_length, :], v[:, :, :maximum_length, :]))
    return tuple(cropped)
