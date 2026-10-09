"""Cache-enabled split execution using the locally vendored LayerSkip helpers."""
from contextlib import contextmanager
import torch
from .llama_model_utils import (
    forward, forward_early, forward_remainder, crop_past_key_values,
)
from .feedback import validate_feedback, LearningState


def check_cache(cache, layers, length):
    if len(cache) != layers or any(k.shape[2] != length or v.shape[2] != length for k, v in cache):
        raise RuntimeError('Incomplete cache at configuration boundary')


@contextmanager
def observe(model, depths):
    """Project immediately; retain token IDs rather than vocabulary-sized logits.

    Lower layers run once per draft step plus once in remainder; upper layers
    run once over the entire block. Concatenation restores identical alignment.
    """
    chunks = {d: [] for d in depths}
    handles = []
    def hook(depth):
        def capture(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # FP32 projection matches forward() / forward_remainder() path.
            chunks[depth].append(model.lm_head(model.model.norm(hidden.float())).argmax(-1))
        return capture
    try:
        for depth in depths:
            handles.append(model.model.layers[depth-1].register_forward_hook(hook(depth)))
        yield chunks
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def native_greedy(model, prompt, limit, eos=()):
    ids = torch.tensor([prompt], device=model.device)
    if limit == 0:
        return []
    # Keep an on-device EOS set so we never sync to CPU inside the loop.
    eos_t = torch.tensor(list(eos), dtype=torch.long, device=ids.device) if eos else None
    collected, cache = [], None
    for _ in range(limit):
        result = model(ids, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        # Cast to float32: matches the FP32 norm+lm_head in our forward().
        # Without this, near-tie tokens diverge because HuggingFace model()
        # returns FP16 logits while forward_remainder() now returns FP32.
        ids = result.logits[:, -1].float().argmax(-1).reshape(1, 1)
        collected.append(ids)
        if eos_t is not None and torch.isin(ids, eos_t).item():
            break
    # Single host sync at the end of the whole sequence.
    return torch.cat(collected, dim=1)[0].tolist()


@torch.inference_mode()
def generate(model, prompt, limit=32, depth=0, drafts=0, exits=(), mode='none', period=4, eos=(),
             adaptive_drafts=False, log_margins=False):
    """Generate tokens with optional speculative decoding.

    adaptive_drafts: scale draft length k per cycle based on rolling
    acceptance rate (last 4 cycles). Reduces verification waste when the
    early exit rejects frequently.

    log_margins: record the top-1 vs top-2 logit margin at each cycle in
    the cycle dict under 'top2_margin'. Useful for diagnosing near-tie
    mismatch prompts without a separate diagnostic run.
    """
    layers = len(model.model.layers)
    if not prompt or limit < 0 or period < 1:
        raise ValueError('Require nonempty prompt, nonnegative limit and positive period')
    if mode not in ('none', 'selected', 'all', 'periodic'):
        raise ValueError('Unknown observation mode')
    if (depth == 0) != (drafts == 0) or not 0 <= depth < layers or drafts < 0:
        raise ValueError('Ordinary=(0,0); speculative depth must precede final layer')
    if len(set(exits)) != len(exits) or any(not 1 <= d < layers for d in exits):
        raise ValueError('Exits must be unique intermediate depths')
    tokens, cycles, state = [], [], LearningState()
    if not limit:
        return dict(tokens=tokens, cycles=cycles, feedback_statistics={})
    # On-device EOS tensor avoids a host sync on every EOS check.
    eos_t = torch.tensor(list(eos), dtype=torch.long,
                         device=model.device) if eos else None
    # Rolling acceptance window for adaptive draft length (last 4 cycles).
    _accept_window: list[float] = []
    _WINDOW = 4
    # Prompt positions populate the KV cache, but only the final position's
    # vocabulary scores are needed to choose the first generated token.
    result = forward(model, torch.tensor([prompt], device=model.device), None,
                     logits_to_keep=1)
    cache = result.past_key_values
    pending = result.logits[:, -1].argmax(-1).reshape(1, 1)
    # Keep first token on GPU; sync deferred to end of cycle.
    tokens.append(pending[0, 0].item())
    del result
    while len(tokens) < limit and tokens[-1] not in eos:
        context = prompt + tokens
        check_cache(cache, layers, len(context)-1)
        # Adaptive draft length: scale k based on recent acceptance rate.
        if adaptive_drafts and depth and _accept_window:
            rate = sum(_accept_window) / len(_accept_window)
            if rate >= 0.5:
                k = min(drafts, limit - len(tokens) - 1)
            elif rate < 0.25:
                k = min(1, limit - len(tokens) - 1)
            else:
                k = min(max(1, drafts - 1), limit - len(tokens) - 1)
        else:
            k = min(drafts, limit - len(tokens) - 1)
        requested = [d for d in exits if k == 0 or d != depth]
        if mode == 'none' or (mode == 'periodic' and len(cycles) % period):
            requested = []
        elif mode == 'selected':
            requested = requested[:1]
        proposed_tensors = []  # GPU tensors — no .item() inside the draft loop
        with observe(model, requested) as collected:
            if k == 0:
                result = forward(model, pending, cache)
            else:
                query, draft_input = None, pending
                for _ in range(k):
                    early = forward_early(model, draft_input, cache, depth, query)
                    cache, query = early.past_key_values, early.exit_query_cache
                    draft_input = early.logits[:, -1].argmax(-1).reshape(1, 1)
                    proposed_tensors.append(draft_input)
                    # Do not retain obsolete logits/cache containers while the
                    # next draft or verifier allocates its working tensors.
                    del early
                    # EOS check stays on-device; single .item() only when hit.
                    if eos_t is not None and torch.isin(draft_input, eos_t).item():
                        break
                block = torch.cat([pending, *proposed_tensors], dim=1)
                result = forward_remainder(model, block, cache, depth, query)
        target_ids = result.logits.argmax(-1)
        # One .tolist() sync per cycle (not per token).
        target = target_ids[0].tolist()
        # Top-2 logit margin: measures how close the argmax decision was.
        # Only computed when log_margins=True to avoid the extra sort cost.
        margin = None
        if log_margins:
            logits_f = result.logits[0].float()  # [seq, vocab]
            top2 = logits_f.topk(2, dim=-1).values  # [seq, 2]
            # Report the margin at the last verified position (the target token).
            margin = (top2[:, 0] - top2[:, 1]).tolist()
        proposed = (torch.cat(proposed_tensors, dim=1)[0].tolist()
                    if proposed_tensors else [])
        matched = 0
        while matched < len(proposed) and proposed[matched] == target[matched]:
            matched += 1
        committed = proposed[:matched] + [target[matched]]
        for i, token in enumerate(committed):
            if token in eos:
                committed = committed[:i+1]
                break
        # Feedback predictions: defer tolist to feedback module boundary.
        predictions = {d: torch.cat(parts, dim=1)[0].tolist() for d, parts in collected.items()}
        feedback = validate_feedback(context, proposed, target, predictions, matched, len(committed))
        state.update(feedback)
        tokens.extend(committed)
        cache = crop_past_key_values(result.past_key_values, len(prompt)+len(tokens)-1)
        del result
        check_cache(cache, layers, len(prompt)+len(tokens)-1)
        pending = target_ids[:, len(committed)-1:len(committed)]
        # Update rolling acceptance window for adaptive draft length.
        if depth and proposed:
            _accept_window.append(matched / len(proposed))
            if len(_accept_window) > _WINDOW:
                _accept_window.pop(0)
        cycles.append(dict(context_ids=context, draft=proposed, target=target,
                           accepted=matched, committed=committed, observations=feedback,
                           top2_margin=margin,
                           shorter_prefix_acceptance={str(n): min(n, matched) for n in range(1, len(proposed)+1)}))
    return dict(tokens=tokens, cycles=cycles, feedback_statistics=state.export())
