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
            chunks[depth].append(model.lm_head(model.model.norm(hidden)).argmax(-1))
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
    tokens, cache = [], None
    for _ in range(limit):
        result = model(ids, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        ids = result.logits[:, -1].argmax(-1).reshape(1, 1)
        token = ids.item()
        tokens.append(token)
        if token in eos:
            break
    return tokens


@torch.inference_mode()
def generate(model, prompt, limit=32, depth=0, drafts=0, exits=(), mode='none', period=4, eos=()):
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
    result = forward(model, torch.tensor([prompt], device=model.device), None)
    cache = result.past_key_values
    pending = result.logits[:, -1].argmax(-1).reshape(1, 1)
    tokens.append(pending.item())
    while len(tokens) < limit and tokens[-1] not in eos:
        context = prompt + tokens
        check_cache(cache, layers, len(context)-1)
        k = min(drafts, limit-len(tokens)-1)
        requested = [d for d in exits if k == 0 or d != depth]
        if mode == 'none' or (mode == 'periodic' and len(cycles) % period):
            requested = []
        elif mode == 'selected':
            requested = requested[:1]
        proposed = []
        proposed_tensors = []
        with observe(model, requested) as collected:
            if k == 0:
                result = forward(model, pending, cache)
            else:
                query, draft_input = None, pending
                for _ in range(k):
                    early = forward_early(model, draft_input, cache, depth, query)
                    cache, query = early.past_key_values, early.exit_query_cache
                    draft_input = early.logits[:, -1].argmax(-1).reshape(1, 1)
                    token = draft_input.item()
                    proposed.append(token)
                    proposed_tensors.append(draft_input)
                    if token in eos:
                        break
                block = torch.cat([pending, *proposed_tensors], dim=1)
                result = forward_remainder(model, block, cache, depth, query)
        target_ids = result.logits.argmax(-1)
        target = target_ids[0].tolist()
        matched = 0
        while matched < len(proposed) and proposed[matched] == target[matched]:
            matched += 1
        committed = proposed[:matched] + [target[matched]]
        for i, token in enumerate(committed):
            if token in eos:
                committed = committed[:i+1]
                break
        predictions = {d: torch.cat(parts, dim=1)[0].tolist() for d, parts in collected.items()}
        feedback = validate_feedback(context, proposed, target, predictions, matched, len(committed))
        state.update(feedback)
        tokens.extend(committed)
        cache = crop_past_key_values(result.past_key_values, len(prompt)+len(tokens)-1)
        check_cache(cache, layers, len(prompt)+len(tokens)-1)
        pending = target_ids[:, len(committed)-1:len(committed)]
        cycles.append(dict(context_ids=context, draft=proposed, target=target,
                           accepted=matched, committed=committed, observations=feedback,
                           shorter_prefix_acceptance={str(n): min(n, matched) for n in range(1, len(proposed)+1)}))
    return dict(tokens=tokens, cycles=cycles, feedback_statistics=state.export())
