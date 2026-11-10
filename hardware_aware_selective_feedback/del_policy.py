"""Independent greedy DEL paper implementation on the local LayerSkip engine.

Algorithm attribution: Zarch et al., DEL, COLM 2025, arXiv:2504.05598v2,
Sections 4.1--4.4. No upstream DEL source is vendored. See DEL_BASELINE.md
for numerical conventions and differences from the released implementation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch

from .engine import _check_cache
from .vendor.self_speculation.llama_model_utils import (
    crop_past_key_values,
    forward,
    forward_early,
    forward_remainder,
)


@dataclass(frozen=True)
class DELConfig:
    max_drafts: int = 18
    decay: float = 0.95
    prefill_window: int = 32
    projection_chunk: int = 32

    def __post_init__(self):
        if min(self.max_drafts, self.prefill_window, self.projection_chunk) < 1:
            raise ValueError("DEL counts must be positive")
        if not 0 < self.decay <= 1:
            raise ValueError("DEL decay must be in (0, 1]")


class DELController:
    """Decayed valid-context agreement/confidence, TPL, and draft thresholds."""

    def __init__(self, layers, config=DELConfig()):
        if layers < 2:
            raise ValueError("DEL requires at least two layers")
        self.layers, self.config = layers, config
        # [matches, mismatches, matched confidence sum, mismatched sum]
        self.stats = {depth: [0.0] * 4 for depth in range(1, layers)}

    def update(self, predictions, confidence, target, valid_count):
        if not 0 < valid_count <= len(target):
            raise ValueError("Invalid DEL committed-prefix length")
        for depth, values in self.stats.items():
            if len(predictions[depth]) != len(target) or len(confidence[depth]) != len(target):
                raise ValueError("Misaligned DEL shadow predictions")
            values[:] = [v * self.config.decay for v in values]
            for predicted, probability, truth in zip(
                predictions[depth][:valid_count],
                confidence[depth][:valid_count],
                target[:valid_count],
                strict=True,
            ):
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Invalid DEL confidence")
                bucket = 0 if predicted == truth else 1
                values[bucket] += 1
                values[bucket + 2] += probability

    def threshold(self, depth):
        matches, misses, true_sum, false_sum = self.stats[depth]
        # Do not invent a class mean when no examples of that class exist.
        # Like upstream's NaN comparison, no early confidence stop until both
        # classes have been observed. Explicit zero avoids propagating NaNs.
        return (true_sum / matches + false_sum / misses) / 2 if matches and misses else 0.0

    def choose(self):
        # d=0 has score 1/L for every depth; deterministic shallow-first ties.
        best = (1.0 / self.layers, 1, 0)
        for depth, (matches, misses, _, _) in self.stats.items():
            if not matches + misses:
                continue
            alpha = matches / (matches + misses)
            expected, power = 1.0, 1.0
            for length in range(1, self.config.max_drafts + 1):
                power *= alpha
                expected += power
                score = expected / (self.layers + length * depth)
                if score > best[0]:
                    best = score, depth, length
        return best[1], best[2], best[0]


@contextmanager
def capture_hidden(model, tail=None):
    """Capture intermediate states without extra transformer forward passes."""
    states = {d: [] for d in range(1, len(model.model.layers))}
    handles = []

    def hook(depth):
        def collect(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[depth].append(hidden[:, -tail:] if tail else hidden)

        return collect

    try:
        for depth in states:
            handles.append(model.model.layers[depth - 1].register_forward_hook(hook(depth)))
        yield states
    finally:
        for handle in handles:
            handle.remove()


def shadow_predictions(model, states, positions, chunk_size):
    """Batch LM-head work, bounding vocabulary-logit allocation on small GPUs.

    Input order is depth-major, with draft/remainder chunks concatenated into
    the same token positions at every layer. Only token IDs and top-1
    probabilities survive projection, not the full layer x token x vocab cube.
    """
    depths = list(states)
    hiddens = []
    for depth in depths:
        hidden = torch.cat(states[depth], dim=1)
        if hidden.shape[1] != positions:
            raise RuntimeError(f"DEL shadow alignment failed at depth {depth}")
        hiddens.append(hidden)
    joined = torch.cat(hiddens, dim=1)
    ids, scores = [], []
    for start in range(0, joined.shape[1], chunk_size):
        logits = model.lm_head(model.model.norm(joined[:, start : start + chunk_size])).float()
        top, token = logits.max(dim=-1)
        probability = (top - logits.logsumexp(dim=-1)).exp().clamp(0, 1)
        ids.append(token)
        scores.append(probability)
        del logits
    ids = torch.cat(ids, dim=1).reshape(len(depths), positions).tolist()
    scores = torch.cat(scores, dim=1).reshape(len(depths), positions).tolist()
    return dict(zip(depths, ids)), dict(zip(depths, scores))


@torch.inference_mode()
def generate_del(model, prompt, limit=32, eos=(), config=DELConfig()):
    """Greedy-only DEL port; controller state resets for every request.

    All shadow projections and controller decisions execute inside this call
    so Meter.run includes them. A complete cache at each round boundary
    permits changing the exit depth without rebuilding the committed prefix.
    """
    if not prompt or limit < 0:
        raise ValueError("Require nonempty prompt and nonnegative limit")
    controller = DELController(len(model.model.layers), config)
    tokens, cycles = [], []
    if limit == 0:
        return dict(tokens=tokens, cycles=cycles, feedback_statistics={})
    ids = torch.tensor([prompt], device=model.device)
    window = min(len(prompt), config.prefill_window)
    with capture_hidden(model, tail=window) as states:
        result = forward(model, ids, None, logits_to_keep=window)
    predictions, confidence = shadow_predictions(model, states, window, config.projection_chunk)
    target = result.logits.argmax(-1)[0].tolist()
    controller.update(predictions, confidence, target, window)
    cache = result.past_key_values
    pending = result.logits[:, -1].argmax(-1).reshape(1, 1)
    tokens.append(pending.item())
    del result, states, predictions, confidence

    while len(tokens) < limit and tokens[-1] not in eos:
        _check_cache(cache, controller.layers, len(prompt) + len(tokens) - 1)
        depth, tpl_length, tpl = controller.choose()
        threshold = controller.threshold(depth)
        # TPL chooses the exit. Dynamic confidence stopping sets the actual
        # length (paper Algorithm 1), up to d_max, not the TPL argmax length.
        budget = min(config.max_drafts, limit - len(tokens) - 1) if tpl_length else 0
        proposed = []
        with capture_hidden(model) as states:
            if budget == 0:
                result = forward(model, pending, cache)
            else:
                query, draft_input = None, pending
                for _ in range(budget):
                    early = forward_early(model, draft_input, cache, depth, query)
                    cache, query = early.past_key_values, early.exit_query_cache
                    logits = early.logits[:, -1].float()
                    top, token = logits.max(-1)
                    probability = (top - logits.logsumexp(-1)).exp().item()
                    draft_input = token.reshape(1, 1)
                    proposed.append(draft_input)
                    del early, logits
                    # Include the triggering token, then stop, as in the
                    # released greedy generator. Remaining budget always wins.
                    if draft_input.item() in eos or probability < threshold:
                        break
                block = torch.cat([pending, *proposed], dim=1)
                result = forward_remainder(model, block, cache, depth, query)
                del query
        target_ids = result.logits.argmax(-1)
        target = target_ids[0].tolist()
        draft = torch.cat(proposed, dim=1)[0].tolist() if proposed else []
        matched = 0
        while matched < len(draft) and draft[matched] == target[matched]:
            matched += 1
        committed = draft[:matched] + [target[matched]]
        for index, token in enumerate(committed):
            if token in eos:
                committed = committed[: index + 1]
                break
        predictions, confidence = shadow_predictions(
            model,
            states,
            len(target),
            config.projection_chunk,
        )
        valid_count = len(committed)
        controller.update(predictions, confidence, target, valid_count)
        tokens.extend(committed)
        cache = crop_past_key_values(result.past_key_values, len(prompt) + len(tokens) - 1)
        pending = target_ids[:, valid_count - 1 : valid_count]
        _check_cache(cache, controller.layers, len(prompt) + len(tokens) - 1)
        cycles.append(
            dict(
                depth=depth if budget else 0,
                tpl_depth=depth,
                tpl_draft_length=tpl_length,
                tpl_score=tpl,
                threshold=threshold,
                draft=draft,
                target=target,
                accepted=matched,
                committed=committed,
                observations=[],
                valid_shadow_predictions=valid_count * (controller.layers - 1),
                excluded_shadow_predictions=(len(target) - valid_count) * (controller.layers - 1),
            )
        )
        del result, states, predictions, confidence
    return dict(
        tokens=tokens, cycles=cycles, feedback_statistics={}, del_statistics=controller.stats
    )
