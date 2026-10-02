"""Portable simulator and an optional, deliberately unoptimized Llama reference backend."""

from dataclasses import dataclass
import hashlib
import math

from .demo import fraction


@dataclass(frozen=True)
class DecodeAction:
    name: str
    depth: int
    length: int


ACTIONS = (DecodeAction("ordinary", 0, 0), DecodeAction("draft4x2", 4, 2),
           DecodeAction("draft4x5", 4, 5), DecodeAction("draft8x3", 8, 3))

# These are fictional parameter sets, NOT measured hardware specifications.
SCENARIOS = {
    "cpu-demo": (1.5, 3.0, 0.025),
    "rtx4060-demo": (1.0, 1.0, 0.020),
    "rtx3050-demo": (1.2, 1.5, 0.022),
    "apple-mps-demo": (0.8, 1.8, 0.018),
    "intel-xpu-demo": (1.1, 2.2, 0.025),
}


class SimulatedBackend:
    source = "SYNTHETIC"

    def __init__(self, scenario="cpu-demo", precision="fp16", seed=42):
        self.scenario, self.precision, self.seed = scenario, precision, seed
        self.energy_scale, self.time_scale, self.controller_j = SCENARIOS[scenario]
        self.metadata = {"backend": "simulation", "scenario": scenario,
                         "precision": precision, "model": "deterministic toy token source",
                         "measurement_scope": "invented total costs; no device is measured",
                         "optimized_kv_cache": False, "real_quantization": False}

    def encode(self, prompt):
        return prompt.split()

    def decode(self, tokens):
        return " ".join(tokens)

    def next_token(self, prefix):
        # Prompt content influences the output, but this still has no language understanding.
        vocabulary = "the model verifies a draft before producing its next token .".split()
        previous = prefix[-1] if prefix else ""
        index = int(hashlib.sha256(f"{len(prefix)}:{previous}".encode()).hexdigest()[:8], 16)
        return vocabulary[index % len(vocabulary)]

    def cycle(self, prefix, action, remaining):
        if remaining <= 0:
            return [], [], 0, False
        if action.length == 0:
            return [self.next_token(prefix)], [], 0, False
        length = min(action.length, remaining - 1)
        candidates = []
        for _ in range(length):
            context = prefix + candidates
            hard = (len(context) // 32) % 3 == 2
            probability = ({4: 0.22, 8: 0.48} if hard else {4: 0.90, 8: 0.97})[action.depth]
            if self.precision == "int4":
                probability -= 0.04
            candidates.append(self.next_token(context) if fraction(self.seed, context, action.depth) < probability else "[draft-error]")
        committed, checks = [], []
        for candidate in candidates:
            correct = self.next_token(prefix + committed)
            checks.append(candidate == correct)
            committed.append(correct)
            if candidate != correct:
                return committed, checks, length, False
        committed.append(self.next_token(prefix + committed))
        return committed, checks, length, False

    def cost(self, action, context, repetition=0):
        energy, latency = {"ordinary": (1.0, 8.0), "draft4x2": (1.4, 10.0),
                           "draft4x5": (2.0, 18.0), "draft8x3": (2.3, 14.0)}[action.name]
        factor = 1 + context * (0.001 if action.length == 0 else 0.0017)
        # Quantization changes relative verification overhead in this invented model.
        quant = {"fp16": 1.0, "int8": 0.78, "int4": 0.60}[self.precision]
        speculative_overhead = 0.25 * action.length if self.precision == "int4" else 0
        noise = 0.92 + 0.16 * fraction(self.seed, context, action.name, repetition)
        return ((energy * quant + speculative_overhead) * factor * self.energy_scale * noise,
                (latency * quant + speculative_overhead * 5) * factor * self.time_scale * noise / 1000)


class TransformersBackend:
    """Real greedy reference implementation for Llama checkpoints.

    Recomputes the whole prefix; uses no KV cache. Speculation verifies with the
    full model, rather than reusing early-layer activations. This supports
    controller/correctness development, not claims of optimized LayerSkip speed.
    """

    source = "REAL_REFERENCE"

    def __init__(self, model_id, device="cpu", precision="fp32", revision=None, local_only=True):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install optional requirements-hf.txt with standard CPython first") from exc
        if precision in ("int4", "int8") and device != "cuda":
            raise ValueError("This prototype enables bitsandbytes int4/int8 only on CUDA; use fp32/fp16 elsewhere")
        if precision == "fp16" and device == "cpu":
            raise ValueError("Use fp32 for this CPU reference backend")
        self.torch, self.device = torch, device
        kwargs = {"local_files_only": local_only, "trust_remote_code": False}
        if revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        load_kwargs = dict(kwargs)
        load_kwargs["torch_dtype"] = torch.float32 if precision == "fp32" else torch.float16
        if precision in ("int4", "int8"):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=precision == "int4", load_in_8bit=precision == "int8")
            load_kwargs["device_map"] = {"": "cuda:0"}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if precision not in ("int4", "int8"):
            self.model.to(device)
        self.model.eval()
        if self.model.config.model_type != "llama":
            raise ValueError("The real reference backend currently supports Llama architecture only")
        self.layers = self.model.model.layers
        if len(self.layers) <= 8:
            raise ValueError("Need more than 8 decoder layers for this action grid")
        self.controller_j = None
        self.metadata = {"backend": device, "model": model_id,
                         "model_revision": getattr(self.model.config, "_commit_hash", None) or revision,
                         "precision": precision, "torch": torch.__version__,
                         "optimized_kv_cache": False, "real_quantization": precision in ("int4", "int8"),
                         "measurement_scope": "reference execution; no cache reuse; energy only when a valid meter is available"}
        self.eos = self.tokenizer.eos_token_id

    def encode(self, prompt):
        # Raw prompts, identical for every policy. Base LayerSkip is not assumed to be chat-tuned.
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def synchronize(self):
        if self.device != "cpu":
            getattr(self.torch, self.device).synchronize()

    def _forward(self, prefix, depth=0):
        torch = self.torch
        inputs = torch.tensor([prefix], dtype=torch.long, device=self.device)
        original_depth = self.model.config.num_hidden_layers
        try:
            if depth:
                self.model.model.layers = self.layers[:depth]
                self.model.config.num_hidden_layers = depth
            with torch.inference_mode():
                return self.model(input_ids=inputs, use_cache=False).logits[0]
        finally:
            self.model.model.layers = self.layers
            self.model.config.num_hidden_layers = original_depth

    def cycle(self, prefix, action, remaining):
        if not prefix:
            raise ValueError("Real model requires a nonempty tokenized prompt")
        if action.length == 0 or remaining == 1:
            token = int(self._forward(prefix)[-1].argmax().item())
            return [token], [], 0, token == self.eos
        candidates = []
        for _ in range(min(action.length, remaining - 1)):
            token = int(self._forward(prefix + candidates, action.depth)[-1].argmax().item())
            candidates.append(token)
            if token == self.eos:
                break
        # Logit p-1 predicts candidate 0, p predicts candidate 1, and so on.
        logits = self._forward(prefix + candidates)
        target = logits[len(prefix) - 1:].argmax(dim=-1).tolist()
        committed, checks = [], []
        for index, candidate in enumerate(candidates):
            correct = target[index]
            checks.append(candidate == correct)
            committed.append(correct)
            if correct == self.eos or candidate != correct:
                return committed, checks, len(candidates), correct == self.eos
        bonus = target[len(candidates)]
        committed.append(bonus)
        return committed, checks, len(candidates), bonus == self.eos
