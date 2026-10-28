"""Run the real LayerSkip model with measured request-level joint selection.

The inference engine, selector, measurement, and experiment live in this package.
The attributed LayerSkip helper is vendored here under its source license.
"""
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "layerskip" / "cache"
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
# Use the existing local login path if present, otherwise Hugging Face's
# ordinary machine login. Never open or print either credential here.
if (CACHE / "token").exists() and "HF_TOKEN" not in os.environ and "HF_TOKEN_PATH" not in os.environ:
    os.environ["HF_TOKEN_PATH"] = str(CACHE / "token")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gzip
import hashlib
import json
import random
import statistics
from contextlib import ExitStack
from urllib.request import Request, urlopen

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from huggingface_hub import snapshot_download

from hardware_aware_selective_feedback.engine import generate, native_greedy, reference_greedy
from hardware_aware_selective_feedback.gpu_clocks import pinned_gpu
from hardware_aware_selective_feedback.graph_drafter import GraphDrafter
from hardware_aware_selective_feedback.measurement import Meter
from hardware_aware_selective_feedback.selector import Action, CostBandit, Selector

MODEL = "facebook/layerskip-llama3.2-1B"
REVISION = "81deb0f88734409cca506bcefcfb5c6bd2667565"
DATASETS = {
    "gsm8k": ("openai/grade-school-math", "grade_school_math/data/{}.jsonl",
              "3101c7d5072418e28b9008a6636bde82a006892c", "MIT"),
    "humaneval": ("openai/human-eval", "data/HumanEval.jsonl.gz",
                  "6d43fb980f9fee3c892a914eda09951f772ad10d", "MIT"),
}
SHORT_PROMPT_TOKENS = 64  # prompt-context boundary for the context-aware selector
MAX_PROMPT_TOKENS = 192   # prompts are truncated to this many tokens


def resolve_dtype(precision, device):
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        if device != "cuda":
            raise ValueError("FP16 is currently supported only on CUDA")
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def source_file(name, split=""):
    repo, path_pattern, revision, license_name = DATASETS[name]
    directory = CACHE / "research_datasets" / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if cached["revision"] != revision:
            raise ValueError(f"Cached {name} revision differs from pinned revision")
    else:
        manifest.write_text(json.dumps({"repo": repo, "revision": revision,
                                         "license": license_name,
                                         "source": f"https://github.com/{repo}"}),
                            encoding="utf-8")
    path = path_pattern.format(split) if name == "gsm8k" else path_pattern
    destination = directory / Path(path).name
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    if not destination.exists():
        with urlopen(Request(url, headers={"User-Agent": "energy-inference-research"}), timeout=90) as response:
            destination.write_bytes(response.read())
    raw = destination.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if name == "humaneval":
        raw = gzip.decompress(raw)
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    return records, {"source": url, "sha256": digest, "revision": revision,
                     "license": license_name}


def load_prompts(calibration, test, seed):
    if min(calibration, test) < 1:
        raise ValueError("Need positive calibration and test counts")
    train, train_meta = source_file("gsm8k", "train")
    held, held_meta = source_file("gsm8k", "test")
    code, code_meta = source_file("humaneval")
    rng = random.Random(seed)
    train = list(enumerate(train))
    held = list(enumerate(held))
    rng.shuffle(train)
    rng.shuffle(held)
    rng.shuffle(code)
    if len(train) < calibration or len(held) < test or len(code) < calibration + test:
        raise ValueError("Insufficient dataset records")
    calibration_rows = [{"id": f"gsm8k-train-{i}", "dataset": "gsm8k",
                         "text": "Question: " + row["question"] + "\nAnswer:"}
                        for i, row in train[:calibration]]
    test_rows = [{"id": f"gsm8k-test-{i}", "dataset": "gsm8k",
                  "text": "Question: " + row["question"] + "\nAnswer:"}
                 for i, row in held[:test]]
    calibration_rows += [{"id": row["task_id"], "dataset": "humaneval", "text": row["prompt"]}
                         for row in code[:calibration]]
    test_rows += [{"id": row["task_id"], "dataset": "humaneval", "text": row["prompt"]}
                  for row in code[calibration:calibration + test]]
    rng.shuffle(calibration_rows)
    rng.shuffle(test_rows)
    return calibration_rows, test_rows, {
        "gsm8k_train": train_meta, "gsm8k_test": held_meta, "humaneval": code_meta,
        "seed": seed, "split": "disjoint GSM8K train/test; disjoint shuffled HumanEval IDs",
        "use": "prompt workloads only; no answer or executable-code quality scoring",
    }


def load_model(device, precision="fp32", quantize=None, attn="auto", compile_model=False):
    """Load the pinned LayerSkip model with optional quantization and compile.

    quantize: None | 'int8' | 'int4'  — bitsandbytes precision (CUDA only)
    attn:     'auto' | 'sdpa' | 'eager' | 'flash_attention_2'
              'auto' selects sdpa on CUDA, eager elsewhere.
    compile_model: wrap with torch.compile(mode='reduce-overhead') after load.
    """
    if quantize and device != "cuda":
        raise ValueError("Quantization (int4/int8) requires CUDA")
    if quantize:
        try:
            import bitsandbytes  # noqa: F401  — presence check
        except ImportError:
            raise RuntimeError(
                "bitsandbytes is not installed. Run: pip install bitsandbytes"
            )
    dtype = resolve_dtype(precision, device)
    snapshot = CACHE / "hub" / "models--facebook--layerskip-llama3.2-1B" / "snapshots" / REVISION
    if not (snapshot / "model.safetensors").exists():
        try:
            snapshot = Path(snapshot_download(repo_id=MODEL, revision=REVISION,
                                             cache_dir=str(CACHE / "hub"),
                                             allow_patterns=["*.json", "*.safetensors",
                                                             "tokenizer.model", "*.tiktoken"]))
        except Exception as error:
            raise RuntimeError("Pinned LayerSkip download failed; use a Hugging Face account "
                               "with checkpoint access and run hf auth login") from error
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    # Attention backend: sdpa uses fused kernels on CUDA (~5-10% latency),
    # but requires re-verifying token matches after switching.
    if attn == "auto":
        attn_impl = "sdpa" if device == "cuda" else "eager"
    else:
        attn_impl = attn
    # Build keyword args for quantization.
    quant_kwargs = {}
    if quantize == "int8":
        quant_kwargs["load_in_8bit"] = True
    elif quantize == "int4":
        quant_kwargs["load_in_4bit"] = True
        quant_kwargs["bnb_4bit_compute_dtype"] = dtype
        quant_kwargs["bnb_4bit_quant_type"] = "nf4"
        quant_kwargs["bnb_4bit_use_double_quant"] = True
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation=attn_impl,
        **quant_kwargs)
    if not quantize:
        # bitsandbytes places the model on the target device during load;
        # for unquantized models we move it explicitly.
        model = model.eval().to(device)
    else:
        model = model.eval()
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requires PyTorch >= 2.0")
        # reduce-overhead removes per-step kernel-launch cost (~5-10% latency).
        # fullgraph=False because the speculative path has dynamic control flow.
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model, tokenizer, REVISION


def build_graph_drafters(model, depths, token_limit, max_draft_length):
    """One static-cache drafter per draft depth, sized for the longest request."""
    max_len = MAX_PROMPT_TOKENS + token_limit + max_draft_length + 1
    return {depth: GraphDrafter(model, depth, max_len) for depth in sorted(set(depths)) if depth}


def first_divergence(tokens, reference):
    """Index of the first differing token (or the shorter length if one is a prefix)."""
    return next((i for i, (a, b) in enumerate(zip(tokens, reference, strict=False)) if a != b),
                min(len(tokens), len(reference)))


def run_one(model, meter, prompt, token_limit, action, eos, reference=None, adaptive_drafts=False,
            graph_drafters=None, stop_on_mismatch=True, min_seconds=0.0):
    active_exits = tuple(d for d in action.observations if d != action.depth)
    # Observer hooks do not fire inside the graphed draft step, so observed
    # requests keep the ordinary forward_early path.
    drafter = (graph_drafters.get(action.depth)
               if graph_drafters and action.depth and not active_exits else None)
    output, measured = meter.run(lambda: generate(
        model, prompt, token_limit, action.depth, action.length,
        active_exits, "all" if active_exits else "none", eos=eos,
        adaptive_drafts=adaptive_drafts, graph_drafter=drafter), min_seconds)
    output["greedy_match"] = reference is None or output["tokens"] == reference
    if not output["greedy_match"] and stop_on_mismatch:
        raise AssertionError(f"Greedy output mismatch for depth={action.depth}, length={action.length}")
    return output, measured


def per_token(record, key):
    """Request cost divided by generated tokens; None when the cost is unmeasured."""
    return None if record[key] is None else record[key] / record["generated_tokens"]


def paired_per_token_delta(record, baseline, key):
    """Per-token cost difference, so policies producing more tokens are not penalised."""
    value, base = per_token(record, key), per_token(baseline, key)
    return None if value is None or base is None else value - base


def warm_up(model, prompt, token_limit, actions, eos, graph_drafters=None, runs=1):
    """Exercise every code path (kernels, allocator, CUDA graphs) before measuring."""
    for _ in range(runs):
        for action in actions:
            exits = tuple(d for d in action.observations if d != action.depth)
            drafter = (graph_drafters.get(action.depth)
                       if graph_drafters and action.depth and not exits else None)
            generate(model, prompt, token_limit, action.depth, action.length, exits,
                     "all" if exits else "none", eos=eos, graph_drafter=drafter)


def prompt_context(prompt):
    """Coarse prompt type used to specialise the selector: dataset plus length bucket."""
    length = "short" if len(prompt["ids"]) < SHORT_PROMPT_TOKENS else "long"
    return f"{prompt['dataset']}:{length}"


def row(prompt, policy, action, output, measured, split, repeat=0, order=0):
    observations = [o for c in output["cycles"] for o in c["observations"]]
    return {
        "prompt_id": prompt["id"], "dataset": prompt["dataset"], "split": split,
        "policy": policy, "repeat": repeat, "order": order,
        "depth": action.depth, "draft_length": action.length,
        "observed_exits": action.observations, "tokens": output["tokens"],
        "generated_tokens": len(output["tokens"]), "latency_s": measured["latency_s"],
        "measure_repetitions": measured.get("measure_repetitions", 1),
        "energy_j": measured["energy_j"], "energy_boundary": measured["boundary"],
        "energy_source": measured["source"],
        "energy_reason": measured["energy_reason"], "valid_feedback": sum(o["valid"] for o in observations),
        "excluded_feedback": sum(not o["valid"] for o in observations),
        "accepted_drafts": sum(c["accepted"] for c in output["cycles"]),
        "drafted_tokens": sum(len(c["draft"]) for c in output["cycles"]),
        "greedy_match": output.get("greedy_match", True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=int, default=4, help="prompts per dataset")
    parser.add_argument("--test", type=int, default=2, help="held-out prompts per dataset")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu", "mps"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--output", type=Path,
                        help="JSON report path (Markdown report uses the same name with .md)")
    parser.add_argument("--energy-source", choices=("auto", "none", "nvml", "rapl", "emi", "file"), default="auto")
    parser.add_argument("--counter-file", help="external cumulative energy counter text file")
    parser.add_argument("--counter-unit", choices=("joules", "millijoules", "microjoules"), default="joules")
    parser.add_argument("--energy-scope", help="required for external file, e.g. whole_device")
    parser.add_argument("--probe", action="store_true", help="check device and energy counter without loading model")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1, help="held-out repeats per prompt")
    parser.add_argument("--objective", choices=("auto", "latency", "energy"), default="auto")
    parser.add_argument("--observer-pairs", type=int, default=3,
                        help="paired base/observed calibration repetitions")
    # --- Efficiency levers ---
    parser.add_argument("--quantize", choices=("int4", "int8"), default=None,
                        help="bitsandbytes weight quantization (CUDA only). "
                             "int4 uses NF4 double-quant (~25%% energy reduction); "
                             "int8 uses LLM.int8 (~10-15%%). "
                             "Re-verify token matches after enabling.")
    parser.add_argument("--compile", action="store_true", dest="compile_model",
                        help="torch.compile the model with mode=reduce-overhead "
                             "(PyTorch >= 2.0, CUDA recommended, ~5-10%% latency).")
    parser.add_argument("--attn", choices=("auto", "sdpa", "eager", "flash_attention_2"),
                        default="auto",
                        help="Attention backend. 'auto' selects sdpa on CUDA, "
                             "eager on CPU/MPS. Re-verify token matches after "
                             "switching from eager.")
    parser.add_argument("--adaptive-drafts", action="store_true",
                        help="Shorten drafts below 50%% recent acceptance and decode "
                             "ordinarily below 30%%, with periodic re-probes.")
    parser.add_argument("--context-selector", action="store_true",
                        help="Score selector arms per prompt context (dataset and "
                             "short/long prompt) once that context has samples.")
    parser.add_argument("--stop-on-mismatch", action="store_true",
                        help="Abort on the first greedy mismatch (old behaviour). By default "
                             "the mismatch is recorded and that configuration is excluded "
                             "from the adaptive selector and bandit.")
    parser.add_argument("--cuda-graph-drafts", action="store_true",
                        help="PROTOTYPE: replay draft steps as CUDA Graphs with a static "
                             "KV cache (CUDA only; not with --compile or --quantize).")
    parser.add_argument("--min-measure-seconds", type=float, default=None,
                        help="Repeat each measured request until this window has elapsed and "
                             "report the per-request mean; NVML updates too coarsely for "
                             "sub-second requests. Default: 1.0 with an energy counter, else 0.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Unmeasured warm-up passes over every action before calibration.")
    parser.add_argument("--gpu-clock-mhz", type=int, default=None,
                        help="Lock the NVIDIA graphics clock during the run (admin/root).")
    parser.add_argument("--gpu-power-limit-w", type=float, default=None,
                        help="Cap NVIDIA board power during the run (admin/root).")
    args = parser.parse_args()
    if min(args.calibration, args.test, args.tokens, args.threads,
           args.observer_pairs, args.repeats) < 1:
        parser.error("counts must be positive")
    if (args.min_measure_seconds or 0) < 0 or args.warmup < 0:
        parser.error("--min-measure-seconds and --warmup must be nonnegative")
    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("xpu" if xpu_available else
                 ("mps" if mps_available else "cpu"))
    else:
        device = args.device
    if (device == "cuda" and not torch.cuda.is_available()) or (device == "xpu" and not xpu_available) or (device == "mps" and not mps_available):
        parser.error(f"{device} is unavailable in this PyTorch installation or on this machine")
    try:
        resolve_dtype(args.precision, device)
    except ValueError as error:
        parser.error(str(error))
    if (args.gpu_clock_mhz is not None or args.gpu_power_limit_w is not None) and device != "cuda":
        parser.error("--gpu-clock-mhz and --gpu-power-limit-w require --device cuda")
    if args.cuda_graph_drafts and (device != "cuda" or args.compile_model or args.quantize):
        parser.error("--cuda-graph-drafts requires --device cuda and no --compile/--quantize")
    meter = Meter(device, args.energy_source, args.counter_file,
                  args.counter_unit, args.energy_scope)
    if args.probe:
        print(json.dumps(meter.describe(), indent=2))
        meter.close()
        return
    if args.objective == "energy" and not meter.available:
        parser.error("Energy requested but no working counter: " + str(meter.reason))
    objective = ("energy" if meter.available else "latency") if args.objective == "auto" else args.objective
    if args.min_measure_seconds is None:
        args.min_measure_seconds = 1.0 if meter.available else 0.0
    torch.set_num_threads(args.threads)
    calibration, heldout, dataset_info = load_prompts(args.calibration, args.test, args.seed)
    model, tokenizer, revision = load_model(
        device, args.precision,
        quantize=args.quantize,
        attn=args.attn,
        compile_model=args.compile_model,
    )
    eos = (tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ()
    for prompt in calibration + heldout:
        prompt["ids"] = tokenizer.encode(prompt.pop("text"), add_special_tokens=True)[:MAX_PROMPT_TOKENS]
    selector = Selector(objective=objective, context_aware=args.context_selector)
    drafters = (build_graph_drafters(model, selector.depths, args.tokens,
                                     max(arm.length for arm in selector.arms))
                if args.cuda_graph_drafts else None)
    bandit = CostBandit(selector.arms, objective=objective)
    rows = []
    mismatches = []
    excluded = set()

    def execute(prompt, policy, action, reference, split, repeat=0, order=0):
        """Run one request, record its row, and exclude a mismatching configuration."""
        result = run_one(model, meter, prompt["ids"], args.tokens, action, eos, reference,
                         args.adaptive_drafts, drafters, args.stop_on_mismatch,
                         args.min_measure_seconds)
        rows.append(row(prompt, policy, action, *result, split, repeat, order))
        if not result[0]["greedy_match"]:
            index = first_divergence(result[0]["tokens"], reference)
            mismatches.append({
                "prompt_id": prompt["id"], "policy": policy, "split": split,
                "repeat": repeat, "depth": action.depth, "draft_length": action.length,
                "token_index": index,
                "actual": result[0]["tokens"][index:index + 1],
                "expected": reference[index:index + 1],
            })
            if action.key not in excluded and action.key != (0, 0):
                excluded.add(action.key)
                selector.exclude(action.key)
                bandit.exclude(action.key)
                print(f"Greedy mismatch: excluding depth={action.depth}, "
                      f"length={action.length} from adaptive selection", flush=True)
        return result

    native_agreement = {}

    def reference_for(prompt):
        """Same-code ordinary decoding is the reference; HF output is a recorded cross-check."""
        reference = reference_greedy(model, prompt["ids"], args.tokens, eos)
        native_agreement[prompt["id"]] = native_greedy(model, prompt["ids"], args.tokens, eos) == reference
        return reference

    print(f"Model loaded; objective={objective}; energy boundary={meter.scope or 'unavailable'}",
          flush=True)

    cleanup = ExitStack()
    try:
        gpu_settings = cleanup.enter_context(
            pinned_gpu(args.gpu_clock_mhz, args.gpu_power_limit_w))
        # Calibrate base actions on the first prompt, then observer overhead
        # with executed same-prompt pairs across distinct calibration prompts.
        first = calibration[0]
        warm_up(model, first["ids"], args.tokens,
                selector.arms + tuple(Action(arm.depth, arm.length, selector.observer_exits(arm.depth)[1])
                                      for arm in selector.arms if arm.depth),
                eos, drafters, args.warmup)
        ref = reference_for(first)
        for arm in selector.arms:
            result = execute(first, "calibration", arm, ref, "calibration")
            if result[0]["greedy_match"]:
                selector.update(arm, result[1], result[0], prompt_context(first))
                bandit.update(arm, result[1], result[0])
        # Calibrate observer overhead on the cheapest base arm, since that is
        # where the selector would actually pay for observations.
        pair_base = selector.best_speculative() or next(arm for arm in selector.arms if arm.depth)
        selected_exits, all_exits = selector.observer_exits(pair_base.depth)
        observed_arm = Action(pair_base.depth, pair_base.length, selected_exits)
        all_arm = Action(pair_base.depth, pair_base.length, all_exits)
        pair_prompt_ids = []
        pair_rng = random.Random(args.seed + 2)
        for pair_index in range(args.observer_pairs):
            prompt = calibration[pair_index % len(calibration)]
            pair_prompt_ids.append(prompt["id"])
            reference = ref if prompt is first else reference_for(prompt)
            modes = [("selected", observed_arm), ("all", all_arm)]
            pair_rng.shuffle(modes)
            for mode, arm in modes:
                base_arm = Action(arm.depth, arm.length)
                order = ["base", "observed"]
                pair_rng.shuffle(order)
                results = {}
                for position, kind in enumerate(order):
                    chosen = base_arm if kind == "base" else arm
                    results[kind] = execute(prompt, "calibration_pair_" + mode + "_" + kind,
                                            chosen, reference, "calibration", pair_index, position)
                    if results[kind][0]["greedy_match"]:
                        selector.update(chosen, results[kind][1], results[kind][0],
                                        prompt_context(prompt))
                if all(r[0]["greedy_match"] for r in results.values()):
                    selector.record_observer_pair(results["base"], results["observed"], mode)

        # Additional calibration prompts: update the online selector.
        for index, prompt in enumerate(calibration[1:]):
            reference = reference_for(prompt)
            action = selector.choose(len(calibration) + len(heldout) * args.repeats - index,
                                     prompt_context(prompt))
            bandit_action = bandit.choose()
            result = execute(prompt, "adaptive_calibration", action, reference, "calibration")
            if result[0]["greedy_match"]:
                selector.update(action, result[1], result[0], prompt_context(prompt))
            bandit_result = execute(prompt, "bandit_calibration", bandit_action, reference,
                                    "calibration")
            if bandit_result[0]["greedy_match"]:
                bandit.update(bandit_action, bandit_result[1], bandit_result[0])

        # Test: same prompt/reference for every policy. Online selector may
        # learn from previous test requests, but no parameters are retuned.
        rng = random.Random(args.seed + 1)
        request_index = 0
        total_requests = len(heldout) * args.repeats
        for prompt_index, prompt in enumerate(heldout):
            reference = reference_for(prompt)
            for repeat in range(args.repeats):
                adaptive_action = selector.choose(total_requests - request_index,
                                                  prompt_context(prompt))
                bandit_action = bandit.choose()
                policies = [
                    ("ordinary", Action()),
                    ("fixed", Action(4, 3)),
                    ("never_observe", Action(pair_base.depth, pair_base.length)),
                    ("selected_observe", observed_arm),
                    ("always_observe", all_arm),
                    ("bandit", bandit_action),
                    ("adaptive", adaptive_action),
                ]
                rng.shuffle(policies)
                for order, (name, action) in enumerate(policies):
                    result = execute(prompt, name, action, reference, "test", repeat, order)
                    if not result[0]["greedy_match"]:
                        continue
                    if name == "adaptive":
                        selector.update(action, result[1], result[0], prompt_context(prompt))
                    elif name == "bandit":
                        bandit.update(action, result[1], result[0])
                request_index += 1
            print(f"Completed test prompt {prompt_index + 1}/{len(heldout)}", flush=True)
    finally:
        try:
            cleanup.close()
        finally:
            meter.close()

    aggregates = {}
    for name in ("ordinary", "fixed", "never_observe", "selected_observe", "always_observe", "bandit", "adaptive"):
        group = [r for r in rows if r["policy"] == name and r["split"] == "test"]
        aggregates[name] = {
            "runs": len(group),
            "median_latency_s": statistics.median(r["latency_s"] for r in group),
            "total_latency_s": sum(r["latency_s"] for r in group),
            "total_generated_tokens": sum(r["generated_tokens"] for r in group),
            "latency_s_per_token": (sum(r["latency_s"] for r in group) /
                                    sum(r["generated_tokens"] for r in group)),
            "energy_j": (sum(r["energy_j"] for r in group)
                         if all(r["energy_j"] is not None for r in group) else None),
            "energy_j_per_token": (sum(r["energy_j"] for r in group) /
                                   sum(r["generated_tokens"] for r in group)
                                   if all(r["energy_j"] is not None for r in group) else None),
            "all_greedy_match": all(r["greedy_match"] for r in group),
            "greedy_mismatches": sum(not r["greedy_match"] for r in group),
        }
    paired_observer_deltas = []
    paired_vs_ordinary = []
    for prompt in heldout:
        for repeat in range(args.repeats):
            group = {r["policy"]: r for r in rows if r["split"] == "test"
                     and r["prompt_id"] == prompt["id"] and r["repeat"] == repeat}
            for mode in ("selected_observe", "always_observe"):
                paired_observer_deltas.append({
                    "prompt_id": prompt["id"], "repeat": repeat, "mode": mode,
                    "latency_delta_s_per_token": paired_per_token_delta(
                        group[mode], group["never_observe"], "latency_s"),
                    "energy_delta_j_per_token": paired_per_token_delta(
                        group[mode], group["never_observe"], "energy_j"),
                })
            for policy in ("fixed", "never_observe", "selected_observe", "always_observe",
                           "bandit", "adaptive"):
                paired_vs_ordinary.append({
                    "prompt_id": prompt["id"], "repeat": repeat, "policy": policy,
                    "latency_delta_s_per_token": paired_per_token_delta(
                        group[policy], group["ordinary"], "latency_s"),
                    "energy_delta_j_per_token": paired_per_token_delta(
                        group[policy], group["ordinary"], "energy_j"),
                })
    # Feasibility checks use only executed held-out policies. They are
    # descriptive, not a counterfactual oracle for unexecuted configurations.
    objective_key = "energy_j" if objective == "energy" else "latency_s"
    fixed_candidates = ("ordinary", "fixed", "never_observe")
    winner_by_prompt = {}
    for prompt in heldout:
        means = {}
        for policy in fixed_candidates:
            # Only exact-output runs can win; a mismatching setting is not a valid candidate.
            group = [r for r in rows if r["split"] == "test" and r["greedy_match"] and
                     r["prompt_id"] == prompt["id"] and r["policy"] == policy]
            if group:
                means[policy] = statistics.median(r[objective_key] / r["generated_tokens"]
                                                  for r in group)
        if means:
            winner_by_prompt[prompt["id"]] = min(means, key=means.get)
    observation_overhead = {}
    for mode in ("selected_observe", "always_observe"):
        selected = [r for r in paired_observer_deltas if r["mode"] == mode]
        deltas = [r["energy_delta_j_per_token" if objective == "energy" else "latency_delta_s_per_token"]
                  for r in selected]
        observation_overhead[mode] = {"mean_delta_per_token": statistics.mean(deltas),
                                      "median_delta_per_token": statistics.median(deltas),
                                      "n": len(deltas)}
    feasibility = {
        "best_executed_fixed_by_prompt": winner_by_prompt,
        "best_fixed_varies_across_prompts": len(set(winner_by_prompt.values())) > 1,
        "observation_overhead": observation_overhead,
        "warning": "Descriptive short-run comparisons; repeat on more prompts and hardware.",
    }
    calibration_rows = [r for r in rows if r["split"] == "calibration"]
    objective_key = "energy_j" if objective == "energy" else "latency_s"
    report = {
        "calibration_runs": len(calibration_rows),
        "observer_pair_prompt_ids": pair_prompt_ids,
        "observer_pair_unique_prompts": len(set(pair_prompt_ids)),
        "calibration_total_objective": sum(r[objective_key] for r in calibration_rows),
        "paired_observer_deltas": paired_observer_deltas,
        "paired_vs_ordinary": paired_vs_ordinary,
        "mismatches": mismatches,
        "reference": "ordinary decoding through the same vendored forward code as speculation",
        "native_hf_agreement": native_agreement,
        "observer_pair_base": [pair_base.depth, pair_base.length],
        "excluded_configurations": sorted(list(key) for key in excluded),
        "feasibility": feasibility,
        "model": MODEL, "revision": revision, "precision": args.precision,
        "device": str(model.device), "processor": platform.processor(),
        "measurement": meter.describe(),
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "accelerator_name": (torch.cuda.get_device_name() if device == "cuda" else
                             torch.xpu.get_device_name() if device == "xpu" else
                             torch.mps.get_name() if device == "mps" and hasattr(torch.mps, "get_name") else None),
        "torch": torch.__version__,
        "transformers": transformers.__version__, "objective": objective,
        "energy_available": meter.available, "energy_boundary": meter.scope,
        "energy_limitation": meter.reason,
        "dataset": dataset_info,
        "arguments": {**vars(args), "output": str(args.output) if args.output else None},
        "gpu_pinning": gpu_settings,
        "selector": selector.summary(), "bandit": bandit.summary(),
        "aggregates": aggregates, "rows": rows,
        "interpretation": "Measured implementation check; small samples do not establish energy savings, superiority, or novelty.",
        "limits": [
            "The implementation uses an attributed LayerSkip inference helper in hardware_aware_selective_feedback.",
            "Request-level adaptation, not token-level adaptation.",
            "Observer overhead uses repeated paired calibration; thermal/order effects remain.",
            "Fixed depth 4 is predetermined; observer baselines use the cheapest calibrated arm.",
            "No optimized serving backend; counter scope may omit other hardware components.",
            "HumanEval supplies prompt diversity; generated code is not executed or scored.",
        ],
    }
    output_file = (args.output if args.output is not None else
                   ROOT / "hardware_aware_selective_feedback" / "results" / "latest.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# LayerSkip selective-feedback experiment",
        "",
        f"Model: {MODEL} at {revision}; {args.precision} on {model.device}.",
        f"Objective: {objective}. Energy boundary: {report['energy_boundary'] or 'unavailable'}.",
        f"Counter: {meter.source or meter.reason}.",
        f"Observer pairs: {len(pair_prompt_ids)} across {len(set(pair_prompt_ids))} calibration prompts.",
        f"Best executed fixed setting varies across prompts: {feasibility['best_fixed_varies_across_prompts']}.",
        f"Prompts: {len(calibration)} calibration, {len(heldout)} held out; "
        f"{args.repeats} held-out repeat(s), {args.tokens} maximum generated tokens.",
        (f"All {len(rows)} executed requests matched the same-code greedy reference."
         if not mismatches else
         f"{len(rows) - len(mismatches)}/{len(rows)} executed requests matched the same-code greedy reference. "
         f"Excluded from adaptive/bandit selection after a mismatch: "
         + ", ".join(f"depth={d} length={k}" for d, k in sorted(excluded)) + ". "
         "Policies with mismatches are not valid exact-output comparisons."),
        "",
        "| Policy | Runs | Latency s/token | Energy J/token | Greedy mismatches |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in aggregates.items():
        energy = (f"{result['energy_j_per_token']:.6f}" if result['energy_j_per_token'] is not None
                  else "unavailable")
        lines.append(f"| {name} | {result['runs']} | {result['latency_s_per_token']:.6f} | {energy} "
                     f"| {result['greedy_mismatches']} |")
    lines += [
        "", f"Observed feedback cost per token at depth={pair_base.depth} length={pair_base.length} "
        "(selected/all): " +
        ", ".join(f"{name} {value['median_delta_per_token']:+.6f} " +
                  ("J" if objective == "energy" else "s")
                  for name, value in observation_overhead.items()) + ".",
        f"Reference: same-code ordinary decoding. Hugging Face native greedy agreed on "
        f"{sum(native_agreement.values())}/{len(native_agreement)} prompts (cross-check only).",
        "Observer overhead is measured by same-prompt paired executions. "
        f"Raw runs, order, token IDs, valid feedback, and paired deltas are in {output_file.name}.",
        "Calibration work is reported separately and is not allocated to any single baseline.",
        "This is a correctness and feasibility run. Small samples and hardware measurement variability "
        "do not establish a speed or energy advantage, and there is no novelty claim.",
    ]
    output_file.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_file), "runs": len(rows), "objective": objective,
                      "greedy_matches": sum(r["greedy_match"] for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
