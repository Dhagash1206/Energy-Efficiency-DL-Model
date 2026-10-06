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
from urllib.request import Request, urlopen

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from huggingface_hub import snapshot_download

from hardware_aware_selective_feedback.engine import generate, native_greedy
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


def load_model(device):
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
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager").eval().to(device)
    return model, tokenizer, REVISION


def run_one(model, meter, prompt, token_limit, action, eos, reference=None):
    active_exits = tuple(d for d in action.observations if d != action.depth)
    output, measured = meter.run(lambda: generate(
        model, prompt, token_limit, action.depth, action.length,
        active_exits, "all" if active_exits else "none", eos=eos))
    if reference is not None and output["tokens"] != reference:
        raise AssertionError(f"Greedy output mismatch for depth={action.depth}, length={action.length}")
    return output, measured


def row(prompt, policy, action, output, measured, split, repeat=0, order=0):
    observations = [o for c in output["cycles"] for o in c["observations"]]
    return {
        "prompt_id": prompt["id"], "dataset": prompt["dataset"], "split": split,
        "policy": policy, "repeat": repeat, "order": order,
        "depth": action.depth, "draft_length": action.length,
        "observed_exits": action.observations, "tokens": output["tokens"],
        "generated_tokens": len(output["tokens"]), "latency_s": measured["latency_s"],
        "energy_j": measured["energy_j"], "energy_boundary": measured["boundary"],
        "energy_source": measured["source"],
        "energy_reason": measured["energy_reason"], "valid_feedback": sum(o["valid"] for o in observations),
        "excluded_feedback": sum(not o["valid"] for o in observations),
        "accepted_drafts": sum(c["accepted"] for c in output["cycles"]),
        "drafted_tokens": sum(len(c["draft"]) for c in output["cycles"]),
        "greedy_match": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=int, default=4, help="prompts per dataset")
    parser.add_argument("--test", type=int, default=2, help="held-out prompts per dataset")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu", "mps"), default="auto")
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
    args = parser.parse_args()
    if min(args.calibration, args.test, args.tokens, args.threads,
           args.observer_pairs, args.repeats) < 1:
        parser.error("counts must be positive")
    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("xpu" if xpu_available else
                 ("mps" if mps_available else "cpu"))
    else:
        device = args.device
    if (device == "cuda" and not torch.cuda.is_available()) or (device == "xpu" and not xpu_available) or (device == "mps" and not mps_available):
        parser.error(f"{device} is unavailable in this PyTorch installation or on this machine")
    meter = Meter(device, args.energy_source, args.counter_file,
                  args.counter_unit, args.energy_scope)
    if args.probe:
        print(json.dumps(meter.describe(), indent=2))
        meter.close()
        return
    if args.objective == "energy" and not meter.available:
        parser.error("Energy requested but no working counter: " + str(meter.reason))
    objective = ("energy" if meter.available else "latency") if args.objective == "auto" else args.objective
    torch.set_num_threads(args.threads)
    calibration, heldout, dataset_info = load_prompts(args.calibration, args.test, args.seed)
    model, tokenizer, revision = load_model(device)
    eos = (tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ()
    for prompt in calibration + heldout:
        prompt["ids"] = tokenizer.encode(prompt.pop("text"), add_special_tokens=True)[:192]
    selector = Selector(objective=objective)
    bandit = CostBandit(selector.arms, objective=objective)
    rows = []
    print(f"Model loaded; objective={objective}; energy boundary={meter.scope or 'unavailable'}",
          flush=True)

    try:
        # Calibrate base actions on the first prompt, then observer overhead
        # with executed same-prompt pairs across distinct calibration prompts.
        first = calibration[0]
        ref = native_greedy(model, first["ids"], args.tokens, eos)
        for arm in selector.arms:
            result = run_one(model, meter, first["ids"], args.tokens, arm, eos, ref)
            selector.update(arm, result[1], result[0])
            bandit.update(arm, result[1], result[0])
            rows.append(row(first, "calibration", arm, *result, "calibration"))
        observed_arm = Action(8, 3, (4,))
        all_arm = Action(8, 3, (4, 12))
        pair_prompt_ids = []
        pair_rng = random.Random(args.seed + 2)
        for pair_index in range(args.observer_pairs):
            prompt = calibration[pair_index % len(calibration)]
            pair_prompt_ids.append(prompt["id"])
            reference = ref if prompt is first else native_greedy(model, prompt["ids"], args.tokens, eos)
            modes = [("selected", observed_arm), ("all", all_arm)]
            pair_rng.shuffle(modes)
            for mode, arm in modes:
                base_arm = Action(arm.depth, arm.length)
                order = ["base", "observed"]
                pair_rng.shuffle(order)
                results = {}
                for position, kind in enumerate(order):
                    chosen = base_arm if kind == "base" else arm
                    results[kind] = run_one(model, meter, prompt["ids"], args.tokens,
                                            chosen, eos, reference)
                    selector.update(chosen, results[kind][1], results[kind][0])
                    rows.append(row(prompt, "calibration_pair_" + mode + "_" + kind,
                                    chosen, *results[kind], "calibration", pair_index, position))
                selector.record_observer_pair(results["base"], results["observed"], mode)

        # Additional calibration prompts: update the online selector.
        for index, prompt in enumerate(calibration[1:]):
            reference = native_greedy(model, prompt["ids"], args.tokens, eos)
            action = selector.choose(len(calibration) + len(heldout) * args.repeats - index)
            bandit_action = bandit.choose()
            result = run_one(model, meter, prompt["ids"], args.tokens, action, eos, reference)
            selector.update(action, result[1], result[0])
            rows.append(row(prompt, "adaptive_calibration", action, *result, "calibration"))
            bandit_result = run_one(model, meter, prompt["ids"], args.tokens, bandit_action, eos, reference)
            bandit.update(bandit_action, bandit_result[1], bandit_result[0])
            rows.append(row(prompt, "bandit_calibration", bandit_action,
                            *bandit_result, "calibration"))

        # Test: same prompt/reference for every policy. Online selector may
        # learn from previous test requests, but no parameters are retuned.
        rng = random.Random(args.seed + 1)
        request_index = 0
        total_requests = len(heldout) * args.repeats
        for prompt_index, prompt in enumerate(heldout):
            reference = native_greedy(model, prompt["ids"], args.tokens, eos)
            for repeat in range(args.repeats):
                adaptive_action = selector.choose(total_requests - request_index)
                bandit_action = bandit.choose()
                policies = [
                    ("ordinary", Action()),
                    ("fixed", Action(4, 3)),
                    ("never_observe", Action(8, 3)),
                    ("selected_observe", Action(8, 3, (4,))),
                    ("always_observe", Action(8, 3, (4, 12))),
                    ("bandit", bandit_action),
                    ("adaptive", adaptive_action),
                ]
                rng.shuffle(policies)
                for order, (name, action) in enumerate(policies):
                    result = run_one(model, meter, prompt["ids"], args.tokens,
                                     action, eos, reference)
                    if name == "adaptive":
                        selector.update(action, result[1], result[0])
                    elif name == "bandit":
                        bandit.update(action, result[1], result[0])
                    rows.append(row(prompt, name, action, *result, "test", repeat, order))
                request_index += 1
            print(f"Completed test prompt {prompt_index + 1}/{len(heldout)}", flush=True)
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
                    "latency_delta_s": group[mode]["latency_s"] - group["never_observe"]["latency_s"],
                    "energy_delta_j": (group[mode]["energy_j"] - group["never_observe"]["energy_j"]
                                       if group[mode]["energy_j"] is not None and group["never_observe"]["energy_j"] is not None
                                       else None),
                })
            for policy in ("fixed", "never_observe", "selected_observe", "always_observe",
                           "bandit", "adaptive"):
                paired_vs_ordinary.append({
                    "prompt_id": prompt["id"], "repeat": repeat, "policy": policy,
                    "latency_delta_s": group[policy]["latency_s"] - group["ordinary"]["latency_s"],
                    "energy_delta_j": (group[policy]["energy_j"] - group["ordinary"]["energy_j"]
                                       if group[policy]["energy_j"] is not None and group["ordinary"]["energy_j"] is not None
                                       else None),
                })
    # Feasibility checks use only executed held-out policies. They are
    # descriptive, not a counterfactual oracle for unexecuted configurations.
    objective_key = "energy_j" if objective == "energy" else "latency_s"
    fixed_candidates = ("ordinary", "fixed", "never_observe")
    winner_by_prompt = {}
    for prompt in heldout:
        means = {}
        for policy in fixed_candidates:
            group = [r for r in rows if r["split"] == "test" and
                     r["prompt_id"] == prompt["id"] and r["policy"] == policy]
            means[policy] = statistics.median(r[objective_key] / r["generated_tokens"] for r in group)
        winner_by_prompt[prompt["id"]] = min(means, key=means.get)
    observation_overhead = {}
    for mode in ("selected_observe", "always_observe"):
        selected = [r for r in paired_observer_deltas if r["mode"] == mode]
        deltas = [r["energy_delta_j" if objective == "energy" else "latency_delta_s"]
                  for r in selected]
        observation_overhead[mode] = {"mean_delta_per_request": statistics.mean(deltas),
                                      "median_delta_per_request": statistics.median(deltas),
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
        "feasibility": feasibility,
        "model": MODEL, "revision": revision, "precision": "fp32",
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
        "dataset": dataset_info, "arguments": vars(args),
        "selector": selector.summary(), "bandit": bandit.summary(),
        "aggregates": aggregates, "rows": rows,
        "interpretation": "Measured implementation check; small samples do not establish energy savings, superiority, or novelty.",
        "limits": [
            "The implementation uses an attributed LayerSkip inference helper in hardware_aware_selective_feedback.",
            "Request-level adaptation, not token-level adaptation.",
            "Observer overhead uses repeated paired calibration; thermal/order effects remain.",
            "Fixed depth 4 and no-observation depth 8 are separate predetermined baselines.",
            "No optimized serving backend; counter scope may omit other hardware components.",
            "HumanEval supplies prompt diversity; generated code is not executed or scored.",
        ],
    }
    destination = ROOT / "hardware_aware_selective_feedback" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    output_file = destination / "latest.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# LayerSkip selective-feedback experiment",
        "",
        f"Model: {MODEL} at {revision}; fp32 on {model.device}.",
        f"Objective: {objective}. Energy boundary: {report['energy_boundary'] or 'unavailable'}.",
        f"Counter: {meter.source or meter.reason}.",
        f"Observer pairs: {len(pair_prompt_ids)} across {len(set(pair_prompt_ids))} calibration prompts.",
        f"Best executed fixed setting varies across prompts: {feasibility['best_fixed_varies_across_prompts']}.",
        f"Prompts: {len(calibration)} calibration, {len(heldout)} held out; "
        f"{args.repeats} held-out repeat(s), {args.tokens} maximum generated tokens.",
        f"All {len(rows)} executed requests matched native greedy token IDs.",
        "",
        "| Policy | Runs | Latency s/token | Energy J/token |",
        "|---|---:|---:|---:|",
    ]
    for name, result in aggregates.items():
        energy = (f"{result['energy_j_per_token']:.6f}" if result['energy_j_per_token'] is not None
                  else "unavailable")
        lines.append(f"| {name} | {result['runs']} | {result['latency_s_per_token']:.6f} | {energy} |")
    lines += [
        "", "Observed feedback cost per request (selected/all): " +
        ", ".join(f"{name} {value['median_delta_per_request']:.6f} " +
                  ("J" if objective == "energy" else "s")
                  for name, value in observation_overhead.items()) + ".",
        "Observer overhead is measured by same-prompt paired executions. "
        "Raw runs, order, token IDs, valid feedback, and paired deltas are in latest.json.",
        "Calibration work is reported separately and is not allocated to any single baseline.",
        "This is a correctness and feasibility run. Small samples and CPU timing noise "
        "do not establish a speed or energy advantage, and there is no novelty claim.",
    ]
    (destination / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_file), "runs": len(rows), "objective": objective,
                      "greedy_matches": sum(r["greedy_match"] for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
