"""Fit a speculative-config joule profile on the GPU that is running.

The profile is written beside the other results and named with that GPU.
A 4060 profile is not reused as a 3050 profile. Pass --require-3050 on the
teammate machine; on any other GPU the command records that the 3050 sweep
has not been run.

Example:
    python -m hardware_aware_selective_feedback.startup_sweep
    python -m hardware_aware_selective_feedback.startup_sweep --require-3050
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from hardware_aware_selective_feedback.device_energy import (
    ACTIONS,
    DeviceRow,
    fit_device,
    startup_choice,
    step_winners,
    train_head_decision,
)
from hardware_aware_selective_feedback.engine import generate
from hardware_aware_selective_feedback.measurement import Meter
from hardware_aware_selective_feedback.reject_probe import find_snapshot

LIMIT = 64
PROMPT_COUNT = 16
REPEAT_PROMPTS = 2


def gpu_slug(name: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in name)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def generation_counts(result: dict) -> dict:
    accepted = 0
    rejected = 0
    full_drafts = []
    for cycle in result["cycles"]:
        drafted = len(cycle["draft"])
        kept = int(cycle["accepted"])
        if kept < 0 or kept > drafted:
            raise RuntimeError("cycle acceptance is outside the draft")
        accepted += kept
        rejected += drafted - kept
        if drafted == 4:
            full_drafts.append(kept)
    committed = len(result["tokens"])
    verify_steps = 1 + len(result["cycles"])
    if committed < 1 or verify_steps < 1:
        raise RuntimeError("generation produced no committed tokens")
    return {
        "committed": committed,
        "verify_steps": verify_steps,
        "accepted": accepted,
        "rejected": rejected,
        "full_draft_acceptance": full_drafts,
        "tokens": result["tokens"],
    }


def prompts(tokenizer, count: int) -> list[list[int]]:
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))[:count]
    encoded = []
    for row in rows:
        text = "Question: " + row["question"] + "\nAnswer:"
        encoded.append(tokenizer(text, add_special_tokens=True).input_ids)
    if len(encoded) < count:
        raise RuntimeError(f"GSM8K test split has {len(encoded)} rows, need {count}")
    return encoded


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, indent=2))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--require-3050", action="store_true")
    parser.add_argument("--prompts", type=int, default=PROMPT_COUNT)
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args()

    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the device profile")
    gpu_name = torch.cuda.get_device_name(0)
    results = Path("hardware_aware_selective_feedback/results")
    if args.require_3050 and "3050" not in gpu_name:
        status = {
            "status": "not_measured",
            "gpu_visible": gpu_name,
            "required": "NVIDIA GeForce RTX 3050",
            "reason": "The RTX 3050 is not in this machine. The 4060 profile is not a 3050 result.",
            "command": "python -m hardware_aware_selective_feedback.startup_sweep --require-3050",
        }
        write_status(results / "rtx3050-device-profile.json", status)
        raise SystemExit(3)

    snapshot = find_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to(device)
    meter = Meter(device, energy_source="nvml")
    if not meter.available:
        raise SystemExit(meter.reason)
    encoded = prompts(tokenizer, args.prompts)
    generate(model, encoded[0], limit=8, depth=0, drafts=0, eos=())

    measured = []
    rows: list[DeviceRow] = []
    histogram: Counter = Counter()
    for prompt_index, prompt in enumerate(encoded):
        repeats = 2 if prompt_index < REPEAT_PROMPTS else 1
        greedy_tokens = None
        for repeat in range(repeats):
            for depth, length in ACTIONS:
                result, reading = meter.run(
                    lambda prompt=prompt, depth=depth, length=length: generate(
                        model, prompt, limit=args.limit, depth=depth, drafts=length, eos=()
                    )
                )
                counted = generation_counts(result)
                if depth == 0 and length == 0 and greedy_tokens is None:
                    greedy_tokens = counted["tokens"]
                matched = counted["tokens"] == greedy_tokens
                joules = reading["energy_j"]
                if joules is None:
                    raise SystemExit("NVML returned no joules")
                setting = f"p{prompt_index}-d{depth}-k{length}"
                record = {
                    "prompt": prompt_index,
                    "repeat": repeat,
                    "depth": depth,
                    "length": length,
                    "matched": matched,
                    "joules": joules,
                    "latency_s": reading["latency_s"],
                    "committed": counted["committed"],
                    "verify_steps": counted["verify_steps"],
                    "accepted": counted["accepted"],
                    "rejected": counted["rejected"],
                }
                measured.append(record)
                if matched:
                    rows.append(DeviceRow(
                        counted["verify_steps"], counted["accepted"], counted["rejected"], joules, setting
                    ))
                    if length == 4:
                        for kept in counted["full_draft_acceptance"]:
                            histogram[(depth, kept)] += 1
                print(
                    f"{setting} repeat {repeat}: verify {counted['verify_steps']} "
                    f"accepted {counted['accepted']} rejected {counted['rejected']} "
                    f"joules {joules:.3f} match={matched}",
                    flush=True,
                )

    profile = fit_device(rows)
    choice = startup_choice(measured)
    depth = choice["depth"] if choice["depth"] else 4
    accepted_counts = []
    for (cycle_depth, kept), count in histogram.items():
        if cycle_depth == depth:
            accepted_counts.extend([kept] * count)
    if not accepted_counts:
        for (cycle_depth, kept), count in histogram.items():
            if cycle_depth == 4:
                accepted_counts.extend([kept] * count)
                depth = 4
    winners = step_winners(profile, depth, accepted_counts) if accepted_counts else []
    decision = train_head_decision(profile, winners)
    report = {
        "gpu": gpu_name,
        "energy_scope": "nvidia_gpu",
        "model": "facebook/layerskip-llama3.2-1B",
        "limit": args.limit,
        "prompts": args.prompts,
        "startup_choice": choice,
        "decision": decision,
        "acceptance_histogram": {
            f"depth-{cycle_depth}-accepted-{kept}": count
            for (cycle_depth, kept), count in sorted(histogram.items())
        },
        **profile,
        "rows": measured,
    }
    output = results / f"{gpu_slug(gpu_name)}-device-profile.json"
    write_status(output, report)
    print(
        f"choice depth {choice['depth']} length {choice['length']} "
        f"mean {choice['mean_joules']:.3f} J; train_head={decision['train_head']}"
    )
    print(decision["reason"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
