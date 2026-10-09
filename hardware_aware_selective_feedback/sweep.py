"""Measure fixed, observer-free decoding settings on paired held-out prompts."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from .measurement import Meter
from .run import load_model, load_prompts, row, run_one
from .engine import native_greedy
from .selector import Action


def summarize(rows, actions):
    """Compare complete paired runs; invalid baselines invalidate comparisons."""
    grouped = defaultdict(dict)
    for record in rows:
        key = (record["depth"], record["draft_length"])
        pair = (record["prompt_id"], record["repeat"])
        if pair in grouped[key]:
            raise ValueError(f"Duplicate request for setting {key}, pair {pair}")
        if record["generated_tokens"] < 1 or record["energy_j"] is None:
            raise ValueError("Each request needs generated tokens and measured energy")
        grouped[key][pair] = record
    ordinary = grouped.get((0, 0), {})
    if not ordinary:
        raise ValueError("Ordinary baseline measurements are required")
    baseline_valid = all(r["greedy_match"] for r in ordinary.values())
    summary = []
    for action in actions:
        records = grouped[action.key]
        if records.keys() != ordinary.keys():
            raise ValueError(f"Incomplete paired measurements for {action.key}")
        group = list(records.values())
        tokens = sum(r["generated_tokens"] for r in group)
        paired = [r["energy_j"] - ordinary[r["prompt_id"], r["repeat"]]["energy_j"]
                  for r in group]
        summary.append({
            "depth": action.depth, "draft_length": action.length,
            "requests": len(group), "energy_j_per_token": sum(r["energy_j"] for r in group) / tokens,
            "latency_s_per_token": sum(r["latency_s"] for r in group) / tokens,
            "median_paired_energy_delta_j": statistics.median(paired),
            "mean_paired_energy_delta_j": statistics.mean(paired),
            "greedy_matches": all(r["greedy_match"] for r in group),
            "valid_comparison": baseline_valid and all(r["greedy_match"] for r in group),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--test", type=int, default=4, help="held-out prompts per dataset")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.test, args.tokens, args.repeats, *args.depths, *args.lengths) < 1:
        parser.error("counts, depths and lengths must be positive")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 requires CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    actions = [Action()] + [Action(d, k) for d in args.depths for k in args.lengths]
    if len({a.key for a in actions}) != len(actions):
        parser.error("depths and lengths must be unique")
    with Meter(args.device) as meter:
        if not meter.available:
            parser.error("Energy counter unavailable: " + str(meter.reason))
        run_sweep(args, meter)


def run_sweep(args, meter):
    """Execute and report a sweep using a caller-owned measurement resource."""
    if not meter.available:
        raise ValueError("A working energy counter is required")
    actions = [Action()] + [Action(d, k) for d in args.depths for k in args.lengths]
    torch.set_num_threads(4)
    _, heldout, dataset_info = load_prompts(1, args.test, args.seed)
    model, tokenizer, revision = load_model(args.device, args.precision, attn="auto")
    if any(a.depth >= len(model.model.layers) for a in actions):
        raise ValueError("Draft depth must be below the model's layer count")
    eos = (tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ()
    for prompt in heldout:
        prompt["ids"] = tokenizer.encode(prompt.pop("text"), add_special_tokens=True)[:192]
    rng = random.Random(args.seed + 1)
    rows = []
    mismatches = []
    for index, prompt in enumerate(heldout):
        reference = native_greedy(model, prompt["ids"], args.tokens, eos)
        for repeat in range(args.repeats):
            order = actions.copy()
            rng.shuffle(order)
            for position, action in enumerate(order):
                output, measured = run_one(model, meter, prompt["ids"], args.tokens,
                                           action, eos)
                matched = output["tokens"] == reference
                if not matched:
                    mismatch = next((i for i, (actual, expected) in
                                     enumerate(zip(output["tokens"], reference))
                                     if actual != expected), min(len(output["tokens"]), len(reference)))
                    mismatches.append({"prompt_id": prompt["id"], "repeat": repeat,
                                       "depth": action.depth, "draft_length": action.length,
                                       "token_index": mismatch,
                                       "actual": output["tokens"][mismatch:mismatch+1],
                                       "expected": reference[mismatch:mismatch+1]})
                record = row(prompt, "fixed_sweep", action, output, measured,
                             "test", repeat, position)
                record["greedy_match"] = matched
                rows.append(record)
        print(f"Completed prompt {index + 1}/{len(heldout)}", flush=True)
    summary = summarize(rows, actions)
    report = {"model_revision": revision, "device": str(model.device),
              "precision": args.precision, "energy_boundary": meter.scope,
              "measurement": meter.describe(), "dataset": dataset_info,
              "arguments": {**vars(args), "output": str(args.output)},
              "summary": summary, "mismatches": mismatches, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Fixed decoding configuration sweep", "",
             f"{len(heldout)} held-out prompts; {args.repeats} repeats; {args.tokens} maximum tokens.",
             f"Energy boundary: {meter.scope}. {len(rows) - len(mismatches)}/{len(rows)} requests matched greedy tokens.", "",
             "| Depth | Draft length | Energy J/token | Latency s/token | Median paired energy delta J/request | Greedy matches |",
             "|---:|---:|---:|---:|---:|---:|"]
    for item in sorted(summary, key=lambda x: x["energy_j_per_token"]):
        lines.append(f"| {item['depth']} | {item['draft_length']} | "
                     f"{item['energy_j_per_token']:.6f} | {item['latency_s_per_token']:.6f} | "
                     f"{item['median_paired_energy_delta_j']:+.6f} | {item['greedy_matches']} |")
    lines += ["", "Only comparisons with a correct ordinary baseline and no candidate mismatches are eligible.",
              "Exploratory sweep: confirm any promising setting on fresh prompts and repeats."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    valid = [item for item in summary if item["valid_comparison"]]
    print(json.dumps({"result": str(args.output), "runs": len(rows),
                      "mismatches": len(mismatches),
                      "best_valid": min(valid, key=lambda x: x["energy_j_per_token"]) if valid else None}))


if __name__ == "__main__":
    main()
