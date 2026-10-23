"""Fixed-configuration sweep: measure all (depth, draft) pairs on paired held-out prompts.

Each prompt is run with every configured action in randomised order.  The
ordinary greedy baseline is included automatically as the reference.  All
comparisons are paired (same prompt, same repeat) to cancel prompt-level
variance.  A comparison is only marked valid when both the greedy baseline
and the candidate produce token-identical output to native_greedy.

Usage:
    python -m hardware_aware_selective_feedback.sweep \\
        --device cuda --precision fp16 \\
        --test 4 --tokens 100 --repeats 2 \\
        --depths 4 --lengths 1 \\
        --output results/my_sweep.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from .engine import batched_greedy, native_greedy
from .gpu_clocks import pinned_gpu
from .measurement import Meter
from .run import MAX_PROMPT_TOKENS, build_graph_drafters, load_model, load_prompts, row, run_one
from .selector import Action


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarize(rows: list[dict], actions: list[Action]) -> list[dict]:
    """Aggregate paired measurements into per-action summary statistics.

    Raises ValueError if any prompt/repeat pair is missing or duplicated, or
    if any row lacks generated tokens or energy.

    A valid_comparison entry is True only when:
      - the ordinary greedy baseline matched native_greedy for all pairs, AND
      - the candidate itself matched native_greedy for all pairs.
    """
    # Group rows by action key and (prompt_id, repeat) pair.
    grouped: dict[tuple, dict] = defaultdict(dict)
    for record in rows:
        key  = (record["depth"], record["draft_length"])
        pair = (record["prompt_id"], record["repeat"])
        if pair in grouped[key]:
            raise ValueError(f"Duplicate record for action={key}, pair={pair}.")
        if record["generated_tokens"] < 1 or record["energy_j"] is None:
            raise ValueError("Every record must have generated tokens and measured energy.")
        grouped[key][pair] = record

    ordinary = grouped.get((0, 0), {})
    if not ordinary:
        raise ValueError("Ordinary greedy baseline (depth=0, length=0) is required.")
    baseline_valid = all(r["greedy_match"] for r in ordinary.values())

    summary = []
    for action in actions:
        records = grouped[action.key]
        if records.keys() != ordinary.keys():
            raise ValueError(
                f"Incomplete paired measurements for action={action.key}: "
                f"expected {set(ordinary.keys())}, got {set(records.keys())}."
            )
        group  = list(records.values())
        tokens = sum(r["generated_tokens"] for r in group)
        paired = [
            r["energy_j"] - ordinary[r["prompt_id"], r["repeat"]]["energy_j"]
            for r in group
        ]
        summary.append({
            "depth":                       action.depth,
            "draft_length":                action.length,
            "requests":                    len(group),
            "energy_j_per_token":          sum(r["energy_j"] for r in group) / tokens,
            "latency_s_per_token":         sum(r["latency_s"] for r in group) / tokens,
            "median_paired_energy_delta_j": statistics.median(paired),
            "mean_paired_energy_delta_j":   statistics.mean(paired),
            "greedy_matches":              all(r["greedy_match"] for r in group),
            "valid_comparison":            baseline_valid and all(r["greedy_match"] for r in group),
        })
    return summary


def measure_batched(model, meter, prompts: list[dict], references: dict, batch_size: int,
                    token_limit: int, eos: tuple, repeats: int, pad_id: int = 0) -> dict:
    """Measure batched ordinary decoding as a baseline against per-request decoding.

    Prompts are processed in consecutive groups of ``batch_size`` (the last
    group may be smaller). Energy and latency are per batch, so only totals
    per generated token are comparable with the per-request sweep rows.
    """
    batches = []
    for repeat in range(repeats):
        for start in range(0, len(prompts), batch_size):
            group = prompts[start:start + batch_size]
            outputs, measured = meter.run(lambda group=group: batched_greedy(
                model, [p["ids"] for p in group], token_limit, eos, pad_id))
            mismatched = [p["id"] for p, out in zip(group, outputs, strict=True)
                          if out != references[p["id"]]]
            batches.append({
                "repeat": repeat, "prompt_ids": [p["id"] for p in group],
                "generated_tokens": sum(len(out) for out in outputs),
                "latency_s": measured["latency_s"], "energy_j": measured["energy_j"],
                "mismatched_prompt_ids": mismatched,
            })
    tokens = sum(b["generated_tokens"] for b in batches)
    energy_known = all(b["energy_j"] is not None for b in batches)
    return {
        "batch_size": batch_size,
        "batches": len(batches),
        "generated_tokens": tokens,
        "energy_j_per_token": (sum(b["energy_j"] for b in batches) / tokens
                               if energy_known and tokens else None),
        "latency_s_per_token": sum(b["latency_s"] for b in batches) / tokens if tokens else None,
        "greedy_matches": not any(b["mismatched_prompt_ids"] for b in batches),
        "records": batches,
    }


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace, meter: Meter) -> None:
    """Run the sweep and write JSON + Markdown reports to ``args.output``."""
    if not meter.available:
        raise ValueError("A working energy counter is required for the sweep.")

    actions = [Action()] + [Action(d, k) for d in args.depths for k in args.lengths]
    torch.set_num_threads(4)

    _, heldout, dataset_info = load_prompts(1, args.test, args.seed)
    model, tokenizer, revision = load_model(args.device, args.precision, attn="auto")

    if any(a.depth >= len(model.model.layers) for a in actions):
        raise ValueError("All draft depths must be strictly below the model's layer count.")

    eos = (tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ()
    for prompt in heldout:
        prompt["ids"] = tokenizer.encode(prompt.pop("text"), add_special_tokens=True)[:MAX_PROMPT_TOKENS]
    drafters = (build_graph_drafters(model, args.depths, args.tokens, max(args.lengths))
                if args.cuda_graph_drafts else None)

    rng  = random.Random(args.seed + 1)
    rows: list[dict]  = []
    mismatches: list[dict] = []
    references: dict[str, list[int]] = {}

    for index, prompt in enumerate(heldout):
        reference = native_greedy(model, prompt["ids"], args.tokens, eos)
        references[prompt["id"]] = reference

        for repeat in range(args.repeats):
            order = actions.copy()
            rng.shuffle(order)

            for position, action in enumerate(order):
                output, measured = run_one(model, meter, prompt["ids"], args.tokens,
                                           action, eos, adaptive_drafts=args.adaptive_drafts,
                                           graph_drafters=drafters)
                matched = output["tokens"] == reference

                if not matched:
                    # Find the first diverging position.
                    diff_pos = next(
                        (i for i, (a, e) in enumerate(zip(output["tokens"], reference)) if a != e),
                        min(len(output["tokens"]), len(reference)),
                    )
                    mismatches.append({
                        "prompt_id":    prompt["id"],
                        "repeat":       repeat,
                        "depth":        action.depth,
                        "draft_length": action.length,
                        "token_index":  diff_pos,
                        "actual":       output["tokens"][diff_pos : diff_pos + 1],
                        "expected":     reference[diff_pos : diff_pos + 1],
                    })

                record = row(prompt, "fixed_sweep", action, output, measured,
                             "test", repeat, position)
                record["greedy_match"] = matched
                rows.append(record)

        print(f"Completed prompt {index + 1}/{len(heldout)}", flush=True)

    pad_id = next((t for t in (tokenizer.pad_token_id, tokenizer.eos_token_id) if t is not None), 0)
    batched = [measure_batched(model, meter, heldout, references, size, args.tokens,
                               eos, args.repeats, pad_id)
               for size in args.batch_sizes]

    # ── Report ────────────────────────────────────────────────────────────
    summary = summarize(rows, actions)
    report  = {
        "model_revision":   revision,
        "device":           str(model.device),
        "precision":        args.precision,
        "energy_boundary":  meter.scope,
        "measurement":      meter.describe(),
        "dataset":          dataset_info,
        "arguments":        {**vars(args), "output": str(args.output)},
        "summary":          summary,
        "mismatches":       mismatches,
        "batched_baseline": batched,
        "rows":             rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown summary table (sorted by energy, best first).
    valid   = [s for s in summary if s["valid_comparison"]]
    lines   = [
        "# Fixed decoding configuration sweep", "",
        f"{len(heldout)} held-out prompts; {args.repeats} repeats; "
        f"{args.tokens} maximum tokens.",
        f"Energy boundary: {meter.scope}.  "
        f"{len(rows) - len(mismatches)}/{len(rows)} requests matched greedy tokens.", "",
        "| Depth | Draft length | Energy J/token | Latency s/token | "
        "Median paired ΔE J | Greedy matches | Valid |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summary, key=lambda x: x["energy_j_per_token"]):
        lines.append(
            f"| {item['depth']} | {item['draft_length']} "
            f"| {item['energy_j_per_token']:.6f} "
            f"| {item['latency_s_per_token']:.6f} "
            f"| {item['median_paired_energy_delta_j']:+.4f} "
            f"| {item['greedy_matches']} "
            f"| {item['valid_comparison']} |"
        )
    if batched:
        lines += [
            "", "## Batched ordinary decoding baseline", "",
            "| Batch size | Energy J/token | Latency s/token | Greedy matches |",
            "|---:|---:|---:|---:|",
        ]
        for item in batched:
            energy = (f"{item['energy_j_per_token']:.6f}"
                      if item["energy_j_per_token"] is not None else "unavailable")
            lines.append(f"| {item['batch_size']} | {energy} "
                         f"| {item['latency_s_per_token']:.6f} | {item['greedy_matches']} |")
    lines += [
        "",
        "A comparison is valid only when both the greedy baseline and the candidate "
        "match native_greedy on all pairs.",
        "Confirm any promising setting on fresh prompts and repeats before reporting.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = min(valid, key=lambda x: x["energy_j_per_token"]) if valid else None
    print(json.dumps({
        "result":     str(args.output),
        "runs":       len(rows),
        "mismatches": len(mismatches),
        "best_valid": best,
        "batched": [{k: v for k, v in b.items() if k != "records"} for b in batched],
    }))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",    choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--test",      type=int, default=4,
                        help="Held-out prompts per dataset.")
    parser.add_argument("--tokens",    type=int, default=16,
                        help="Maximum generated tokens per prompt.")
    parser.add_argument("--repeats",   type=int, default=2,
                        help="Number of repeated runs per (prompt, action) pair.")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--depths",    type=int, nargs="+", default=[4, 8, 12],
                        help="Early-exit layer depths to sweep.")
    parser.add_argument("--lengths",   type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Draft token lengths to sweep.")
    parser.add_argument("--output",    type=Path, required=True,
                        help="Path for the JSON results file (.md report alongside).")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[],
                        help="Also measure batched ordinary decoding at these batch sizes, "
                             "e.g. --batch-sizes 2 4.")
    parser.add_argument("--cuda-graph-drafts", action="store_true",
                        help="PROTOTYPE: replay draft steps as CUDA Graphs (CUDA only).")
    parser.add_argument("--adaptive-drafts", action="store_true",
                        help="Shorten drafts below 50%% acceptance; decode ordinarily below 30%%.")
    parser.add_argument("--gpu-clock-mhz", type=int, default=None,
                        help="Lock the NVIDIA graphics clock during the sweep (admin/root).")
    parser.add_argument("--gpu-power-limit-w", type=float, default=None,
                        help="Cap NVIDIA board power during the sweep (admin/root).")
    args = parser.parse_args()

    if min(args.test, args.tokens, args.repeats, *args.depths, *args.lengths,
           *args.batch_sizes) < 1:
        parser.error("All counts, depths, and lengths must be positive integers.")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 requires --device cuda.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable on this machine.")
    if (args.gpu_clock_mhz is not None or args.gpu_power_limit_w is not None) and args.device != "cuda":
        parser.error("--gpu-clock-mhz and --gpu-power-limit-w require --device cuda.")
    if args.cuda_graph_drafts and args.device != "cuda":
        parser.error("--cuda-graph-drafts requires --device cuda.")

    actions = [Action()] + [Action(d, k) for d in args.depths for k in args.lengths]
    if len({a.key for a in actions}) != len(actions):
        parser.error("(depth, length) pairs must be unique.")

    with Meter(args.device) as meter:
        if not meter.available:
            parser.error("Energy counter unavailable: " + str(meter.reason))
        with pinned_gpu(args.gpu_clock_mhz, args.gpu_power_limit_w):
            run_sweep(args, meter)


if __name__ == "__main__":
    main()
