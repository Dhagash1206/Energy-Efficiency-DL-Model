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

from .engine import native_greedy
from .measurement import Meter
from .run import load_model, load_prompts, row, run_one
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
        prompt["ids"] = tokenizer.encode(prompt.pop("text"), add_special_tokens=True)[:192]

    rng  = random.Random(args.seed + 1)
    rows: list[dict]  = []
    mismatches: list[dict] = []

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
    args = parser.parse_args()

    if min(args.test, args.tokens, args.repeats, *args.depths, *args.lengths) < 1:
        parser.error("All counts, depths, and lengths must be positive integers.")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 requires --device cuda.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable on this machine.")

    actions = [Action()] + [Action(d, k) for d in args.depths for k in args.lengths]
    if len({a.key for a in actions}) != len(actions):
        parser.error("(depth, length) pairs must be unique.")

    with Meter(args.device) as meter:
        if not meter.available:
            parser.error("Energy counter unavailable: " + str(meter.reason))
        run_sweep(args, meter)


if __name__ == "__main__":
    main()
