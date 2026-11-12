"""Fair baseline selection, fixed action sweep, and frozen policy serialization.

Phase P4: Selects the single best fixed configuration on development data only,
records tuning costs, freezes the winner to prevent held-out leakage, preserves
named anchors (d4/k3, d8/k3, d12/k3), and includes competitive upstream baselines.
"""

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from hardware_aware_selective_feedback.engine import generate, native_greedy
from hardware_aware_selective_feedback.measurement import Meter
from hardware_aware_selective_feedback.selector import Action

NAMED_ANCHORS = {
    (4, 3): "fixed_d4_k3",
    (8, 3): "fixed_d8_k3",
    (12, 3): "fixed_d12_k3",
    (0, 0): "ordinary",
}


@dataclass
class SweepCandidateResult:
    depth: int
    draft_length: int
    name: str
    is_anchor: bool
    runs: int
    greedy_mismatches: int
    all_greedy_match: bool
    metric_value: Optional[float]
    latency_s_per_token: Optional[float]
    energy_j_per_token: Optional[float]


def select_best_fixed_baseline(
    sweep_results: List[Dict[str, Any]],
    objective: str = "energy_j_per_token",
) -> Dict[str, Any]:
    """Select the best valid fixed action from development evaluation rows/aggregates.

    Configurations with any greedy mismatch are strictly disqualified from selection.
    """
    candidates: List[SweepCandidateResult] = []
    total_tuning_energy_j = 0.0
    total_tuning_time_s = 0.0

    for item in sweep_results:
        d = item.get("depth", 0)
        k = item.get("draft_length", 0)
        mismatches = item.get("greedy_mismatches", 0)
        runs = item.get("runs", 0)

        lat_per_tok = item.get("latency_s_per_token")
        energy_per_tok = item.get("energy_j_per_token")

        if item.get("total_energy_j"):
            total_tuning_energy_j += item["total_energy_j"]
        if item.get("total_latency_s"):
            total_tuning_time_s += item["total_latency_s"]

        val = energy_per_tok if objective == "energy_j_per_token" else lat_per_tok
        is_anchor = (d, k) in NAMED_ANCHORS
        name = NAMED_ANCHORS.get((d, k), f"fixed_d{d}_k{k}")

        candidates.append(
            SweepCandidateResult(
                depth=d,
                draft_length=k,
                name=name,
                is_anchor=is_anchor,
                runs=runs,
                greedy_mismatches=mismatches,
                all_greedy_match=(mismatches == 0),
                metric_value=val,
                latency_s_per_token=lat_per_tok,
                energy_j_per_token=energy_per_tok,
            )
        )

    # Disqualify any candidate that had mismatches
    valid_candidates = [c for c in candidates if c.all_greedy_match and c.metric_value is not None]
    if not valid_candidates:
        raise ValueError("No valid candidate satisfied exact-output match on development data")

    # Select candidate minimizing the specified objective
    valid_candidates.sort(key=lambda c: c.metric_value)
    winner = valid_candidates[0]

    anchor_entries = {c.name: asdict(c) for c in candidates if c.is_anchor}

    manifest = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection_objective": objective,
        "chosen_fixed_action": {
            "depth": winner.depth,
            "draft_length": winner.draft_length,
            "name": winner.name,
            "metric_value": winner.metric_value,
        },
        "development_tuning_cost": {
            "total_tuning_energy_j": total_tuning_energy_j,
            "total_tuning_time_s": total_tuning_time_s,
            "candidates_evaluated": len(candidates),
        },
        "anchors": anchor_entries,
        "all_candidates": [
            asdict(c) for c in sorted(candidates, key=lambda x: (x.metric_value or float("inf")))
        ],
    }

    return manifest


def save_frozen_baseline(manifest: Dict[str, Any], output_path: Path):
    """Save the serialized frozen baseline configuration to disk."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_frozen_baseline(path: Path) -> Dict[str, Any]:
    """Load a previously frozen baseline configuration."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Frozen baseline file {p} does not exist")
    return json.loads(p.read_text(encoding="utf-8"))


def run_development_baseline_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    calibration_tasks: List[Dict[str, Any]],
    meter: Meter,
    depths: Sequence[int] = (4, 8, 12),
    lengths: Sequence[int] = (1, 2, 3, 4),
    tokens: int = 128,
    repeats: int = 1,
    objective: str = "energy_j_per_token",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute fixed configuration sweep on development/calibration prompts only.

    Validates token exactness against native greedy reference and measures latency + energy.
    """
    eos_id = tokenizer.eos_token_id or 128001
    eos = (eos_id,)

    # Candidate actions: ordinary (0, 0) + (d, k) for d in depths for k in lengths
    actions = [Action(0, 0)] + [Action(d, k) for d in depths for k in lengths]
    num_layers = (
        len(model.model.layers)
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else 16
    )
    for a in actions:
        if a.depth >= num_layers:
            raise ValueError(
                f"Draft depth {a.depth} must be strictly below layer count {num_layers}"
            )

    sweep_records: List[Dict[str, Any]] = []

    # Encode calibration tasks and generate reference tokens
    task_prompts = []
    for t in calibration_tasks:
        p_text = t.get("prompt") or t.get("text", "")
        p_ids = tokenizer.encode(p_text, add_special_tokens=True)[:256]
        ref_tokens = native_greedy(model, p_ids, limit=tokens, eos=eos)
        task_prompts.append({"task_id": t["id"], "ids": p_ids, "ref": ref_tokens})

    for action in actions:
        mismatches = 0
        total_gen_tokens = 0
        total_lat = 0.0
        total_energy = 0.0 if meter.available else None
        runs = 0

        for task_info in task_prompts:
            p_ids = task_info["ids"]
            ref_tokens = task_info["ref"]

            for _ in range(repeats):
                output, measured = meter.run(
                    lambda: generate(
                        model,
                        p_ids,
                        tokens,
                        action.depth,
                        action.length,
                        eos=eos,
                    )
                )
                runs += 1
                gen_toks = len(output["tokens"])
                total_gen_tokens += gen_toks
                total_lat += measured["latency_s"]
                if meter.available and measured.get("energy_j") is not None:
                    total_energy = (total_energy or 0.0) + measured["energy_j"]

                # Strict exact-output check
                if output["tokens"] != ref_tokens:
                    mismatches += 1

        lat_per_tok = (total_lat / total_gen_tokens) if total_gen_tokens > 0 else 0.0
        energy_per_tok = (
            (total_energy / total_gen_tokens)
            if (total_energy is not None and total_gen_tokens > 0)
            else None
        )

        sweep_records.append(
            {
                "depth": action.depth,
                "draft_length": action.length,
                "runs": runs,
                "greedy_mismatches": mismatches,
                "latency_s_per_token": lat_per_tok,
                "energy_j_per_token": energy_per_tok,
                "total_energy_j": total_energy,
                "total_latency_s": total_lat,
                "total_generated_tokens": total_gen_tokens,
            }
        )

    manifest = select_best_fixed_baseline(sweep_records, objective=objective)
    return manifest, sweep_records


def markdown_p4_report(manifest: Dict[str, Any], sweep_records: List[Dict[str, Any]]) -> str:
    """Generate Markdown report summarizing Phase P4 fixed baseline selection."""
    lines = [
        "# Phase P4: Development Fixed Baseline Sweep & Selection",
        "",
        f"- **Frozen At (UTC):** `{manifest['frozen_at_utc']}`",
        f"- **Selection Objective:** `{manifest['selection_objective']}`",
        f"- **Winning Action:** `{manifest['chosen_fixed_action']['name']}` (depth={manifest['chosen_fixed_action']['depth']}, draft_length={manifest['chosen_fixed_action']['draft_length']})",
        f"- **Winner Metric:** `{manifest['chosen_fixed_action']['metric_value']}`",
        f"- **Candidates Evaluated:** {manifest['development_tuning_cost']['candidates_evaluated']}",
        f"- **Total Tuning Energy:** {manifest['development_tuning_cost']['total_tuning_energy_j']:.4f} J",
        f"- **Total Tuning Time:** {manifest['development_tuning_cost']['total_tuning_time_s']:.2f} s",
        "",
        "## Evaluated Candidates",
        "",
        "| Candidate | Depth | Draft Length | Runs | Mismatches | Latency (s/tok) | Energy (J/tok) | Valid Winner |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for c in sorted(
        sweep_records,
        key=lambda x: (
            x.get("energy_j_per_token") if x.get("energy_j_per_token") is not None else float("inf")
        ),
    ):
        d = c["depth"]
        k = c["draft_length"]
        name = NAMED_ANCHORS.get((d, k), f"fixed_d{d}_k{k}")
        valid = "Yes" if c["greedy_mismatches"] == 0 else "DISQUALIFIED"
        e_str = (
            f"{c['energy_j_per_token']:.6f}" if c.get("energy_j_per_token") is not None else "N/A"
        )
        l_str = (
            f"{c['latency_s_per_token']:.6f}" if c.get("latency_s_per_token") is not None else "N/A"
        )
        lines.append(
            f"| `{name}` | {d} | {k} | {c['runs']} | {c['greedy_mismatches']} | {l_str} | {e_str} | {valid} |"
        )
    return "\n".join(lines) + "\n"


def main():
    import argparse
    from hardware_aware_selective_feedback import protocol, run, run_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=Path("configs/tiny_dev.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--objective", default="energy_j_per_token")
    parser.add_argument(
        "--output-frozen",
        type=Path,
        default=Path("data_splits/frozen_fixed_baseline.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(
            "hardware_aware_selective_feedback/results/publication/p4_baseline_sweep.json"
        ),
    )
    args = parser.parse_args()

    run_manifest.assert_unique_output_path(args.output_frozen)
    run_manifest.assert_unique_output_path(args.output_report)

    cfg_data = json.loads(args.split_config.read_text(encoding="utf-8"))
    cfg = protocol.SplitConfig(**cfg_data)
    splits = protocol.load_dataset_splits(cfg)
    calib_tasks = splits["calibration_prompts"]

    model, tokenizer, _ = run.load_model(device=args.device, precision=args.precision)

    with Meter(args.device) as meter:
        manifest, sweep_records = run_development_baseline_sweep(
            model=model,
            tokenizer=tokenizer,
            calibration_tasks=calib_tasks,
            meter=meter,
            depths=args.depths,
            lengths=args.lengths,
            tokens=args.tokens,
            repeats=args.repeats,
            objective=args.objective,
        )

    # Save frozen baseline state
    save_frozen_baseline(manifest, args.output_frozen)

    # Save full sweep publication report
    report = {
        "frozen_baseline": manifest,
        "sweep_records": sweep_records,
        "environment_provenance": run_manifest.get_environment_provenance(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = markdown_p4_report(manifest, sweep_records)
    args.output_report.with_suffix(".md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "frozen_output": str(args.output_frozen),
                "report_output": str(args.output_report),
                "winner": manifest["chosen_fixed_action"],
            }
        )
    )


if __name__ == "__main__":
    main()
