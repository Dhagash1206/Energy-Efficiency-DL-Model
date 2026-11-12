"""Calibration-inclusive energy projections from matched saved requests.

Curves extrapolate observed mean request costs to a stationary workload. They
are not observed lifetime measurements, confidence intervals, or end-to-end
deployment costs. All shared calibration and idle sampling is charged once
to the candidate; fixed baselines receive no calibration charge.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .del_comparison import compare_policies


def energy_components(rows):
    result = {}
    tokens = sum(r["generated_tokens"] for r in rows)
    for metric in ("static_idle_energy_j", "dynamic_energy_j"):
        values = [r.get(metric) for r in rows]
        total = sum(values) if values and all(v is not None for v in values) else None
        result[metric] = total
        result[metric + "_per_token"] = total / tokens if total is not None and tokens else None
    return result


def break_even(report, candidate="adaptive", baseline="ordinary"):
    rows = report["rows"]
    matched = compare_policies(rows, candidate, baseline)
    result = {
        "candidate": candidate,
        "baseline": baseline,
        "valid": matched["valid"],
        "invalid_reason": matched["invalid_reason"],
        "n_breakeven": None,
    }
    if not matched["valid"]:
        return result
    groups = {
        name: [r for r in rows if r["split"] == "test" and r["policy"] == name]
        for name in (candidate, baseline)
    }
    calibration = [r for r in rows if r["split"] == "calibration"]
    used = calibration + groups[candidate] + groups[baseline]
    if not calibration or any(r.get("energy_j") is None for r in used):
        result.update(valid=False, invalid_reason="missing calibration or measured energy")
        return result
    boundaries = {r.get("energy_boundary", report.get("energy_boundary")) for r in used}
    if len(boundaries) != 1 or boundaries.intersection({None, "unavailable"}):
        result.update(valid=False, invalid_reason="missing or mixed energy measurement boundaries")
        return result
    idle_calibration = report.get("measurement", {}).get("idle_calibration") or {}
    calibration_j = sum(r["energy_j"] for r in calibration) + idle_calibration.get(
        "total_energy_j", 0
    )
    candidate_j = sum(r["energy_j"] for r in groups[candidate]) / len(groups[candidate])
    baseline_j = sum(r["energy_j"] for r in groups[baseline]) / len(groups[baseline])
    saving = baseline_j - candidate_j
    result.update(
        calibration_energy_j=calibration_j,
        candidate_mean_energy_j=candidate_j,
        baseline_mean_energy_j=baseline_j,
        saving_per_request_j=saving,
        measured_requests_per_policy=len(groups[candidate]),
        n_breakeven=max(0, math.ceil(calibration_j / saving)) if saving > 0 else None,
        no_crossing_reason="candidate mean request energy is not lower" if saving <= 0 else None,
        energy_boundary=next(iter(boundaries)),
        formula="ceil(C / (E_baseline - E_candidate)), when savings are positive",
        accounting="all shared calibration plus idle samples charged to candidate; baseline calibration=0",
        limitation="stationary-workload projection; loading, reference generation, and decisions outside Meter excluded",
    )
    return result


def comparisons(report):
    policies = report["aggregates"]
    return {
        name: break_even(report, baseline=name)
        for name in ("ordinary", "fixed", "never_observe", "fixed_d12_k3", "fixed_depth12")
        if name in policies
    }


def markdown(entries):
    lines = [
        "",
        "## Calibration-inclusive energy break-even",
        "",
        "Stationary-workload projections, not measured lifetime savings. All shared calibration",
        "and idle sampling is charged to adaptive; fixed baselines have zero calibration charge.",
        "Model loading, greedy reference generation, and decisions outside Meter remain excluded.",
        "",
        "| Baseline | Exact-output energy comparison | Break-even requests |",
        "|---|---|---:|",
    ]
    for name, entry in entries.items():
        reason = entry.get("invalid_reason") or entry.get("no_crossing_reason") or "valid"
        crossing = (
            str(entry["n_breakeven"])
            if entry["n_breakeven"] is not None
            else "no supported crossing"
        )
        lines.append(f"| {name} | {reason} | {crossing} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=500)
    args = parser.parse_args()
    if args.max_requests < 1:
        parser.error("max-requests must be positive")
    targets = {ext: Path(str(args.output_prefix) + ext) for ext in (".json", ".md", ".png", ".pdf")}
    if any(p.exists() for p in targets.values()) or args.report.resolve() in {
        p.resolve() for p in targets.values()
    }:
        parser.error("Choose unused output paths; existing results are never overwritten")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    entries = comparisons(report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    # Bound plotting memory even when projecting to a very large request count.
    requests = sorted({round(args.max_requests * i / 1000) for i in range(1001)})
    valid = [e for e in entries.values() if e["valid"]]
    if valid:
        sample = valid[0]
        axes[0].plot(
            requests,
            [
                sample["calibration_energy_j"] + n * sample["candidate_mean_energy_j"]
                for n in requests
            ],
            label="adaptive + all calibration",
            color="black",
            linewidth=2,
        )
    for entry in valid:
        name = entry["baseline"]
        axes[0].plot(requests, [n * entry["baseline_mean_energy_j"] for n in requests], label=name)
        crossing = entry["n_breakeven"]
        label = f"{name} (N*={crossing})" if crossing is not None else f"{name} (no crossing)"
        (line,) = axes[1].plot(
            requests,
            [n * entry["saving_per_request_j"] - entry["calibration_energy_j"] for n in requests],
            label=label,
        )
        if crossing is not None and crossing <= args.max_requests:
            axes[1].axvline(crossing, color=line.get_color(), linestyle=":", alpha=0.4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        ylabel="Cumulative GPU energy (J)"
        if report.get("energy_boundary") == "nvidia_gpu"
        else "Cumulative scoped energy (J)",
        title="Projected cumulative cost",
    )
    axes[1].set(
        ylabel="Baseline minus adaptive incl. calibration (J)", title="Projected net savings"
    )
    for ax in axes:
        ax.set_xlabel("Requests from the same workload distribution")
        ax.grid(alpha=0.2)
        if valid:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid energy comparison", transform=ax.transAxes, ha="center")
    fig.suptitle(
        "Measured mean-cost extrapolation; excluded overhead is not amortized", fontsize=10
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[".png"], dpi=200)
    fig.savefig(targets[".pdf"])
    plt.close(fig)
    summary = {
        "source_report": str(args.report),
        "source_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "max_requests": args.max_requests,
        "comparisons": entries,
    }
    targets[".json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    targets[".md"].write_text("\n".join(markdown(entries)) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in targets.values()]}))


if __name__ == "__main__":
    main()
