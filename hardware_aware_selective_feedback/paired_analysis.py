"""Prompt-clustered and stream-level paired bootstrap analysis.

Phase P5 / P7: Evaluates paired differences (baseline - candidate) using true
cluster-level resampling. By resampling prompt or stream clusters with replacement
(rather than individual request rows), repeats within a prompt remain coupled,
preventing artificially narrow confidence intervals on correlated workloads.
"""

import json
from pathlib import Path
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BootstrapResult:
    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    strictly_positive: bool
    strictly_negative: bool
    crosses_zero: bool
    num_bootstrap_samples: int
    num_clusters: int
    total_observations: int


def bootstrap_clustered_paired_difference(
    records: List[Dict[str, Any]],
    candidate_key: str = "candidate_value",
    baseline_key: str = "baseline_value",
    cluster_key: str = "prompt_id",
    confidence_level: float = 0.95,
    num_samples: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Compute true cluster-level bootstrap confidence interval for paired differences.

    Groups observations by cluster_key (e.g. prompt_id or stream_id), resamples clusters
    with replacement, and computes the paired mean difference (baseline - candidate).
    """
    if not records:
        raise ValueError("Cannot perform bootstrap on empty records")

    clusters: Dict[Any, List[Tuple[float, float]]] = defaultdict(list)
    all_diffs = []
    for r in records:
        b_val = r[baseline_key]
        c_val = r[candidate_key]
        c_id = r[cluster_key]
        clusters[c_id].append((b_val, c_val))
        all_diffs.append(b_val - c_val)

    cluster_ids = list(clusters.keys())
    num_clusters = len(cluster_ids)
    if num_clusters == 0:
        raise ValueError("No valid clusters found")

    observed_mean = sum(all_diffs) / len(all_diffs)

    rng = random.Random(seed)
    bootstrap_means: List[float] = []

    for _ in range(num_samples):
        resampled_cluster_ids = [
            cluster_ids[rng.randrange(num_clusters)] for _ in range(num_clusters)
        ]
        sample_diffs: List[float] = []
        for cid in resampled_cluster_ids:
            for b_val, c_val in clusters[cid]:
                sample_diffs.append(b_val - c_val)

        if sample_diffs:
            bootstrap_means.append(sum(sample_diffs) / len(sample_diffs))

    bootstrap_means.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * len(bootstrap_means))
    upper_idx = int((1.0 - alpha) * len(bootstrap_means))

    ci_low = bootstrap_means[lower_idx]
    ci_high = bootstrap_means[upper_idx]

    strictly_pos = ci_low > 0
    strictly_neg = ci_high < 0
    crosses_0 = ci_low <= 0 <= ci_high

    return BootstrapResult(
        mean_difference=observed_mean,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=confidence_level,
        strictly_positive=strictly_pos,
        strictly_negative=strictly_neg,
        crosses_zero=crosses_0,
        num_bootstrap_samples=num_samples,
        num_clusters=num_clusters,
        total_observations=len(all_diffs),
    )


def bootstrap_paired_difference(
    baseline_values: List[float],
    candidate_values: List[float],
    cluster_ids: Optional[List[Any]] = None,
    confidence_level: float = 0.95,
    num_samples: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Convenience wrapper for paired values, supporting explicit cluster_ids."""
    if len(baseline_values) != len(candidate_values):
        raise ValueError("Baseline and candidate values must have identical lengths")

    if cluster_ids is None:
        # If no cluster IDs provided, each pair is its own cluster
        cluster_ids = list(range(len(baseline_values)))
    elif len(cluster_ids) != len(baseline_values):
        raise ValueError("cluster_ids must have identical length to values")

    records = [
        {"baseline_value": b, "candidate_value": c, "prompt_id": cid}
        for b, c, cid in zip(baseline_values, candidate_values, cluster_ids)
    ]
    return bootstrap_clustered_paired_difference(
        records,
        candidate_key="candidate_value",
        baseline_key="baseline_value",
        cluster_key="prompt_id",
        confidence_level=confidence_level,
        num_samples=num_samples,
        seed=seed,
    )


def analyze_experiment_paired_differences(
    report: Dict[str, Any],
    comparisons: Optional[List[Tuple[str, str]]] = None,
    metric: str = "energy_j",
    num_samples: int = 2000,
) -> Dict[str, Any]:
    """Perform prompt-clustered paired bootstrap across specified policy pairs."""
    if comparisons is None:
        comparisons = [
            ("fixed", "ordinary"),
            ("adaptive", "ordinary"),
            ("adaptive", "fixed"),
        ]

    rows = report.get("rows", [])
    test_rows = [r for r in rows if r.get("split") == "test"]

    # Index by (policy, prompt_id, repeat)
    table = {}
    for r in test_rows:
        key = (r["policy"], r["prompt_id"], r.get("repeat", 0))
        table[key] = r

    results = {}
    for candidate, baseline in comparisons:
        paired_records = []
        # Find matching (prompt_id, repeat) pairs
        for r in test_rows:
            if r["policy"] == candidate:
                pid = r["prompt_id"]
                rep = r.get("repeat", 0)
                b_row = table.get((baseline, pid, rep))
                if b_row is not None:
                    b_val = b_row.get(metric)
                    c_val = r.get(metric)
                    if b_val is not None and c_val is not None:
                        paired_records.append(
                            {
                                "baseline_value": b_val,
                                "candidate_value": c_val,
                                "prompt_id": pid,
                            }
                        )

        comp_name = f"{candidate}_vs_{baseline}"
        if paired_records:
            boot = bootstrap_clustered_paired_difference(
                paired_records,
                candidate_key="candidate_value",
                baseline_key="baseline_value",
                cluster_key="prompt_id",
                num_samples=num_samples,
            )
            results[comp_name] = asdict(boot)
        else:
            results[comp_name] = {"error": "no_matched_pairs"}

    return {
        "metric": metric,
        "comparisons": results,
    }


def markdown_paired_analysis_report(analysis: Dict[str, Any]) -> str:
    """Generate Markdown report summarizing prompt-clustered bootstrap results."""
    lines = [
        "# Phase P7: Prompt-Clustered Paired Bootstrap Analysis",
        "",
        f"- **Metric:** `{analysis['metric']}` (baseline - candidate, >0 favors candidate)",
        "",
        "| Comparison (Candidate vs Baseline) | Mean Difference | 95% CI Lower | 95% CI Upper | Strictly Positive | Crosses Zero | Clusters | Observations |",
        "|---|---:|---:|---:|:---:|:---:|---:|---:|",
    ]
    for comp_name, res in analysis.get("comparisons", {}).items():
        if "error" in res:
            lines.append(f"| `{comp_name}` | N/A | N/A | N/A | N/A | N/A | 0 | 0 |")
            continue
        cand, base = comp_name.split("_vs_")
        pos_str = (
            "Yes"
            if res["strictly_positive"]
            else ("Negative" if res["strictly_negative"] else "No")
        )
        cz_str = "Yes" if res["crosses_zero"] else "No"
        lines.append(
            f"| `{cand}` vs `{base}` | {res['mean_difference']:+.4f} | {res['ci_lower']:+.4f} | "
            f"{res['ci_upper']:+.4f} | {pos_str} | {cz_str} | {res['num_clusters']} | {res['total_observations']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    import argparse
    from hardware_aware_selective_feedback import run_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "hardware_aware_selective_feedback/results/publication/p5_live_experiment.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "hardware_aware_selective_feedback/results/publication/p7_paired_analysis.json"
        ),
    )
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    run_manifest.assert_unique_output_path(args.output)

    if not args.report.exists():
        parser.error(f"Report file {args.report} does not exist")

    rep_data = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = analyze_experiment_paired_differences(rep_data, num_samples=args.samples)
    analysis["environment_provenance"] = run_manifest.get_environment_provenance()
    analysis["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    md = markdown_paired_analysis_report(analysis)
    args.output.with_suffix(".md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "comparisons_evaluated": len(analysis["comparisons"])}
        )
    )


if __name__ == "__main__":
    main()
