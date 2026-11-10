"""Matched, exact-output comparisons. Invalid/incomplete pairs never win."""

import statistics


def compare_policies(rows, candidate, baseline):
    groups = {name: {} for name in (candidate, baseline)}
    for row in rows:
        name = row["policy"]
        if row["split"] != "test" or name not in groups:
            continue
        key = row["prompt_id"], row["repeat"]
        if key in groups[name]:
            raise ValueError(f"Duplicate {name} pair {key}")
        groups[name][key] = row
    left, right = groups[candidate], groups[baseline]
    keys = sorted(left.keys() | right.keys())
    reason = None
    if not keys or left.keys() != right.keys():
        reason = "missing matched requests"
    elif any(
        not left[k]["greedy_match"]
        or not right[k]["greedy_match"]
        or left[k]["tokens"] != right[k]["tokens"]
        or left[k]["generated_tokens"] <= 0
        or left[k]["generated_tokens"] != len(left[k]["tokens"])
        or right[k]["generated_tokens"] != len(right[k]["tokens"])
        for k in keys
    ):
        reason = "greedy mismatch or inconsistent generated tokens"
    result = dict(
        candidate=candidate,
        baseline=baseline,
        valid=reason is None,
        invalid_reason=reason,
        pairs=len(keys),
        metrics={},
    )
    if reason:
        return result
    for metric in ("energy_j", "latency_s"):
        if any(left[k][metric] is None or right[k][metric] is None for k in keys):
            result["metrics"][metric] = None
            continue
        total_left = sum(left[k][metric] for k in keys)
        total_right = sum(right[k][metric] for k in keys)
        deltas = {}
        for key in keys:
            count = left[key]["generated_tokens"]
            deltas.setdefault(key[0], []).append((left[key][metric] - right[key][metric]) / count)
        # Prompt is the unit of analysis; repeats are not independent prompts.
        prompt_deltas = {p: statistics.mean(values) for p, values in deltas.items()}
        result["metrics"][metric] = {
            "candidate_reduction_percent": 100 * (1 - total_left / total_right)
            if total_right > 0
            else None,
            "mean_prompt_delta_per_token": statistics.mean(prompt_deltas.values()),
            "prompt_deltas_per_token": prompt_deltas,
            "candidate_better_prompts": sum(v < 0 for v in prompt_deltas.values()),
            "prompts": len(prompt_deltas),
        }
    return result


def comparison_markdown(comparisons):
    lines = [
        "",
        "## DEL comparison (independent greedy paper port)",
        "",
        "This is a local implementation comparison, not a reproduction of published DEL speedups.",
        "DEL's shadow projections and decisions are measured. Adaptive request-level decisions/updates",
        "remain outside the generation meter; shared calibration is separate. These are not end-to-end savings.",
        "Positive reduction means adaptive used less cost. No significance claim is made.",
        "",
        "| Comparison | Exact-output valid | GPU energy reduction | Latency reduction |",
        "|---|---|---:|---:|",
    ]
    for baseline, result in comparisons.items():
        values = []
        for metric in ("energy_j", "latency_s"):
            entry = result["metrics"].get(metric)
            number = entry["candidate_reduction_percent"] if entry else None
            values.append(f"{number:.2f}%" if number is not None else "unavailable")
        lines.append(f"| adaptive vs {baseline} | {result['valid']} | {values[0]} | {values[1]} |")
        if not result["valid"]:
            lines.append(f"\nComparison invalid: {result['invalid_reason']}.\n")
    return lines
