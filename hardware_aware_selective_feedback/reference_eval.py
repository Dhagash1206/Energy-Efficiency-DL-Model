"""Authoritative reference evaluation, independent parity checks, and failure aggregation.

Phase P1: Provides structured evaluation of saved or live model outputs against
pinned reference protocols, scoring both strict and flexible metrics, and
exporting detailed per-item failure diagnostics.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from hardware_aware_selective_feedback.benchmark_protocols import (
    CodeEvaluationResult,
    extract_flexible_gsm8k,
    extract_strict_gsm8k,
)


def evaluate_completion(
    task: Dict[str, Any],
    completion: str,
    code_checker: Optional[Callable[[Dict[str, Any], str], CodeEvaluationResult]] = None,
) -> Dict[str, Any]:
    """Evaluate a single completion against gold reference standards without leaking gold answers into generation."""
    dataset = task.get("dataset")
    record: Dict[str, Any] = {
        "task_id": task["id"],
        "dataset": dataset,
        "completion": completion,
    }

    if dataset == "gsm8k":
        expected_raw = task.get("answer", "")
        expected = extract_strict_gsm8k(expected_raw)
        strict_pred = extract_strict_gsm8k(completion)
        flex_pred = extract_flexible_gsm8k(completion)

        record.update(
            {
                "expected": expected,
                "strict_pred": strict_pred,
                "flexible_pred": flex_pred,
                "strict_passed": bool(expected is not None and strict_pred == expected),
                "flexible_passed": bool(expected is not None and flex_pred == expected),
                "failure_reason": (
                    None
                    if (strict_pred == expected)
                    else "absent_marker"
                    if strict_pred is None and flex_pred == expected
                    else "wrong_answer"
                    if strict_pred is not None
                    else "no_number_found"
                ),
            }
        )

    elif dataset == "humaneval":
        if code_checker is not None:
            eval_res = code_checker(task, completion)
            record.update(
                {
                    "passed": eval_res.passed,
                    "category": eval_res.category,
                    "exit_code": eval_res.exit_code,
                    "error_message": eval_res.error_message,
                }
            )
        else:
            record.update(
                {
                    "passed": None,
                    "category": "not_executed",
                    "exit_code": None,
                    "error_message": None,
                }
            )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    return record


def summarize_reference_evaluation(
    rows: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    code_checker: Optional[Callable[[Dict[str, Any], str], CodeEvaluationResult]] = None,
) -> Dict[str, Any]:
    """Aggregate per-policy evaluation results across all tasks."""
    task_map = {t["id"]: t for t in tasks}
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    code_eval_cache: Dict[Tuple[str, str], CodeEvaluationResult] = {}

    for row in rows:
        if row.get("split") == "test" and row.get("repeat", 0) == 0:
            policy = row["policy"]
            task = task_map.get(row.get("prompt_id"))
            if not task:
                continue
            completion = row.get("completion", "")

            # Cached code check helper to avoid redundant docker invocations for identical completions
            def cached_checker(t, c):
                key = (t["id"], c)
                if key not in code_eval_cache:
                    code_eval_cache[key] = code_checker(t, c)
                return code_eval_cache[key]

            eval_func = cached_checker if code_checker else None
            record = evaluate_completion(task, completion, eval_func)
            record["greedy_match"] = row.get("greedy_match", True)
            record["hit_token_limit"] = row.get("hit_token_limit", False)
            by_policy[policy].append(record)

    summary: Dict[str, Any] = {
        "evaluation_contract": {
            "sampling": "repeat 0 deterministic greedy completions only",
            "gsm8k": "strict (####) and flexible extraction reported side-by-side",
            "humaneval": "isolated container execution with fine-grained error categories",
        },
        "policies": {},
    }

    for policy, records in by_policy.items():
        gsm_records = [r for r in records if r["dataset"] == "gsm8k"]
        he_records = [r for r in records if r["dataset"] == "humaneval"]

        policy_summary: Dict[str, Any] = {
            "total_tasks": len(records),
            "gsm8k": {
                "tasks": len(gsm_records),
                "strict_passed": sum(1 for r in gsm_records if r.get("strict_passed")),
                "strict_accuracy": (
                    sum(1 for r in gsm_records if r.get("strict_passed")) / len(gsm_records)
                    if gsm_records
                    else None
                ),
                "flexible_passed": sum(1 for r in gsm_records if r.get("flexible_passed")),
                "flexible_accuracy": (
                    sum(1 for r in gsm_records if r.get("flexible_passed")) / len(gsm_records)
                    if gsm_records
                    else None
                ),
                "failure_breakdown": {
                    "wrong_answer": sum(
                        1 for r in gsm_records if r.get("failure_reason") == "wrong_answer"
                    ),
                    "absent_marker": sum(
                        1 for r in gsm_records if r.get("failure_reason") == "absent_marker"
                    ),
                    "no_number_found": sum(
                        1 for r in gsm_records if r.get("failure_reason") == "no_number_found"
                    ),
                },
            },
            "humaneval": {
                "tasks": len(he_records),
                "passed": sum(1 for r in he_records if r.get("passed") is True),
                "pass@1": (
                    sum(1 for r in he_records if r.get("passed") is True) / len(he_records)
                    if he_records and all(r.get("passed") is not None for r in he_records)
                    else None
                ),
                "category_counts": {
                    "passed": sum(1 for r in he_records if r.get("category") == "passed"),
                    "assertion_error": sum(
                        1 for r in he_records if r.get("category") == "assertion_error"
                    ),
                    "syntax_error": sum(
                        1 for r in he_records if r.get("category") == "syntax_error"
                    ),
                    "runtime_error": sum(
                        1 for r in he_records if r.get("category") == "runtime_error"
                    ),
                    "timed_out": sum(1 for r in he_records if r.get("category") == "timed_out"),
                    "infrastructure_error": sum(
                        1 for r in he_records if r.get("category") == "infrastructure_error"
                    ),
                    "not_executed": sum(
                        1 for r in he_records if r.get("category") == "not_executed"
                    ),
                },
            },
            "records": records,
        }
        summary["policies"][policy] = policy_summary

    return summary
