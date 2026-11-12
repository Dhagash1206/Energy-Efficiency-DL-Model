"""Model quality diagnostics and full-depth development evaluation.

Phase P3: Evaluates full-depth baseline capability, separating reasoning errors,
formatting errors, repetition loops, truncation/budget limits, and code execution
failures before decoding optimizations are applied.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from hardware_aware_selective_feedback.benchmark_protocols import (
    CodeEvaluationResult,
    extract_flexible_gsm8k,
    extract_strict_gsm8k,
    run_isolated_docker_check,
)
from hardware_aware_selective_feedback.engine import native_greedy
from hardware_aware_selective_feedback.stopping import (
    StopCriteria,
    detect_repetition,
    evaluate_committed_stopping,
)


@dataclass
class DiagnosticResult:
    task_id: str
    dataset: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    is_repetitive: bool
    hit_token_limit: bool
    passed: bool
    failure_category: Optional[
        str
    ]  # reasoning, formatting, repetition, truncation, syntax, assertion, runtime, timeout
    raw_completion: str
    latency_s: float


def classify_task_failure(
    dataset: str,
    raw_completion: str,
    expected_answer: Optional[str],
    finish_reason: str,
    is_repetitive: bool,
    hit_token_limit: bool,
    code_eval: Optional[CodeEvaluationResult] = None,
) -> Tuple[bool, Optional[str]]:
    """Determine whether task succeeded and diagnose specific failure category."""
    if dataset == "gsm8k":
        expected = extract_strict_gsm8k(expected_answer or "")
        strict_pred = extract_strict_gsm8k(raw_completion)
        flex_pred = extract_flexible_gsm8k(raw_completion)

        if expected is not None and strict_pred == expected:
            return True, None
        if is_repetitive:
            return False, "repetition"
        if hit_token_limit:
            return False, "truncation"
        if expected is not None and flex_pred == expected:
            return False, "formatting_missing_marker"
        return False, "reasoning_wrong_answer"

    elif dataset == "humaneval":
        if code_eval is not None:
            if code_eval.passed:
                return True, None
            if is_repetitive:
                return False, "repetition"
            if hit_token_limit:
                return False, "truncation"
            if code_eval.category == "syntax_error":
                return False, "syntax_error"
            if code_eval.category == "assertion_error":
                return False, "assertion_error"
            if code_eval.category == "timed_out":
                return False, "timeout"
            if code_eval.category == "runtime_error":
                return False, "runtime_error"
            return False, code_eval.category

        if is_repetitive:
            return False, "repetition"
        if hit_token_limit:
            return False, "truncation"
        return False, "unexecuted"

    return False, "unknown"


def evaluate_diagnostics_on_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: List[Dict[str, Any]],
    max_tokens: int = 256,
    device: str = "cuda",
    docker_image: str = "python:3.12-slim",
    run_docker: bool = False,
) -> Dict[str, Any]:
    """Run full-depth greedy generation on development tasks to diagnose failure causes."""
    eos_id = tokenizer.eos_token_id or 128001
    criteria = StopCriteria(eos_token_ids=(eos_id,), max_tokens=max_tokens)

    results: List[DiagnosticResult] = []
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()

    for task in tasks:
        prompt_text = task["text"]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        t0 = time.perf_counter()
        generated_ids = native_greedy(model, prompt_ids, limit=max_tokens, eos=(eos_id,))
        t1 = time.perf_counter()

        stop_decision = evaluate_committed_stopping(
            generated_ids, criteria, tokenizer=tokenizer, is_committed=True
        )
        is_rep, _ = detect_repetition(generated_ids, ngram_size=16, threshold=3)
        hit_limit = len(generated_ids) >= max_tokens and (
            not generated_ids or generated_ids[-1] != eos_id
        )

        completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        code_eval = None
        if task["dataset"] == "humaneval" and run_docker:
            code_eval = run_isolated_docker_check(
                task, completion_text, image=docker_image, timeout=3.0
            )

        passed, cat = classify_task_failure(
            dataset=task["dataset"],
            raw_completion=completion_text,
            expected_answer=task.get("answer"),
            finish_reason=stop_decision.finish_reason or ("length" if hit_limit else "eos"),
            is_repetitive=is_rep,
            hit_token_limit=hit_limit,
            code_eval=code_eval,
        )

        results.append(
            DiagnosticResult(
                task_id=task["id"],
                dataset=task["dataset"],
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(generated_ids),
                finish_reason=stop_decision.finish_reason or ("length" if hit_limit else "eos"),
                is_repetitive=is_rep,
                hit_token_limit=hit_limit,
                passed=passed,
                failure_category=cat,
                raw_completion=completion_text,
                latency_s=t1 - t0,
            )
        )

    t_end = time.perf_counter()

    peak_mem_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if (device == "cuda" and torch.cuda.is_available())
        else 0
    )

    gsm_results = [r for r in results if r.dataset == "gsm8k"]
    he_results = [r for r in results if r.dataset == "humaneval"]

    summary = {
        "total_evaluated": len(results),
        "total_wall_time_s": t_end - t_start,
        "peak_gpu_memory_mb": peak_mem_mb,
        "gsm8k": {
            "count": len(gsm_results),
            "passed": sum(1 for r in gsm_results if r.passed),
            "repetition_count": sum(1 for r in gsm_results if r.is_repetitive),
            "truncation_count": sum(1 for r in gsm_results if r.hit_token_limit),
            "failure_categories": {
                cat: sum(1 for r in gsm_results if r.failure_category == cat)
                for cat in set(r.failure_category for r in gsm_results if r.failure_category)
            },
        },
        "humaneval": {
            "count": len(he_results),
            "passed": sum(1 for r in he_results if r.passed),
            "repetition_count": sum(1 for r in he_results if r.is_repetitive),
            "truncation_count": sum(1 for r in he_results if r.hit_token_limit),
            "failure_categories": {
                cat: sum(1 for r in he_results if r.failure_category == cat)
                for cat in set(r.failure_category for r in he_results if r.failure_category)
            },
        },
        "items": [asdict(r) for r in results],
    }

    return summary


def markdown_summary(summary: Dict[str, Any], precision: str = "fp16", device: str = "cuda") -> str:
    """Generate Markdown diagnostic report for Phase P3 model decision gate."""
    lines = [
        "# Phase P3: Model Quality Development Diagnostics",
        "",
        f"- **Evaluated Tasks:** {summary['total_evaluated']}",
        f"- **Precision:** {precision} on {device}",
        f"- **Peak GPU Memory:** {summary['peak_gpu_memory_mb']} MiB",
        f"- **Wall Time:** {summary['total_wall_time_s']:.2f}s",
        "",
        "## Summary by Dataset",
        "",
        "| Dataset | Tasks | Passed | Repetition Loops | Truncation (Hit Budget) | Main Failure Mode |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for ds in ("gsm8k", "humaneval"):
        d_info = summary.get(ds, {})
        passed = d_info.get("passed", 0)
        cnt = d_info.get("count", 0)
        rep = d_info.get("repetition_count", 0)
        trunc = d_info.get("truncation_count", 0)
        cats = d_info.get("failure_categories", {})
        top_cat = max(cats.items(), key=lambda x: x[1])[0] if cats else "none"
        lines.append(f"| {ds} | {cnt} | {passed}/{cnt} | {rep} | {trunc} | {top_cat} |")

    lines += [
        "",
        "## Per-Task Failure Diagnostics",
        "",
        "| Task ID | Dataset | Prompt Tokens | Output Tokens | Finish Reason | Repetitive | Diagnosed Failure | Latency (s) |",
        "|---|---|---:|---:|---|:---:|---|---:|",
    ]
    for item in summary.get("items", []):
        rep_str = "Yes" if item.get("is_repetitive") else "No"
        fail_str = item.get("failure_category") or "PASSED"
        lines.append(
            f"| `{item['task_id']}` | {item['dataset']} | {item['prompt_tokens']} | "
            f"{item['completion_tokens']} | `{item['finish_reason']}` | {rep_str} | "
            f"`{fail_str}` | {item['latency_s']:.3f} |"
        )

    lines += [
        "",
        "## P3 Decision Gate Assessment",
        "",
        "- **Capability vs Decoding:** Full-depth baseline answers the tasks directly; any failure here represents checkpoint capability or formatting, not speculative decoding error.",
        "- **Repetition & Truncation:** Unwanted repetition loops and token limit exhaustions are separated from semantic reasoning failures.",
    ]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    import json
    from pathlib import Path
    from hardware_aware_selective_feedback import protocol, run_manifest, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=Path("configs/tiny_dev.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument(
        "--docker", action="store_true", help="Execute HumanEval code in isolated Docker"
    )
    parser.add_argument("--docker-image", default="python:3.12-slim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_manifest.assert_unique_output_path(args.output)

    # Load tasks from split config
    cfg_data = json.loads(args.split_config.read_text(encoding="utf-8"))
    cfg = protocol.SplitConfig(**cfg_data)
    splits = protocol.load_dataset_splits(cfg)
    dev_tasks = splits["test_prompts"]

    model, tokenizer, _ = run.load_model(device=args.device, precision=args.precision)

    summary = evaluate_diagnostics_on_prompts(
        model=model,
        tokenizer=tokenizer,
        tasks=dev_tasks,
        max_tokens=args.tokens,
        device=args.device,
        docker_image=args.docker_image,
        run_docker=args.docker,
    )

    summary["environment_provenance"] = run_manifest.get_environment_provenance()
    summary["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_report = markdown_summary(summary, precision=args.precision, device=args.device)
    args.output.with_suffix(".md").write_text(md_report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evaluated": summary["total_evaluated"]}))


if __name__ == "__main__":
    main()
