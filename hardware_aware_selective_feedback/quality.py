"""Task scores for full-prompt runs; HumanEval executes only in constrained Docker.

GSM8K follows the upstream strict #### numeric-answer extraction. HumanEval
assembles prompt + completion + tests + check(entry_point), as upstream does.
One greedy completion per task (repeat 0) is scored; timing repeats are not
independent samples. This module never executes generated Python on the host.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
MARKER = "HUMANEVAL_TESTS_COMPLETED"
WRAPPER = """import contextlib, io, sys, signal
program = sys.stdin.read()
def time_expired(*args):
    raise TimeoutError()
signal.signal(signal.SIGALRM, time_expired)
try:
    signal.setitimer(signal.ITIMER_REAL, float(sys.argv[1]))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(program, '<humaneval>', 'exec'), {})
except TimeoutError:
    sys.exit(124)
except BaseException:
    sys.exit(1)
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
print('HUMANEVAL_TESTS_COMPLETED')
"""


def extract_answer(text):
    match = ANSWER.search(text)
    return match.group(1).strip().replace(",", "") if match else None


def docker_check(task, completion, image, timeout=3.0):
    """Run one test in a Linux container without host mounts or network access."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("HumanEval timeout must be positive and finite")
    if not re.fullmatch(r"[A-Za-z_]\w*", task["entry_point"]):
        raise ValueError("Invalid HumanEval entry point")
    prompt_text = task.get("prompt") or task.get("text", "")
    program = prompt_text + completion + "\n" + task["test"]
    program += f"\ncheck({task['entry_point']})\n"
    name = "energy-humaneval-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=256m",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        "-i",
        image,
        "python",
        "-I",
        "-c",
        WRAPPER,
        str(timeout),
    ]
    try:
        result = subprocess.run(
            command,
            input=program,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 5,
        )
        if result.returncode in (125, 126, 127):
            raise RuntimeError("Docker evaluation infrastructure failed: " + result.stderr[:500])
        passed = result.returncode == 0 and result.stdout.strip() == MARKER
        return {
            "passed": passed,
            "result": "passed" if passed else "timed_out" if result.returncode == 124 else "failed",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "result": "timed_out", "exit_code": None}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)


def summarize(rows, tasks, checker=None):
    """Score repeat 0, retaining all failures and refusing duplicate task/policy rows."""
    by_id = {t["id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task IDs")
    groups, seen, cache = defaultdict(list), set(), {}
    for row in rows:
        if row["split"] != "test" or row["repeat"] != 0:
            continue
        key = row["policy"], row["prompt_id"]
        if key in seen:
            raise ValueError(f"Duplicate task/policy completion: {key}")
        seen.add(key)
        task = by_id[row["prompt_id"]]
        completion = row["completion"]
        detail = {
            "prompt_id": row["prompt_id"],
            "dataset": task["dataset"],
            "greedy_match": row["greedy_match"],
            "hit_token_limit": row.get("hit_token_limit"),
        }
        if task["dataset"] == "gsm8k":
            expected = extract_answer(task["answer"])
            if expected is None:
                raise ValueError("GSM8K reference has no strict #### answer")
            predicted = extract_answer(completion)
            detail.update(predicted=predicted, expected=expected, passed=predicted == expected)
        elif task["dataset"] == "humaneval":
            cache_key = task["id"], completion
            if checker is not None and cache_key not in cache:
                cache[cache_key] = checker(task, completion)
            detail.update(cache.get(cache_key, {"passed": None, "result": "not_executed"}))
        else:
            raise ValueError("Unknown task dataset")
        groups[row["policy"]].append(detail)
    if not groups:
        raise ValueError("No repeat-0 held-out completions available for quality scoring")
    output = {}
    for policy, details in groups.items():
        if {d["prompt_id"] for d in details} != set(by_id):
            raise ValueError(f"Incomplete quality task coverage for {policy}")
        entry = {"details": details}
        for dataset, metric in (("gsm8k", "accuracy"), ("humaneval", "pass@1")):
            subset = [d for d in details if d["dataset"] == dataset]
            evaluated = [d for d in subset if d["passed"] is not None]
            complete = bool(subset) and len(evaluated) == len(subset)
            entry[dataset] = {
                "tasks": len(subset),
                "evaluated": len(evaluated),
                "passed": sum(d["passed"] for d in evaluated),
                metric: sum(d["passed"] for d in evaluated) / len(subset) if complete else None,
            }
        output[policy] = entry
    return {
        "sampling": "one greedy completion per held-out task, repeat 0 only",
        "gsm8k_extraction": "upstream strict #### numeric match; missing marker is incorrect",
        "humaneval_processing": "raw completion, no repair or code-fence stripping",
        "policies": output,
    }


def markdown(summary):
    lines = [
        "",
        "## Task quality (one greedy completion per task)",
        "",
        "Timing repeats are not independent quality samples. Scores use repeat 0, including greedy mismatches.",
        "",
        "| Policy | GSM8K accuracy | GSM8K tasks | HumanEval pass@1 | HumanEval tasks |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, data in summary["policies"].items():
        gsm, code = data["gsm8k"], data["humaneval"]
        accuracy = f"{gsm['accuracy']:.4f}" if gsm["accuracy"] is not None else "not evaluated"
        passed = f"{code['pass@1']:.4f}" if code["pass@1"] is not None else "not executed"
        lines.append(f"| {policy} | {accuracy} | {gsm['tasks']} | {passed} | {code['tasks']} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--docker-image", required=True, help="Locally available Linux Python image"
    )
    parser.add_argument("--timeout", type=float, default=3)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if (
        args.output.resolve() == args.report.resolve()
        or args.output.exists()
        or args.output.with_suffix(".md").exists()
    ):
        parser.error("Choose unused output paths; source reports are never overwritten")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not report.get("arguments", {}).get("task_quality"):
        parser.error(
            "Requires a full-prompt --task-quality run; historical truncated prompts are not scored"
        )
    inspection = subprocess.run(
        ["docker", "image", "inspect", args.docker_image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Use the immutable image ID rather than a mutable tag for every task.
    image_id = inspection.stdout.strip()
    summary = summarize(
        report["rows"],
        report["quality_tasks"],
        lambda task, completion: docker_check(task, completion, image_id, args.timeout),
    )
    if set(summary["policies"]) != set(report["aggregates"]):
        parser.error("Quality policy coverage differs from the experiment aggregates")
    summary.update(
        source_report=str(args.report),
        source_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        image=image_id,
        timeout_s=args.timeout,
        precision=report["precision"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text("\n".join(markdown(summary)) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "policies": list(summary["policies"])}))


if __name__ == "__main__":
    main()
