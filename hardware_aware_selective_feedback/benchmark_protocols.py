"""Pinned benchmark protocols, reference extraction filters, and failure diagnostics.

Phase P1: Pinned GSM8K 5-shot/0-shot protocols with strict and flexible extraction,
plus HumanEval execution with fine-grained failure categorization (passed, assertion_error,
syntax_error, runtime_error, timed_out, infrastructure_error).
"""

import math
import re
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Strict GSM8K regex: matches "#### <number>"
STRICT_ANSWER_RE = re.compile(r"####\s*(\-?[0-9\.\,]+)")

# Flexible number finder: finds candidate numbers with optional negatives, decimals, commas
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

# lm-eval style phrasing patterns for flexible extraction
ANSWER_PHRASES = [
    re.compile(
        r"(?:the\s+answer\s+is\s+|answer:\s*|equals\s*|is\s+)(\-?[0-9\.\,]+)", re.IGNORECASE
    ),
    re.compile(r"=\s*(\-?[0-9\.\,]+)"),
]

HUMANEVAL_SUCCESS_MARKER = "HUMANEVAL_TESTS_COMPLETED"

# Refined docker wrapper that distinguishes SyntaxError, AssertionError, RuntimeError, and TimeoutError
CONTAINER_WRAPPER = """import contextlib, io, sys, signal

program = sys.stdin.read()

def time_expired(*args):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, time_expired)

try:
    signal.setitimer(signal.ITIMER_REAL, float(sys.argv[1]))
    # First check syntax
    try:
        compiled = compile(program, '<humaneval>', 'exec')
    except SyntaxError as e:
        print(f"SYNTAX_ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compiled, {})
except TimeoutError:
    sys.exit(124)
except AssertionError as e:
    print(f"ASSERTION_ERROR: {e}", file=sys.stderr)
    sys.exit(3)
except BaseException as e:
    print(f"RUNTIME_ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(4)
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)

print('HUMANEVAL_TESTS_COMPLETED')
"""


def normalize_numeric_string(raw: str) -> Optional[str]:
    """Normalize numeric strings: strip whitespace, currency symbols, and commas.

    Normalizes integers like '42.0' to '42' if integer-equivalent.
    """
    if raw is None:
        return None
    cleaned = raw.strip().replace(",", "").rstrip(".")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        # If integer equivalent, format as integer
        if math.isfinite(val) and val.is_integer() and "." in cleaned:
            return str(int(val))
        return cleaned
    except ValueError:
        return None


def extract_strict_gsm8k(text: str) -> Optional[str]:
    """Strict GSM8K extraction: requires '#### <numeric>' marker.

    Returns None if marker is absent or value is non-numeric.
    """
    if not text:
        return None
    match = STRICT_ANSWER_RE.search(text)
    if not match:
        return None
    return normalize_numeric_string(match.group(1))


def extract_flexible_gsm8k(text: str) -> Optional[str]:
    """Flexible GSM8K extraction (aligned with upstream lm-evaluation-harness flexible filter).

    1. Checks for #### marker first.
    2. If absent, checks common 'The answer is X' or '= X' phrases.
    3. If absent, takes the last valid numeric token in the solution, ignoring subsequent 'Question:' prompts.
    """
    if not text:
        return None

    # Truncate any repeated 'Question:' sequence if present
    q_index = text.find("\nQuestion:")
    if q_index != -1:
        text = text[:q_index]

    # 1. Strict #### check
    strict = extract_strict_gsm8k(text)
    if strict is not None:
        return strict

    # 2. Phrasing check
    for phrase_re in ANSWER_PHRASES:
        matches = phrase_re.findall(text)
        if matches:
            norm = normalize_numeric_string(matches[-1])
            if norm is not None:
                return norm

    # 3. Last number in text
    numbers = NUMBER_RE.findall(text)
    if numbers:
        for num in reversed(numbers):
            norm = normalize_numeric_string(num)
            if norm is not None:
                return norm

    return None


@dataclass
class CodeEvaluationResult:
    passed: bool
    category: (
        str  # passed, assertion_error, syntax_error, runtime_error, timed_out, infrastructure_error
    )
    exit_code: Optional[int]
    error_message: Optional[str] = None


def run_isolated_docker_check(
    task: Dict[str, Any],
    completion: str,
    image: str = "python:3.12-slim",
    timeout: float = 3.0,
) -> CodeEvaluationResult:
    """Run generated code in a secure, isolated Docker container with zero network and read-only filesystem."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("HumanEval timeout must be positive and finite")
    if not re.fullmatch(r"[A-Za-z_]\w*", task.get("entry_point", "")):
        raise ValueError(f"Invalid HumanEval entry point: {task.get('entry_point')}")

    prompt_text = task.get("prompt") or task.get("text", "")
    program = prompt_text + completion + "\n" + task["test"]
    program += f"\ncheck({task['entry_point']})\n"
    container_name = "energy-humaneval-" + uuid.uuid4().hex

    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
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
        CONTAINER_WRAPPER,
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
    except FileNotFoundError:
        return CodeEvaluationResult(
            passed=False,
            category="infrastructure_error",
            exit_code=None,
            error_message="Docker binary not found on host",
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)
        return CodeEvaluationResult(
            passed=False,
            category="timed_out",
            exit_code=124,
            error_message=f"Container process timed out after {timeout}s",
        )
    except Exception as e:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)
        return CodeEvaluationResult(
            passed=False,
            category="infrastructure_error",
            exit_code=None,
            error_message=f"Docker invocation error: {e}",
        )

    # Classify exit codes:
    # 0: Passed
    # 2: SyntaxError
    # 3: AssertionError
    # 4: RuntimeError
    # 124: Timeout
    # 125, 126, 127: Docker infrastructure failure
    if result.returncode in (125, 126, 127):
        return CodeEvaluationResult(
            passed=False,
            category="infrastructure_error",
            exit_code=result.returncode,
            error_message="Docker engine infrastructure error: " + result.stderr[:300],
        )

    if result.returncode == 0 and HUMANEVAL_SUCCESS_MARKER in result.stdout:
        return CodeEvaluationResult(passed=True, category="passed", exit_code=0)
    elif result.returncode == 2:
        return CodeEvaluationResult(
            passed=False,
            category="syntax_error",
            exit_code=2,
            error_message=result.stderr.strip()[:300],
        )
    elif result.returncode == 3:
        return CodeEvaluationResult(
            passed=False,
            category="assertion_error",
            exit_code=3,
            error_message=result.stderr.strip()[:300],
        )
    elif result.returncode == 124:
        return CodeEvaluationResult(
            passed=False,
            category="timed_out",
            exit_code=124,
            error_message="Timeout expired during execution",
        )
    elif result.returncode == 4:
        return CodeEvaluationResult(
            passed=False,
            category="runtime_error",
            exit_code=4,
            error_message=result.stderr.strip()[:300],
        )
    else:
        return CodeEvaluationResult(
            passed=False,
            category="assertion_error" if "AssertionError" in result.stderr else "runtime_error",
            exit_code=result.returncode,
            error_message=result.stderr.strip()[:300],
        )


# ==============================================================================
# Comprehensive Scorer Fixtures
# ==============================================================================

MATH_SCORER_FIXTURES: List[Dict[str, Any]] = [
    {
        "id": "correct_integer",
        "text": "The calculation gives #### 42",
        "expected_strict": "42",
        "expected_flexible": "42",
    },
    {
        "id": "correct_comma",
        "text": "Total is 1,234 dollars\n#### 1,234",
        "expected_strict": "1234",
        "expected_flexible": "1234",
    },
    {
        "id": "correct_negative",
        "text": "Net loss is negative.\n#### -15",
        "expected_strict": "-15",
        "expected_flexible": "-15",
    },
    {
        "id": "correct_decimal",
        "text": "Result is #### 3.1415",
        "expected_strict": "3.1415",
        "expected_flexible": "3.1415",
    },
    {
        "id": "integer_decimal_normalization",
        "text": "Value is #### 50.0",
        "expected_strict": "50",
        "expected_flexible": "50",
    },
    {
        "id": "absent_marker_flexible_phrasing",
        "text": "After calculations, the answer is 76.",
        "expected_strict": None,
        "expected_flexible": "76",
    },
    {
        "id": "absent_marker_last_number",
        "text": "First we had 10 apples. Then we bought 20 more. Total is 30.",
        "expected_strict": None,
        "expected_flexible": "30",
    },
    {
        "id": "multiple_numbers_with_marker",
        "text": "Step 1 gives 10. Step 2 gives 20.\n#### 30",
        "expected_strict": "30",
        "expected_flexible": "30",
    },
    {
        "id": "repeated_question_trailing",
        "text": "We find 15.\n#### 15\nQuestion: A car travels 50 miles. How many...",
        "expected_strict": "15",
        "expected_flexible": "15",
    },
    {
        "id": "no_number_at_all",
        "text": "I do not know how to solve this problem.",
        "expected_strict": None,
        "expected_flexible": None,
    },
]

CODE_SCORER_FIXTURES: List[Dict[str, Any]] = [
    {
        "id": "correct_solution",
        "task": {
            "prompt": "def add(a, b):\n",
            "test": "def check(f): assert f(2, 3) == 5\n",
            "entry_point": "add",
        },
        "completion": "    return a + b\n",
        "expected_category": "passed",
        "expected_passed": True,
    },
    {
        "id": "wrong_logic",
        "task": {
            "prompt": "def add(a, b):\n",
            "test": "def check(f): assert f(2, 3) == 5\n",
            "entry_point": "add",
        },
        "completion": "    return a - b\n",
        "expected_category": "assertion_error",
        "expected_passed": False,
    },
    {
        "id": "syntax_error",
        "task": {
            "prompt": "def add(a, b):\n",
            "test": "def check(f): assert f(2, 3) == 5\n",
            "entry_point": "add",
        },
        "completion": "    return a +++ \n",
        "expected_category": "syntax_error",
        "expected_passed": False,
    },
    {
        "id": "runtime_error_undefined_var",
        "task": {
            "prompt": "def add(a, b):\n",
            "test": "def check(f): assert f(2, 3) == 5\n",
            "entry_point": "add",
        },
        "completion": "    return undefined_variable\n",
        "expected_category": "runtime_error",
        "expected_passed": False,
    },
    {
        "id": "infinite_loop_timeout",
        "task": {
            "prompt": "def loop(n):\n",
            "test": "def check(f): assert f(1) == 0\n",
            "entry_point": "loop",
        },
        "completion": "    while True:\n        pass\n",
        "expected_category": "timed_out",
        "expected_passed": False,
    },
]
