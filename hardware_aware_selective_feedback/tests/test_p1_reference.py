import unittest
from unittest.mock import patch

from hardware_aware_selective_feedback.benchmark_protocols import (
    CODE_SCORER_FIXTURES,
    MATH_SCORER_FIXTURES,
    CodeEvaluationResult,
    extract_flexible_gsm8k,
    extract_strict_gsm8k,
    run_isolated_docker_check,
)
from hardware_aware_selective_feedback.reference_eval import (
    evaluate_completion,
    summarize_reference_evaluation,
)


class TestPhaseP1ReferenceEvaluation(unittest.TestCase):
    def test_math_extraction_fixtures(self):
        for fixture in MATH_SCORER_FIXTURES:
            with self.subTest(fixture=fixture["id"]):
                strict = extract_strict_gsm8k(fixture["text"])
                flex = extract_flexible_gsm8k(fixture["text"])
                self.assertEqual(
                    strict,
                    fixture["expected_strict"],
                    f"Strict extraction failed for {fixture['id']}",
                )
                self.assertEqual(
                    flex,
                    fixture["expected_flexible"],
                    f"Flexible extraction failed for {fixture['id']}",
                )

    def test_code_evaluation_fixtures_mocked_execution(self):
        for fixture in CODE_SCORER_FIXTURES:
            with self.subTest(fixture=fixture["id"]):
                # Create mock behavior corresponding to fixture expectation
                def mock_docker_run(cmd, input, **kwargs):
                    category = fixture["expected_category"]
                    if category == "passed":
                        return unittest.mock.Mock(
                            returncode=0, stdout="HUMANEVAL_TESTS_COMPLETED\n", stderr=""
                        )
                    elif category == "syntax_error":
                        return unittest.mock.Mock(
                            returncode=2, stdout="", stderr="SYNTAX_ERROR: invalid syntax"
                        )
                    elif category == "assertion_error":
                        return unittest.mock.Mock(
                            returncode=3, stdout="", stderr="ASSERTION_ERROR: assert 2 == 5"
                        )
                    elif category == "timed_out":
                        return unittest.mock.Mock(
                            returncode=124, stdout="", stderr="Timeout expired"
                        )
                    elif category == "runtime_error":
                        return unittest.mock.Mock(
                            returncode=4, stdout="", stderr="RUNTIME_ERROR: NameError: undefined"
                        )
                    raise ValueError(f"Unhandled category {category}")

                with patch("subprocess.run", side_effect=mock_docker_run):
                    res = run_isolated_docker_check(fixture["task"], fixture["completion"])
                    self.assertEqual(res.category, fixture["expected_category"])
                    self.assertEqual(res.passed, fixture["expected_passed"])

    def test_docker_infrastructure_failure_handling(self):
        task = {"prompt": "def f():\n", "test": "assert f() == 1\n", "entry_point": "f"}

        # Simulate docker daemon failure (code 125)
        with patch(
            "subprocess.run",
            return_value=unittest.mock.Mock(
                returncode=125, stdout="", stderr="docker: daemon is down"
            ),
        ):
            res = run_isolated_docker_check(task, "    return 1\n")
            self.assertFalse(res.passed)
            self.assertEqual(res.category, "infrastructure_error")

        # Simulate missing docker binary
        with patch("subprocess.run", side_effect=FileNotFoundError("docker not found")):
            res = run_isolated_docker_check(task, "    return 1\n")
            self.assertFalse(res.passed)
            self.assertEqual(res.category, "infrastructure_error")

    def test_evaluate_completion_diagnostics(self):
        # GSM8K: absent marker diagnostic
        gsm_task = {"id": "gsm-1", "dataset": "gsm8k", "answer": "Solution.\n#### 76"}
        rec = evaluate_completion(gsm_task, "The answer is 76.")
        self.assertFalse(rec["strict_passed"])
        self.assertTrue(rec["flexible_passed"])
        self.assertEqual(rec["failure_reason"], "absent_marker")

        # GSM8K: wrong answer
        rec_wrong = evaluate_completion(gsm_task, "Calculated #### 15")
        self.assertFalse(rec_wrong["strict_passed"])
        self.assertFalse(rec_wrong["flexible_passed"])
        self.assertEqual(rec_wrong["failure_reason"], "wrong_answer")

    def test_summarize_reference_evaluation(self):
        tasks = [
            {"id": "gsm-1", "dataset": "gsm8k", "answer": "#### 42"},
            {"id": "gsm-2", "dataset": "gsm8k", "answer": "#### 10"},
            {
                "id": "he-1",
                "dataset": "humaneval",
                "prompt": "def f():\n",
                "test": "",
                "entry_point": "f",
            },
        ]
        rows = [
            {
                "split": "test",
                "repeat": 0,
                "policy": "fixed",
                "prompt_id": "gsm-1",
                "completion": "The answer is 42.",
            },
            {
                "split": "test",
                "repeat": 0,
                "policy": "fixed",
                "prompt_id": "gsm-2",
                "completion": "#### 10",
            },
            {
                "split": "test",
                "repeat": 0,
                "policy": "fixed",
                "prompt_id": "he-1",
                "completion": "return 1",
            },
        ]

        def dummy_checker(task, completion):
            return CodeEvaluationResult(passed=True, category="passed", exit_code=0)

        summary = summarize_reference_evaluation(rows, tasks, code_checker=dummy_checker)
        self.assertIn("fixed", summary["policies"])
        fixed_sum = summary["policies"]["fixed"]

        # GSM8K: 1 strict passed, 2 flexible passed
        self.assertEqual(fixed_sum["gsm8k"]["strict_passed"], 1)
        self.assertEqual(fixed_sum["gsm8k"]["strict_accuracy"], 0.5)
        self.assertEqual(fixed_sum["gsm8k"]["flexible_passed"], 2)
        self.assertEqual(fixed_sum["gsm8k"]["flexible_accuracy"], 1.0)
        self.assertEqual(fixed_sum["gsm8k"]["failure_breakdown"]["absent_marker"], 1)

        # HumanEval: 1 passed
        self.assertEqual(fixed_sum["humaneval"]["pass@1"], 1.0)
        self.assertEqual(fixed_sum["humaneval"]["category_counts"]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
