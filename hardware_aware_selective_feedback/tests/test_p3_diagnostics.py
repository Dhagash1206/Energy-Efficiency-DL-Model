import unittest
from unittest.mock import MagicMock, patch

import torch

from hardware_aware_selective_feedback.benchmark_protocols import CodeEvaluationResult
from hardware_aware_selective_feedback.model_diagnostics import (
    classify_task_failure,
    evaluate_diagnostics_on_prompts,
)


class TestPhaseP3Diagnostics(unittest.TestCase):
    def test_classify_gsm8k_failures(self):
        # 1. Correct answer
        passed, cat = classify_task_failure(
            dataset="gsm8k",
            raw_completion="Work #### 42",
            expected_answer="#### 42",
            finish_reason="eos",
            is_repetitive=False,
            hit_token_limit=False,
        )
        self.assertTrue(passed)
        self.assertIsNone(cat)

        # 2. Formatting failure (answer 42 present, but missing #### marker)
        passed, cat = classify_task_failure(
            dataset="gsm8k",
            raw_completion="The answer is 42.",
            expected_answer="#### 42",
            finish_reason="eos",
            is_repetitive=False,
            hit_token_limit=False,
        )
        self.assertFalse(passed)
        self.assertEqual(cat, "formatting_missing_marker")

        # 3. Reasoning failure (wrong answer)
        passed, cat = classify_task_failure(
            dataset="gsm8k",
            raw_completion="#### 99",
            expected_answer="#### 42",
            finish_reason="eos",
            is_repetitive=False,
            hit_token_limit=False,
        )
        self.assertFalse(passed)
        self.assertEqual(cat, "reasoning_wrong_answer")

        # 4. Repetition loop failure
        passed, cat = classify_task_failure(
            dataset="gsm8k",
            raw_completion="repeat repeat repeat",
            expected_answer="#### 42",
            finish_reason="repetition",
            is_repetitive=True,
            hit_token_limit=False,
        )
        self.assertFalse(passed)
        self.assertEqual(cat, "repetition")

        # 5. Truncation failure
        passed, cat = classify_task_failure(
            dataset="gsm8k",
            raw_completion="Incomplete work...",
            expected_answer="#### 42",
            finish_reason="length",
            is_repetitive=False,
            hit_token_limit=True,
        )
        self.assertFalse(passed)
        self.assertEqual(cat, "truncation")

    def test_classify_humaneval_failures(self):
        # Syntax error
        code_res = CodeEvaluationResult(passed=False, category="syntax_error", exit_code=2)
        passed, cat = classify_task_failure(
            dataset="humaneval",
            raw_completion="def f():",
            expected_answer=None,
            finish_reason="eos",
            is_repetitive=False,
            hit_token_limit=False,
            code_eval=code_res,
        )
        self.assertFalse(passed)
        self.assertEqual(cat, "syntax_error")

        # Assertion error
        code_res = CodeEvaluationResult(passed=False, category="assertion_error", exit_code=3)
        passed, cat = classify_task_failure(
            dataset="humaneval",
            raw_completion="return 0",
            expected_answer=None,
            finish_reason="eos",
            is_repetitive=False,
            hit_token_limit=False,
            code_eval=code_res,
        )
        self.assertFalse(passed)
        self.assertEqual(cat, "assertion_error")

    def test_evaluate_diagnostics_on_prompts(self):
        tasks = [
            {"id": "gsm-1", "dataset": "gsm8k", "text": "What is 2+2?", "answer": "#### 4"},
            {
                "id": "he-1",
                "dataset": "humaneval",
                "text": "def f():",
                "test": "",
                "entry_point": "f",
            },
        ]

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 128001
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "The answer is 4."

        model = MagicMock()
        model.device = torch.device("cpu")

        with patch(
            "hardware_aware_selective_feedback.model_diagnostics.native_greedy",
            return_value=[10, 20],
        ):
            diag = evaluate_diagnostics_on_prompts(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                max_tokens=50,
                device="cpu",
                run_docker=False,
            )

            self.assertEqual(diag["total_evaluated"], 2)
            self.assertEqual(diag["gsm8k"]["count"], 1)
            self.assertEqual(diag["humaneval"]["count"], 1)
            self.assertEqual(
                diag["gsm8k"]["failure_categories"].get("formatting_missing_marker"), 1
            )


if __name__ == "__main__":
    unittest.main()
