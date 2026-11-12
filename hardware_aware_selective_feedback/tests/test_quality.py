import subprocess
import unittest
from unittest.mock import patch

from hardware_aware_selective_feedback.quality import docker_check, extract_answer, summarize


class QualityTests(unittest.TestCase):
    def test_strict_gsm8k_answer_extraction(self):
        self.assertEqual(extract_answer("work\n#### -1,234.5"), "-1234.5")
        self.assertIsNone(extract_answer("the answer is 42"))

    def test_repeat_zero_only_missing_answers_fail_and_greedy_mismatches_are_scored(self):
        task = {"id": "gsm", "dataset": "gsm8k", "answer": "work\n#### 42"}
        row = {
            "split": "test",
            "repeat": 0,
            "policy": "fixed",
            "prompt_id": "gsm",
            "completion": "42",
            "greedy_match": False,
        }
        result = summarize([row, {**row, "repeat": 1, "completion": "#### 42"}], [task])
        score = result["policies"]["fixed"]["gsm8k"]
        self.assertEqual(score["accuracy"], 0)
        self.assertEqual(score["tasks"], 1)
        with self.assertRaises(ValueError):
            summarize([row, row], [task])

    def test_humaneval_is_pending_without_execution_and_identical_outputs_share_checks(self):
        task = {"id": "code", "dataset": "humaneval"}
        row = {
            "split": "test",
            "repeat": 0,
            "policy": "ordinary",
            "prompt_id": "code",
            "completion": "pass",
            "greedy_match": True,
        }
        rows = [row, {**row, "policy": "adaptive"}]
        self.assertIsNone(summarize(rows, [task])["policies"]["ordinary"]["humaneval"]["pass@1"])
        calls = []

        def check(task, completion):
            calls.append(completion)
            return {"passed": True, "result": "passed"}

        result = summarize(rows, [task], check)
        self.assertEqual(calls, ["pass"])
        self.assertEqual(result["policies"]["adaptive"]["humaneval"]["pass@1"], 1)

    def test_docker_command_isolated_and_timeout_cleans_container(self):
        task = {"prompt": "def f():\n", "test": "def check(f): assert f()==1", "entry_point": "f"}
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[1] == "run":
                self.assertIn("--network=none", command)
                self.assertIn("--read-only", command)
                self.assertIn("--cap-drop=ALL", command)
                self.assertNotIn("-v", command)
                self.assertIn("check(f)", kwargs["input"])
                raise subprocess.TimeoutExpired(command, 1)
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "hardware_aware_selective_feedback.quality.subprocess.run", side_effect=fake_run
        ):
            result = docker_check(task, "    return 1\n", "sha256:test", 1)
        self.assertEqual(result["result"], "timed_out")
        self.assertEqual(commands[-1][1:3], ["rm", "-f"])

    def test_infrastructure_failure_is_not_a_wrong_answer(self):
        task = {"prompt": "", "test": "", "entry_point": "f"}
        with patch(
            "hardware_aware_selective_feedback.quality.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 125, "", "daemon unavailable"),
                subprocess.CompletedProcess([], 0, "", ""),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "infrastructure"):
                docker_check(task, "", "sha256:test")


if __name__ == "__main__":
    unittest.main()
