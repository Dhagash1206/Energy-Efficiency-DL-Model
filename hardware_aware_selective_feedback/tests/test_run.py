import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hardware_aware_selective_feedback import run
from hardware_aware_selective_feedback.engine import generate
from hardware_aware_selective_feedback.run import resolve_dtype


class ModelPrecisionTests(unittest.TestCase):
    def test_fp32_supported_on_cpu_and_cuda(self):
        self.assertIs(resolve_dtype("fp32", "cpu"), torch.float32)
        self.assertIs(resolve_dtype("fp32", "cuda"), torch.float32)

    def test_fp16_supported_on_cuda(self):
        self.assertIs(resolve_dtype("fp16", "cuda"), torch.float16)

    def test_fp16_rejected_on_cpu(self):
        with self.assertRaisesRegex(ValueError, "only on CUDA"):
            resolve_dtype("fp16", "cpu")

    def test_unknown_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported precision"):
            resolve_dtype("bf16", "cuda")


class FakeTokenizer:
    eos_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [ord(character) % 32 for character in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "#### 42"


def corrupt_depth4_length3(model, prompt, limit, depth, drafts, *args, **kwargs):
    """generate() that gives a wrong last token for depth=4, length=3 only."""
    output = generate(model, prompt, limit, depth, drafts, *args, **kwargs)
    if (depth, drafts) == (4, 3):
        output["tokens"][-1] = (output["tokens"][-1] + 1) % 32
    return output


class MismatchHandlingTests(unittest.TestCase):
    """End-to-end run.main on a tiny model with one configuration forced to mismatch."""

    def _run(self, *extra_args):
        torch.manual_seed(3)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=13,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
                attn_implementation="eager",
            )
        ).eval()

        def prompts(prefix):
            return [
                {
                    "id": f"{prefix}-{i}",
                    "dataset": "gsm8k",
                    "text": f"question {prefix} {i}",
                    "answer": "#### 42",
                }
                for i in range(2)
            ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = [
                "run",
                "--device",
                "cpu",
                "--calibration",
                "1",
                "--test",
                "1",
                "--tokens",
                "6",
                "--observer-pairs",
                "1",
                "--energy-source",
                "none",
                "--threads",
                "1",
                "--output",
                str(output),
                *extra_args,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    run, "load_prompts", return_value=(prompts("cal"), prompts("test"), {})
                ),
                patch.object(run, "load_model", return_value=(model, FakeTokenizer(), "rev")),
                patch.object(run, "generate", side_effect=corrupt_depth4_length3),
                patch("builtins.print"),
            ):
                run.main()
            return json.loads(output.read_text()), output.with_suffix(".md").read_text()

    def test_mismatch_is_recorded_and_configuration_excluded(self):
        report, markdown = self._run()
        self.assertEqual(len(report["aggregates"]), 8)
        depth12 = [r for r in report["rows"] if r["policy"] == "fixed_d12_k3"]
        self.assertEqual(len(depth12), 2)
        self.assertTrue(
            all(
                (r["depth"], r["draft_length"], r["observed_exits"]) == (12, 3, []) for r in depth12
            )
        )
        self.assertEqual(report["excluded_configurations"], [[4, 3]])
        self.assertTrue(report["mismatches"])
        self.assertTrue(
            all((m["depth"], m["draft_length"]) == (4, 3) for m in report["mismatches"])
        )
        adaptive_rows = [r for r in report["rows"] if r["policy"] in ("adaptive", "bandit")]
        self.assertTrue(adaptive_rows)
        self.assertFalse(any((r["depth"], r["draft_length"]) == (4, 3) for r in adaptive_rows))
        self.assertEqual(report["aggregates"]["fixed"]["greedy_mismatches"], 2)
        self.assertTrue(report["aggregates"]["ordinary"]["all_greedy_match"])
        self.assertIn("Excluded from adaptive/bandit selection", markdown)

    def test_stop_on_mismatch_keeps_old_behaviour(self):
        with self.assertRaisesRegex(AssertionError, "depth=4, length=3"):
            self._run("--stop-on-mismatch")

    def test_full_prompt_quality_and_sparse_observers_are_wired_through_runner(self):
        with patch.object(FakeTokenizer, "encode", return_value=[1] * 200):
            report, markdown = self._run("--task-quality", "--observer-period", "4")
        self.assertTrue(all(r["prompt_tokens"] == 200 for r in report["rows"]))
        self.assertEqual(report["observer_sampling"]["period"], 4)
        self.assertEqual(len(report["quality_tasks"]), 2)
        score = report["task_quality"]["policies"]["fixed"]["gsm8k"]
        self.assertEqual(score["accuracy"], 1)
        self.assertEqual(score["tasks"], 2)
        self.assertIn("HumanEval pass@1", markdown)

    def test_existing_results_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "saved.json"
            target.write_text("original")
            with patch.object(sys, "argv", ["run", "--output", str(target)]), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    run.main()
            self.assertEqual(target.read_text(), "original")

    def test_environment_provenance_and_split_config_wired(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "split.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "gsm8k_calibration": 1,
                        "gsm8k_test": 1,
                        "humaneval_calibration": 1,
                        "humaneval_test": 1,
                    }
                ),
                encoding="utf-8",
            )
            report, markdown = self._run("--split-config", str(cfg_path))
            self.assertIn("environment_provenance", report)
            self.assertIn("git", report["environment_provenance"])
            self.assertIn("commit", report["environment_provenance"]["git"])
            self.assertIn("diff_sha256", report["environment_provenance"]["git"])
            test_rows = [r for r in report["rows"] if r["split"] == "test"]
            self.assertEqual(len(test_rows), 8 * 2)  # 8 policies * 2 test prompts

    def test_frozen_baseline_and_cost_ledger_wired(self):
        with tempfile.TemporaryDirectory() as directory:
            fb_path = Path(directory) / "frozen_baseline.json"
            fb_path.write_text(
                json.dumps(
                    {
                        "chosen_fixed_action": {
                            "depth": 8,
                            "draft_length": 2,
                            "name": "fixed_d8_k2",
                            "metric_value": 0.45,
                        }
                    }
                ),
                encoding="utf-8",
            )
            report, markdown = self._run("--frozen-baseline", str(fb_path))
            self.assertIn("frozen_baseline", report)
            self.assertEqual(
                report["frozen_baseline"]["chosen_fixed_action"]["name"], "fixed_d8_k2"
            )
            fixed_rows = [
                r for r in report["rows"] if r["policy"] == "fixed" and r["split"] == "test"
            ]
            self.assertTrue(all(r["depth"] == 8 and r["draft_length"] == 2 for r in fixed_rows))
            self.assertIn("cost_ledger", report)
            self.assertIn("cost_ledger_reconciliation", report)
            self.assertTrue(report["cost_ledger_reconciliation"]["is_reconciled"])
            self.assertIn("cumulative_trajectories", report)

    def test_controller_study_adds_frozen_context_policy(self):
        report, markdown = self._run("--controller-study")
        self.assertIn("frozen_context", report["aggregates"])
        fc_rows = [r for r in report["rows"] if r["policy"] == "frozen_context"]
        self.assertTrue(len(fc_rows) > 0)


if __name__ == "__main__":
    unittest.main()
