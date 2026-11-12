import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hardware_aware_selective_feedback.del_policy import (
    DELConfig,
    DELController,
    capture_hidden,
    generate_del,
    shadow_predictions,
)
from hardware_aware_selective_feedback.del_comparison import compare_policies
from hardware_aware_selective_feedback.engine import native_greedy
from hardware_aware_selective_feedback.vendor.self_speculation.llama_model_utils import forward


class ControllerTests(unittest.TestCase):
    def test_tpl_extremes_and_finite_thresholds(self):
        controller = DELController(4, DELConfig(max_drafts=3))
        predictions = {d: [1, 2] for d in range(1, 4)}
        scores = {d: [0.8, 0.6] for d in predictions}
        controller.update(predictions, scores, [1, 2], 2)
        self.assertEqual(controller.choose()[:2], (1, 3))
        self.assertEqual(controller.threshold(1), 0.0)
        controller = DELController(4)
        controller.update(predictions, scores, [3, 4], 2)
        self.assertEqual(controller.choose()[:2], (1, 0))

    def test_valid_prefix_decay_and_threshold(self):
        controller = DELController(2, DELConfig(decay=0.5))
        controller.update({1: [1, 9, 8]}, {1: [0.8, 0.2, 0.99]}, [1, 2, 3], 2)
        self.assertEqual(controller.stats[1][:2], [1, 1])
        self.assertAlmostEqual(controller.threshold(1), 0.5)
        controller.update({1: [1]}, {1: [0.6]}, [1], 1)
        self.assertEqual(controller.stats[1][:2], [1.5, 0.5])
        self.assertAlmostEqual(controller.stats[1][2], 1.0)

    def test_invalid_parameters(self):
        for kwargs in (
            {"decay": 0},
            {"decay": float("nan")},
            {"max_drafts": 0},
            {"projection_chunk": 0},
        ):
            with self.assertRaises(ValueError):
                DELConfig(**kwargs)


class DELGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        cls.model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                attn_implementation="eager",
            )
        ).eval()

    def test_greedy_agreement_budgets_eos_and_chunking(self):
        for prompt in ([1], [1, 3, 5], [2, 4] * 20):
            reference = native_greedy(self.model, prompt, 20)
            for limit in (0, 1, 2, 11, 20):
                for chunk in (1, 7, 32):
                    output = generate_del(
                        self.model,
                        prompt,
                        limit,
                        config=DELConfig(max_drafts=4, projection_chunk=chunk),
                    )
                    self.assertEqual(output["tokens"], reference[:limit])
            for stop in (reference[0], reference[3]):
                self.assertEqual(
                    generate_del(self.model, prompt, 20, eos=(stop,))["tokens"],
                    native_greedy(self.model, prompt, 20, eos=(stop,)),
                )

    def test_changing_depth_and_ordinary_rounds_preserve_cache(self):
        prompt = [1, 3, 5]
        reference = native_greedy(self.model, prompt, 25)
        choices = iter([(1, 3, 0.5), (3, 3, 0.5), (2, 0, 0.25), (2, 2, 0.4)] * 25)
        with (
            patch.object(DELController, "choose", side_effect=lambda: next(choices)),
            patch.object(DELController, "threshold", return_value=0.0),
        ):
            output = generate_del(self.model, prompt, 25, config=DELConfig(max_drafts=3))
        self.assertEqual(output["tokens"], reference)
        self.assertTrue({0, 1, 2, 3}.issubset({c["depth"] for c in output["cycles"]}))
        self.assertTrue(any(c["accepted"] < len(c["draft"]) for c in output["cycles"]))

    def test_confidence_stops_drafting_after_trigger_token(self):
        with (
            patch.object(DELController, "choose", return_value=(2, 3, 0.5)),
            patch.object(DELController, "threshold", return_value=1.0),
        ):
            output = generate_del(self.model, [1, 3], 15)
        self.assertEqual(output["tokens"], native_greedy(self.model, [1, 3], 15))
        self.assertTrue(all(len(c["draft"]) <= 1 for c in output["cycles"]))

    def test_shadow_predictions_against_independent_hidden_states(self):
        ids = torch.tensor([[1, 3, 5, 7]])
        with torch.inference_mode():
            with capture_hidden(self.model, tail=3) as states:
                forward(self.model, ids, None, logits_to_keep=3)
            predictions, confidence = shadow_predictions(self.model, states, 3, 2)
            native = self.model(ids, output_hidden_states=True)
            for d in range(1, 4):
                logits = self.model.lm_head(self.model.model.norm(native.hidden_states[d][:, -3:]))
                probs, tokens = logits.softmax(-1).max(-1)
                self.assertEqual(predictions[d], tokens[0].tolist())
                torch.testing.assert_close(torch.tensor(confidence[d]), probs[0])

    def test_hook_cleanup_on_error(self):
        with self.assertRaises(RuntimeError):
            with capture_hidden(self.model):
                raise RuntimeError("test")
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))


class ComparisonTests(unittest.TestCase):
    def rows(self):
        return [
            dict(
                policy=p,
                split="test",
                prompt_id="p",
                repeat=r,
                tokens=[1, 2],
                generated_tokens=2,
                greedy_match=True,
                energy_j=e,
                latency_s=e / 10,
            )
            for r in range(2)
            for p, e in (("adaptive", 2), ("del", 4))
        ]

    def test_paired_metric_and_prompt_unit(self):
        result = compare_policies(self.rows(), "adaptive", "del")
        self.assertTrue(result["valid"])
        energy = result["metrics"]["energy_j"]
        self.assertEqual(energy["candidate_reduction_percent"], 50)
        self.assertEqual(energy["prompts"], 1)
        self.assertEqual(result["pairs"], 2)

    def test_mismatch_missing_and_duplicate_pairs(self):
        rows = self.rows()
        rows[0]["greedy_match"] = False
        self.assertFalse(compare_policies(rows, "adaptive", "del")["valid"])
        self.assertFalse(compare_policies(self.rows()[:-1], "adaptive", "del")["valid"])
        with self.assertRaises(ValueError):
            compare_policies(self.rows() + [self.rows()[0]], "adaptive", "del")
        rows = self.rows()
        rows[0]["energy_j"] = None
        self.assertIsNone(compare_policies(rows, "adaptive", "del")["metrics"]["energy_j"])

    def run_report(self, corrupt_del=False):
        from hardware_aware_selective_feedback import run
        from hardware_aware_selective_feedback.tests.test_run import FakeTokenizer

        torch.manual_seed(3)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=13,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                attn_implementation="eager",
            )
        ).eval()

        def prompts(prefix):
            return [{"id": prefix, "dataset": "gsm8k", "text": "abc"}]

        def del_result(*args, **kwargs):
            output = generate_del(*args, **kwargs)
            if corrupt_del:
                output["tokens"][-1] = (output["tokens"][-1] + 1) % 32
            return output

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "del.json"
            argv = [
                "run",
                "--include-del",
                "--device",
                "cpu",
                "--energy-source",
                "none",
                "--tokens",
                "6",
                "--observer-pairs",
                "1",
                "--threads",
                "1",
                "--del-max-drafts",
                "3",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    run, "load_prompts", return_value=(prompts("cal"), prompts("test"), {})
                ),
                patch.object(run, "load_model", return_value=(model, FakeTokenizer(), "test")),
                patch.object(run, "generate_del", side_effect=del_result),
                patch("builtins.print"),
            ):
                run.main()
            report = json.loads(output.read_text())
            return report, output.with_suffix(".md").read_text()

    def test_runner_writes_separate_del_and_fixed_baselines(self):
        report, markdown = self.run_report()
        self.assertEqual(len(report["aggregates"]), 9)
        self.assertTrue(report["del_comparisons"]["del"]["valid"])
        self.assertTrue(report["del_comparisons"]["fixed_d12_k3"]["valid"])
        del_row = next(r for r in report["rows"] if r["policy"] == "del")
        self.assertIsNone(del_row["depth"])
        self.assertIn("del_cycles", del_row)
        self.assertIn("not end-to-end", markdown)

    def test_del_mismatch_invalidates_comparison_without_excluding_fixed_actions(self):
        report, markdown = self.run_report(corrupt_del=True)
        self.assertEqual(report["excluded_configurations"], [])
        self.assertEqual(report["mismatches"][0]["policy"], "del")
        self.assertIsNone(report["mismatches"][0]["depth"])
        self.assertFalse(report["del_comparisons"]["del"]["valid"])
        self.assertTrue(report["del_comparisons"]["fixed_d12_k3"]["valid"])
        self.assertIn("Comparison invalid", markdown)


if __name__ == "__main__":
    unittest.main()
