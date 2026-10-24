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
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=13,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
            attn_implementation="eager")).eval()
        prompts = lambda prefix: [{"id": f"{prefix}-{i}", "dataset": "gsm8k",
                                   "text": f"question {prefix} {i}"} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            argv = ["run", "--device", "cpu", "--calibration", "1", "--test", "1",
                    "--tokens", "6", "--observer-pairs", "1", "--energy-source", "none",
                    "--threads", "1", "--output", str(output), *extra_args]
            with patch.object(sys, "argv", argv), \
                 patch.object(run, "load_prompts", return_value=(prompts("cal"), prompts("test"), {})), \
                 patch.object(run, "load_model", return_value=(model, FakeTokenizer(), "rev")), \
                 patch.object(run, "generate", side_effect=corrupt_depth4_length3), \
                 patch("builtins.print"):
                run.main()
            return json.loads(output.read_text()), output.with_suffix(".md").read_text()

    def test_mismatch_is_recorded_and_configuration_excluded(self):
        report, markdown = self._run()
        self.assertEqual(report["excluded_configurations"], [[4, 3]])
        self.assertTrue(report["mismatches"])
        self.assertTrue(all((m["depth"], m["draft_length"]) == (4, 3) for m in report["mismatches"]))
        adaptive_rows = [r for r in report["rows"] if r["policy"] in ("adaptive", "bandit")]
        self.assertTrue(adaptive_rows)
        self.assertFalse(any((r["depth"], r["draft_length"]) == (4, 3) for r in adaptive_rows))
        self.assertEqual(report["aggregates"]["fixed"]["greedy_mismatches"], 2)
        self.assertTrue(report["aggregates"]["ordinary"]["all_greedy_match"])
        self.assertIn("Excluded from adaptive/bandit selection", markdown)

    def test_stop_on_mismatch_keeps_old_behaviour(self):
        with self.assertRaisesRegex(AssertionError, "depth=4, length=3"):
            self._run("--stop-on-mismatch")


if __name__ == "__main__":
    unittest.main()
