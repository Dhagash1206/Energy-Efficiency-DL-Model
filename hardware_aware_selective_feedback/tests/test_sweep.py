"""Regression checks for valid paired energy comparisons and cleanup."""
import unittest
from unittest.mock import Mock, patch

from ..measurement import Meter
from ..selector import Action
from ..sweep import measure_batched, summarize


def request(depth=0, length=0, energy=10.0, matched=True, tokens=5):
    return {"depth": depth, "draft_length": length, "prompt_id": "example",
            "repeat": 0, "generated_tokens": tokens, "energy_j": energy,
            "latency_s": 1.0, "greedy_match": matched}


class SweepTests(unittest.TestCase):
    actions = (Action(), Action(4, 1))

    def test_paired_saving(self):
        result = summarize([request(), request(4, 1, 8.0)], self.actions)[1]
        self.assertEqual(result["energy_j_per_token"], 1.6)
        # Per token: 8/5 - 10/5 = -0.4 J/token.
        self.assertAlmostEqual(result["mean_paired_energy_delta_j_per_token"], -0.4)
        self.assertTrue(result["valid_comparison"])

    def test_paired_delta_is_per_token_not_per_request(self):
        # 12 J for 10 tokens beats 10 J for 5 tokens: +2 J/request but -0.8 J/token.
        result = summarize([request(), request(4, 1, 12.0, tokens=10)], self.actions)[1]
        self.assertAlmostEqual(result["median_paired_energy_delta_j_per_token"], 1.2 - 2.0)

    def test_invalid_baseline_cannot_support_a_win(self):
        results = summarize([request(matched=False), request(4, 1, 1.0)], self.actions)
        self.assertFalse(any(r["valid_comparison"] for r in results))

    def test_invalid_candidate_is_not_eligible(self):
        result = summarize([request(), request(4, 1, 1.0, False)], self.actions)[1]
        self.assertFalse(result["valid_comparison"])

    def test_missing_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize([request()], self.actions)

    def test_duplicate_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([request(), request()], self.actions)

    def test_meter_closes_on_failure(self):
        meter = Meter("cpu", energy_source="none")
        meter.close = Mock()
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with meter:
                raise RuntimeError("load failed")
        meter.close.assert_called_once()


class FakeMeter:
    """Runs the measured function and reports 2 J and 0.5 s per call."""

    def run(self, function, min_seconds=0.0):
        return function(), {"latency_s": 0.5, "energy_j": 2.0}


class BatchedBaselineTests(unittest.TestCase):
    prompts = [{"id": name, "ids": [index + 1]} for index, name in enumerate("abc")]
    references = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}

    def _measure(self, decoded, batch_size=2, repeats=1):
        with patch("hardware_aware_selective_feedback.sweep.batched_greedy",
                   side_effect=lambda model, ids, limit, eos, pad: [decoded[i[0]] for i in ids]):
            return measure_batched(None, FakeMeter(), self.prompts, self.references,
                                   batch_size, 2, (), repeats)

    def test_groups_prompts_and_reports_per_token_cost(self):
        result = self._measure({1: [1, 1], 2: [2, 2], 3: [3, 3]}, repeats=2)
        self.assertEqual([r["prompt_ids"] for r in result["records"][:2]], [["a", "b"], ["c"]])
        self.assertEqual(result["batches"], 4)
        self.assertEqual(result["generated_tokens"], 12)
        self.assertAlmostEqual(result["energy_j_per_token"], 8.0 / 12)
        self.assertTrue(result["greedy_matches"])

    def test_mismatch_is_reported(self):
        result = self._measure({1: [1, 1], 2: [2, 9], 3: [3, 3]})
        self.assertFalse(result["greedy_matches"])
        self.assertEqual(result["records"][0]["mismatched_prompt_ids"], ["b"])


if __name__ == "__main__":
    unittest.main()
