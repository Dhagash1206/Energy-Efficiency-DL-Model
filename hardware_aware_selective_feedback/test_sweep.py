"""Regression checks for valid paired energy comparisons and cleanup."""
import unittest
from unittest.mock import Mock

from .measurement import Meter
from .selector import Action
from .sweep import summarize


def request(depth=0, length=0, energy=10.0, matched=True):
    return {"depth": depth, "draft_length": length, "prompt_id": "example",
            "repeat": 0, "generated_tokens": 5, "energy_j": energy,
            "latency_s": 1.0, "greedy_match": matched}


class SweepTests(unittest.TestCase):
    actions = (Action(), Action(4, 1))

    def test_paired_saving(self):
        result = summarize([request(), request(4, 1, 8.0)], self.actions)[1]
        self.assertEqual(result["energy_j_per_token"], 1.6)
        self.assertEqual(result["mean_paired_energy_delta_j"], -2.0)
        self.assertTrue(result["valid_comparison"])

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


if __name__ == "__main__":
    unittest.main()
