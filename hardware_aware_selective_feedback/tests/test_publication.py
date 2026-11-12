import copy
import unittest

from hardware_aware_selective_feedback.publication import break_even, energy_components


def example():
    rows = [
        {
            "split": "calibration",
            "policy": "calibration",
            "energy_j": 20,
            "energy_boundary": "nvidia_gpu",
        }
    ]
    for prompt in ("a", "b"):
        for policy, energy in (("ordinary", 10), ("adaptive", 7)):
            rows.append(
                {
                    "split": "test",
                    "prompt_id": prompt,
                    "repeat": 0,
                    "policy": policy,
                    "tokens": [1, 2],
                    "generated_tokens": 2,
                    "greedy_match": True,
                    "energy_j": energy,
                    "latency_s": 1,
                    "energy_boundary": "nvidia_gpu",
                }
            )
    return {"rows": rows, "measurement": {"idle_calibration": {"total_energy_j": 1}}}


class PublicationTests(unittest.TestCase):
    def test_break_even_includes_all_calibration_and_idle_once(self):
        data = example()
        before = copy.deepcopy(data)
        result = break_even(data)
        self.assertTrue(result["valid"])
        self.assertEqual(result["calibration_energy_j"], 21)
        self.assertEqual(result["n_breakeven"], 7)
        self.assertEqual(data, before)

    def test_round_up_and_no_crossing(self):
        data = example()
        data["rows"][0]["energy_j"] = 21
        self.assertEqual(break_even(data)["n_breakeven"], 8)
        for row in data["rows"]:
            if row["policy"] == "adaptive":
                row["energy_j"] = 11
        result = break_even(data)
        self.assertTrue(result["valid"])
        self.assertIsNone(result["n_breakeven"])

    def test_missing_pairs_mismatches_energy_and_boundaries_do_not_produce_claims(self):
        for kind in ("missing_pair", "mismatch", "missing_energy", "mixed_boundary"):
            data = example()
            if kind == "missing_pair":
                data["rows"].pop()
            elif kind == "mismatch":
                data["rows"][-1]["greedy_match"] = False
            elif kind == "missing_energy":
                data["rows"][-1]["energy_j"] = None
            else:
                data["rows"][-1]["energy_boundary"] = "cpu_packages"
            with self.subTest(kind=kind):
                self.assertFalse(break_even(data)["valid"])
                self.assertIsNone(break_even(data)["n_breakeven"])

    def test_components_remain_missing_without_idle_measurements(self):
        self.assertIsNone(energy_components([{"generated_tokens": 2}])["dynamic_energy_j"])
        result = energy_components(
            [{"generated_tokens": 2, "static_idle_energy_j": 3, "dynamic_energy_j": -1}]
        )
        self.assertEqual(result["dynamic_energy_j_per_token"], -0.5)


if __name__ == "__main__":
    unittest.main()
