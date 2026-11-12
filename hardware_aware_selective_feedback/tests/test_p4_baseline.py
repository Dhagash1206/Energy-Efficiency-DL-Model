import tempfile
import unittest
from pathlib import Path

from hardware_aware_selective_feedback.baseline_selection import (
    load_frozen_baseline,
    save_frozen_baseline,
    select_best_fixed_baseline,
)


class TestPhaseP4BaselineSelection(unittest.TestCase):
    def test_select_best_fixed_baseline_disqualifies_mismatches(self):
        sweep_data = [
            # Candidate 1: low energy, but 1 greedy mismatch -> Disqualified
            {
                "depth": 4,
                "draft_length": 3,
                "runs": 10,
                "greedy_mismatches": 1,
                "energy_j_per_token": 0.50,
                "latency_s_per_token": 0.02,
                "total_energy_j": 100.0,
                "total_latency_s": 4.0,
            },
            # Candidate 2: clean match, higher energy
            {
                "depth": 8,
                "draft_length": 3,
                "runs": 10,
                "greedy_mismatches": 0,
                "energy_j_per_token": 0.65,
                "latency_s_per_token": 0.025,
                "total_energy_j": 130.0,
                "total_latency_s": 5.0,
            },
            # Candidate 3: clean match, lowest valid energy -> Winner!
            {
                "depth": 12,
                "draft_length": 3,
                "runs": 10,
                "greedy_mismatches": 0,
                "energy_j_per_token": 0.60,
                "latency_s_per_token": 0.022,
                "total_energy_j": 120.0,
                "total_latency_s": 4.5,
            },
        ]

        manifest = select_best_fixed_baseline(sweep_data, objective="energy_j_per_token")
        winner = manifest["chosen_fixed_action"]

        self.assertEqual(winner["depth"], 12)
        self.assertEqual(winner["draft_length"], 3)
        self.assertEqual(winner["metric_value"], 0.60)
        self.assertEqual(winner["name"], "fixed_d12_k3")

        # Tuning cost accumulated
        cost = manifest["development_tuning_cost"]
        self.assertEqual(cost["total_tuning_energy_j"], 350.0)
        self.assertEqual(cost["candidates_evaluated"], 3)

        # Anchors recorded
        self.assertIn("fixed_d4_k3", manifest["anchors"])
        self.assertIn("fixed_d8_k3", manifest["anchors"])
        self.assertIn("fixed_d12_k3", manifest["anchors"])

    def test_save_and_load_frozen_baseline(self):
        sweep_data = [
            {
                "depth": 4,
                "draft_length": 3,
                "runs": 5,
                "greedy_mismatches": 0,
                "energy_j_per_token": 0.70,
                "latency_s_per_token": 0.03,
                "total_energy_j": 50.0,
                "total_latency_s": 2.0,
            }
        ]
        manifest = select_best_fixed_baseline(sweep_data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "frozen_baseline.json"
            save_frozen_baseline(manifest, out_file)
            self.assertTrue(out_file.exists())

            loaded = load_frozen_baseline(out_file)
            self.assertEqual(
                loaded["chosen_fixed_action"]["name"], manifest["chosen_fixed_action"]["name"]
            )
            self.assertEqual(
                loaded["chosen_fixed_action"]["metric_value"],
                manifest["chosen_fixed_action"]["metric_value"],
            )


if __name__ == "__main__":
    unittest.main()
