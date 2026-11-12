import unittest

from hardware_aware_selective_feedback.cost_ledger import CostLedger
from hardware_aware_selective_feedback.paired_analysis import bootstrap_paired_difference


class TestPhaseP5CostLedgerAndPairedAnalysis(unittest.TestCase):
    def test_cost_ledger_reconciliation(self):
        ledger = CostLedger()
        ledger.record(
            request_index=0,
            prompt_id="p1",
            policy="adaptive",
            repeat=0,
            generated_tokens=10,
            generation_energy_j=5.0,
            overhead_energy_j=0.5,
            generation_latency_s=0.2,
            overhead_latency_s=0.01,
        )
        ledger.record(
            request_index=1,
            prompt_id="p2",
            policy="adaptive",
            repeat=0,
            generated_tokens=10,
            generation_energy_j=6.0,
            overhead_energy_j=0.6,
            generation_latency_s=0.25,
            overhead_latency_s=0.01,
        )

        reconciliation = ledger.reconcile()
        self.assertTrue(reconciliation["is_reconciled"])
        self.assertEqual(reconciliation["total_generation_energy_j"], 11.0)
        self.assertEqual(reconciliation["total_overhead_energy_j"], 1.1)
        self.assertAlmostEqual(reconciliation["total_request_energy_j"], 12.1)
        self.assertEqual(reconciliation["missing_energy_entries"], 0)

    def test_cumulative_trajectory_crossing_and_projection(self):
        ledger = CostLedger()
        # Candidate has setup cost 10 J, but saves 2 J per request (candidate=3J, baseline=5J)
        for i in range(10):
            pid = f"p{i}"
            ledger.record(
                request_index=i,
                prompt_id=pid,
                policy="candidate",
                repeat=0,
                generated_tokens=10,
                generation_energy_j=3.0,
                overhead_energy_j=0.0,
                generation_latency_s=0.1,
                overhead_latency_s=0.0,
            )
            ledger.record(
                request_index=i,
                prompt_id=pid,
                policy="baseline",
                repeat=0,
                generated_tokens=10,
                generation_energy_j=5.0,
                overhead_energy_j=0.0,
                generation_latency_s=0.15,
                overhead_latency_s=0.0,
            )

        trajectory = ledger.compute_cumulative_trajectory(
            candidate_policy="candidate",
            baseline_policy="baseline",
            setup_cost_candidate_j=10.0,
            setup_cost_baseline_j=0.0,
        )

        self.assertEqual(trajectory["matched_requests"], 10)
        self.assertEqual(trajectory["mean_saving_per_request_j"], 2.0)
        # 10 / 2 = 5 requests to cross
        self.assertEqual(trajectory["projected_stationary_crossing"], 5)
        self.assertEqual(trajectory["observed_crossing_step"], 5)
        self.assertIsNone(trajectory["no_crossing_reason"])

    def test_no_crossing_when_candidate_uses_more_energy(self):
        ledger = CostLedger()
        for i in range(5):
            pid = f"p{i}"
            ledger.record(
                request_index=i,
                prompt_id=pid,
                policy="candidate",
                repeat=0,
                generated_tokens=10,
                generation_energy_j=6.0,
                overhead_energy_j=0.0,
                generation_latency_s=0.2,
                overhead_latency_s=0.0,
            )
            ledger.record(
                request_index=i,
                prompt_id=pid,
                policy="baseline",
                repeat=0,
                generated_tokens=10,
                generation_energy_j=5.0,
                overhead_energy_j=0.0,
                generation_latency_s=0.2,
                overhead_latency_s=0.0,
            )

        trajectory = ledger.compute_cumulative_trajectory(
            candidate_policy="candidate",
            baseline_policy="baseline",
            setup_cost_candidate_j=5.0,
            setup_cost_baseline_j=0.0,
        )
        self.assertIsNone(trajectory["projected_stationary_crossing"])
        self.assertIn("not lower", trajectory["no_crossing_reason"])

    def test_paired_bootstrap_confidence_interval(self):
        # Baseline uses 10 J consistently, candidate uses 7 J -> true mean saving 3 J
        baseline = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7]
        candidate = [7.0, 7.1, 6.9, 7.0, 7.2, 6.8, 7.1, 7.0]

        res = bootstrap_paired_difference(baseline, candidate, confidence_level=0.95, seed=42)
        self.assertAlmostEqual(res.mean_difference, 3.0, delta=0.2)
        self.assertTrue(res.strictly_positive)
        self.assertFalse(res.crosses_zero)
        self.assertGreater(res.ci_lower, 2.5)
        self.assertLess(res.ci_upper, 3.5)

        # Inconclusive case: mean diff near 0
        noisy_cand = [10.0, 9.8, 10.2, 10.1, 9.7, 10.1, 10.0, 9.9]
        res_noisy = bootstrap_paired_difference(
            baseline, noisy_cand, confidence_level=0.95, seed=42
        )
        self.assertTrue(res_noisy.crosses_zero)

    def test_prompt_clustered_bootstrap(self):
        # 4 unique prompts with 2 repeats each (8 total records)
        records = [
            {"prompt_id": "p1", "baseline_value": 10.0, "candidate_value": 7.0},
            {"prompt_id": "p1", "baseline_value": 10.1, "candidate_value": 7.1},
            {"prompt_id": "p2", "baseline_value": 12.0, "candidate_value": 8.0},
            {"prompt_id": "p2", "baseline_value": 11.9, "candidate_value": 8.1},
            {"prompt_id": "p3", "baseline_value": 5.0, "candidate_value": 3.0},
            {"prompt_id": "p3", "baseline_value": 5.1, "candidate_value": 3.1},
            {"prompt_id": "p4", "baseline_value": 8.0, "candidate_value": 6.0},
            {"prompt_id": "p4", "baseline_value": 8.2, "candidate_value": 5.9},
        ]
        from hardware_aware_selective_feedback.paired_analysis import (
            bootstrap_clustered_paired_difference,
        )

        res = bootstrap_clustered_paired_difference(records, num_samples=500, seed=42)
        self.assertEqual(res.num_clusters, 4)
        self.assertEqual(res.total_observations, 8)
        self.assertTrue(res.strictly_positive)
        self.assertFalse(res.crosses_zero)
        self.assertGreater(res.ci_lower, 2.0)


if __name__ == "__main__":
    unittest.main()
