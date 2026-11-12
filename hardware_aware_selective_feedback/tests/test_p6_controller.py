import unittest

from hardware_aware_selective_feedback.controller_study import (
    EconomicProbeDecisionRule,
    FrozenContextSelector,
    extract_permitted_features,
)
from hardware_aware_selective_feedback.selector import Action


class TestPhaseP6ControllerStudy(unittest.TestCase):
    def test_information_boundary_extraction(self):
        prompt = {
            "id": "p1",
            "dataset": "gsm8k",
            "text": "Solve 2+2",
            "answer": "#### 4",  # Reference answer MUST NOT be in features
        }
        features = extract_permitted_features(prompt, prompt_tokens=15)
        self.assertEqual(features.prompt_length_tokens, 15)
        self.assertFalse(hasattr(features, "answer"))
        self.assertFalse(hasattr(features, "expected"))

    def test_economic_probe_decision_rule(self):
        rule = EconomicProbeDecisionRule(
            probe_cost_j=0.05,
            decision_cost_j=0.001,
            uncertainty_margin_j=0.01,
            horizon_requests=10,
        )

        # 1. High benefit: 0.02 J saving * 10 requests = 0.20 J > 0.061 J hurdle -> BUY
        buy, diag = rule.should_buy_probe(
            estimated_saving_per_request_j=0.02,
            current_uncertainty=1.0,
        )
        self.assertTrue(buy)
        self.assertGreater(diag["net_expected_gain_j"], 0.1)

        # 2. Low benefit: 0.001 J saving * 10 requests = 0.01 J < 0.061 J hurdle -> DO NOT BUY
        buy_low, diag_low = rule.should_buy_probe(
            estimated_saving_per_request_j=0.001,
            current_uncertainty=1.0,
        )
        self.assertFalse(buy_low)
        self.assertLess(diag_low["net_expected_gain_j"], 0.0)

    def test_frozen_context_selector(self):
        short = Action(4, 3)
        long = Action(8, 3)
        selector = FrozenContextSelector(short, long, threshold_tokens=64)

        feat_short = extract_permitted_features({}, prompt_tokens=32)
        feat_long = extract_permitted_features({}, prompt_tokens=128)

        self.assertEqual(selector.select(feat_short), short)
        self.assertEqual(selector.select(feat_long), long)

    def test_run_controller_study_analyzes_experiment(self):
        from hardware_aware_selective_feedback.controller_study import run_controller_study

        sample_exp = {
            "aggregates": {
                "fixed": {"runs": 4, "energy_j_per_token": 1.20, "latency_s_per_token": 0.022},
                "adaptive": {"runs": 4, "energy_j_per_token": 1.25, "latency_s_per_token": 0.023},
            },
            "observation_overhead": {
                "selected_observe": {"median_delta_per_request": 7.96},
            },
        }
        res = run_controller_study(sample_exp)
        self.assertFalse(res["adaptive_beats_fixed"])
        self.assertIn("Retain the development-selected fixed baseline", res["recommendation"])
        self.assertFalse(res["probe_economics"]["economic_justification"])


if __name__ == "__main__":
    unittest.main()
