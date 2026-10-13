import unittest
from hardware_aware_selective_feedback.selector import Action, CostBandit, Selector


class SelectorTests(unittest.TestCase):
    def test_valid_feedback_only_and_no_counterfactual_cost(self):
        policy = Selector(depths=(1, 2), lengths=(1,))
        output = {"tokens": [4, 5], "cycles": [{"observations": [
            {"depth": 2, "valid": True, "match": True},
            {"depth": 2, "valid": False, "match": None},
        ]}]}
        policy.update(Action(1, 1, (2,)), {"latency_s": 2.0, "energy_j": None}, output)
        self.assertEqual(policy.matches[2], [1, 1])
        self.assertEqual(dict(policy.costs), {})
        self.assertEqual(policy.observed_costs[(1, 1)], [1.0])

    def test_unknown_actions_explored_and_observation_cost_measured(self):
        policy = Selector(depths=(1, 2), lengths=(1,))
        self.assertEqual(policy.choose(10).key, (0, 0))
        sample = ({"tokens": [1], "cycles": []}, {"latency_s": 1.0, "energy_j": None})
        policy.update(Action(), sample[1], sample[0])
        self.assertEqual(policy.choose(10).key, (1, 1))
        policy.record_observer_pair(sample, (sample[0], {"latency_s": 1.2, "energy_j": None}), "selected")
        self.assertAlmostEqual(policy.observer_cost["selected"][0], 0.2)

    def test_paid_observation_does_not_bias_base_cost(self):
        policy = Selector(depths=(1, 2), lengths=(1,))
        output = {"tokens": [1], "cycles": []}
        policy.update(Action(1, 1), {"latency_s": 1.0}, output)
        policy.update(Action(1, 1, (2,)), {"latency_s": 10.0}, output)
        self.assertEqual(policy.costs[(1, 1)], [1.0])
        self.assertEqual(policy.observed_costs[(1, 1)], [10.0])
        self.assertEqual(policy.action_score(Action(1, 1)), policy.action_score(Action(1, 1, (2,))))

    def test_executed_draft_only_counts_valid_prefix(self):
        policy = Selector(depths=(1, 2), lengths=(3,))
        output = {"tokens": [4, 5], "cycles": [{"draft": [8, 9, 10],
                  "accepted": 1, "observations": []}]}
        policy.update(Action(1, 3), {"latency_s": 2.0}, output)
        self.assertEqual(policy.matches[1], [1, 2])

    def test_independent_bandit_needs_executed_cost(self):
        arms = (Action(), Action(1, 1))
        bandit = CostBandit(arms)
        self.assertEqual(bandit.choose(), arms[0])
        bandit.update(arms[0], {"latency_s": 2.0}, {"tokens": [1]})
        self.assertEqual(bandit.choose(), arms[1])
        bandit.update(arms[1], {"latency_s": 1.0}, {"tokens": [1]})
        self.assertEqual(set(bandit.summary()), {str(a.key) for a in arms})

    def test_paid_feedback_only_on_calibrated_configuration(self):
        policy = Selector()
        output = {"tokens": [1], "cycles": []}
        for arm in policy.arms:
            policy.update(arm, {"latency_s": 0.5 if arm.key == (8, 3) else 2.0}, output)
        base = (output, {"latency_s": 0.5})
        for _ in range(3):
            policy.record_observer_pair(base, (output, {"latency_s": 0.51}), "selected")
            policy.record_observer_pair(base, (output, {"latency_s": 0.52}), "all")
        chosen = policy.choose(10)
        self.assertEqual(chosen.key, (8, 3))
        self.assertIn(chosen.observations, ((4,), (4, 12)))
        policy.costs[(4, 3)] = [0.1]
        self.assertEqual(policy.choose(10).observations, ())

    def test_energy_requires_real_reading(self):
        with self.assertRaises(ValueError):
            Selector(objective="energy").update(Action(), {"energy_j": None}, {"tokens": [1]})


if __name__ == "__main__":
    unittest.main()
