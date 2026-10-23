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
        self.assertEqual(policy._matches[2], [1, 1])
        self.assertEqual(dict(policy._costs), {})
        self.assertEqual(policy._observed_costs[(1, 1)], [1.0])

    def test_unknown_actions_explored_and_observation_cost_measured(self):
        policy = Selector(depths=(1, 2), lengths=(1,))
        self.assertEqual(policy.choose(10).key, (0, 0))
        sample = ({"tokens": [1], "cycles": []}, {"latency_s": 1.0, "energy_j": None})
        policy.update(Action(), sample[1], sample[0])
        self.assertEqual(policy.choose(10).key, (1, 1))
        policy.record_observer_pair(sample, (sample[0], {"latency_s": 1.2, "energy_j": None}), "selected")
        self.assertAlmostEqual(policy._observer_cost["selected"][0], 0.2)

    def test_paid_observation_does_not_bias_base_cost(self):
        policy = Selector(depths=(1, 2), lengths=(1,))
        output = {"tokens": [1], "cycles": []}
        policy.update(Action(1, 1), {"latency_s": 1.0}, output)
        policy.update(Action(1, 1, (2,)), {"latency_s": 10.0}, output)
        self.assertEqual(policy._costs[(1, 1)], [1.0])
        self.assertEqual(policy._observed_costs[(1, 1)], [10.0])
        self.assertEqual(policy._arm_score(Action(1, 1)), policy._arm_score(Action(1, 1, (2,))))

    def test_executed_draft_only_counts_valid_prefix(self):
        policy = Selector(depths=(1, 2), lengths=(3,))
        output = {"tokens": [4, 5], "cycles": [{"draft": [8, 9, 10],
                  "accepted": 1, "observations": []}]}
        policy.update(Action(1, 3), {"latency_s": 2.0}, output)
        self.assertEqual(policy._matches[1], [1, 2])

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
        policy._costs[(4, 3)] = [0.1]
        self.assertEqual(policy.choose(10).observations, ())

    def test_energy_requires_real_reading(self):
        with self.assertRaises(ValueError):
            Selector(objective="energy").update(Action(), {"energy_j": None}, {"tokens": [1]})


if __name__ == "__main__":
    unittest.main()


class ContextSelectorTests(unittest.TestCase):
    @staticmethod
    def _record(policy, action, cost, context):
        policy.update(action, {"latency_s": cost}, {"tokens": [1], "cycles": []}, context)

    def _trained(self, context_aware):
        policy = Selector(depths=(1,), lengths=(1,), context_aware=context_aware)
        ordinary, speculative = Action(), Action(1, 1)
        # Code prompts favour speculation; math prompts favour ordinary decoding.
        for _ in range(2):
            self._record(policy, ordinary, 1.0, "code")
            self._record(policy, speculative, 0.5, "code")
            self._record(policy, ordinary, 1.0, "math")
            self._record(policy, speculative, 3.0, "math")
        return policy

    def test_context_aware_picks_best_arm_per_context(self):
        policy = self._trained(context_aware=True)
        self.assertEqual(policy.choose(context="code").key, (1, 1))
        self.assertEqual(policy.choose(context="math").key, (0, 0))
        self.assertIn("code", policy.summary()["context_costs_per_token"])

    def test_unseen_context_falls_back_to_global_costs(self):
        policy = self._trained(context_aware=True)
        self.assertEqual(policy.choose(context="new").key, policy.choose().key)

    def test_context_ignored_when_disabled(self):
        policy = self._trained(context_aware=False)
        self.assertEqual(policy.choose(context="code").key, policy.choose(context="math").key)
        self.assertEqual(policy.summary()["context_costs_per_token"], {})
