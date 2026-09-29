import unittest
from dataclasses import replace
from pathlib import Path

from .controller import Controller, State, expected_committed_tokens
from .demo import ToyModel, load_actions, run_policy


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.actions = load_actions(Path(__file__).with_name("synthetic_profile.json"))

    def test_expected_yield_includes_correction_or_bonus(self):
        self.assertEqual(expected_committed_tokens(0, 5), 1)
        self.assertEqual(expected_committed_tokens(1, 5), 6)
        self.assertAlmostEqual(expected_committed_tokens(0.5, 2), 1.75)

    def test_uncertainty_overlap_falls_back(self):
        ordinary = replace(self.actions[0], context_cost=0)
        speculative = replace(self.actions[1], energy_j=2.9,
                              acceptance_prior=1, relative_margin=0.10, context_cost=0)
        controller = Controller([ordinary, speculative])
        rows = controller.estimates(State(0, 20))
        self.assertLess(rows[1]["energy_per_token"], rows[0]["energy_per_token"])
        chosen, _ = controller.choose(State(0, 20))
        self.assertTrue(chosen["action"].ordinary)

    def test_bad_acceptance_triggers_ordinary(self):
        state = State(100, 20)
        state.history[4].extend([False] * 16)
        state.history[8].extend([False] * 16)
        chosen, _ = Controller(self.actions).choose(state)
        self.assertTrue(chosen["action"].ordinary)

    def test_last_token_uses_ordinary(self):
        chosen, _ = Controller(self.actions).choose(State(95, 1))
        self.assertTrue(chosen["action"].ordinary)

    def test_estimates_include_controller_overhead(self):
        controller = Controller(self.actions, overhead_j=0.25, overhead_ms=1.0)
        ordinary = controller.estimates(State(0, 20))[0]
        self.assertAlmostEqual(ordinary["energy_per_token"], 1.25)
        self.assertAlmostEqual(ordinary["latency_per_token_ms"], 9.0)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            run_policy(self.actions, "energy", tokens=0)

    def test_rejected_suffix_is_discarded(self):
        model = ToyModel()
        model.draft_token = lambda prefix, depth: "wrong"
        committed, checks = model.cycle([], self.actions[2], 10)
        self.assertEqual(committed, [model.target_token([])])
        self.assertEqual(checks, [False])

    def test_all_accepted_get_bonus(self):
        model = ToyModel()
        model.draft_token = lambda prefix, depth: model.target_token(prefix)
        committed, checks = model.cycle([], self.actions[2], 10)
        self.assertEqual(len(committed), 6)
        self.assertEqual(checks, [True] * 5)

    def test_all_policies_match_across_budgets_and_seeds(self):
        for seed in (0, 7, 42):
            for budget in (1, 2, 6, 49, 96):
                reference = run_policy(self.actions, "ordinary", budget, seed)["tokens"]
                for policy in ("fixed", "latency", "energy"):
                    with self.subTest(seed=seed, budget=budget, policy=policy):
                        result = run_policy(self.actions, policy, budget, seed)
                        self.assertEqual(result["tokens"], reference)
                        self.assertEqual(len(result["tokens"]), budget)
                        self.assertGreater(result["simulated_energy_j"], 0)


if __name__ == "__main__":
    unittest.main()
