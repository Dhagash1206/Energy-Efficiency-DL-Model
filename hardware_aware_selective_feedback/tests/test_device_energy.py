import unittest

from hardware_aware_selective_feedback.device_energy import (
    DeviceRow,
    acceptance_steps,
    cheapest_changes,
    fit_device,
    fit_exit_profiles,
    startup_choice,
    step_winners,
    train_head_decision,
)


def _rows(accepted_j: float, rejected_j: float) -> list[DeviceRow]:
    rows = []
    specs = (
        (64, 0, 0, "greedy"),
        (40, 20, 4, "short"),
        (24, 30, 16, "long"),
        (32, 10, 8, "mid"),
    )
    for verify, accepted, rejected, setting in specs:
        for delta in (0.0, 0.05):
            joules = 0.2 + 1.0 * verify + accepted_j * accepted + rejected_j * rejected + delta
            rows.append(DeviceRow(verify, accepted, rejected, joules, setting))
    return rows


class DeviceEnergyTests(unittest.TestCase):
    def test_positive_reject_cost_is_wasted_work(self):
        fitted = fit_device(_rows(accepted_j=0.2, rejected_j=0.9))
        self.assertTrue(fitted["rejected_is_waste"])
        self.assertGreater(fitted["rejected_j"], 0)

    def test_equal_token_cost_is_not_wasted_work(self):
        fitted = fit_device(_rows(accepted_j=1.0, rejected_j=1.0))
        self.assertFalse(fitted["rejected_is_waste"])

    def test_full_accept_and_full_reject_choose_different_lengths(self):
        profile = {"verify_j": 1.0, "accepted_j": 0.2, "rejected_j": 0.9}
        winners = step_winners(profile, depth=4, accepted_counts=[4, 0])
        self.assertEqual(winners[0], (4, 4))
        self.assertEqual(winners[1], (0, 0))
        decision = train_head_decision({"rejected_is_waste": True}, winners)
        self.assertTrue(decision["train_head"])

    def test_one_winner_does_not_train(self):
        profile = {"verify_j": 1.0, "accepted_j": 0.2, "rejected_j": 0.9}
        winners = step_winners(profile, depth=4, accepted_counts=[4, 4, 4])
        decision = train_head_decision({"rejected_is_waste": True}, winners)
        self.assertFalse(decision["train_head"])
        self.assertEqual(decision["reason"], "one action wins every decode step")

    def test_each_exit_gets_its_own_joule_fit(self):
        measured = []
        for depth, rejected_j in ((4, 0.4), (8, 0.9)):
            for length, accepted, rejected in ((1, 8, 2), (2, 6, 6), (4, 4, 12), (4, 10, 4)):
                verify = 20
                for repeat in (0, 1):
                    joules = 0.1 + 1.2 * verify + 0.5 * accepted + rejected_j * rejected + 0.01 * repeat
                    measured.append({
                        "depth": depth,
                        "length": length,
                        "verify_steps": verify,
                        "accepted": accepted,
                        "rejected": rejected,
                        "joules": joules,
                    })
        fitted = fit_exit_profiles(measured)
        self.assertEqual(set(fitted), {4, 8})
        self.assertGreater(fitted[8]["rejected_j"], fitted[4]["rejected_j"])

    def test_histogram_shows_the_cheapest_action_changes(self):
        profile = {"verify_j": 1.18, "accepted_j": 0.63, "rejected_j": 0.44}
        counts = acceptance_steps(
            {"depth-4-accepted-0": 2, "depth-4-accepted-4": 2},
            4,
        )
        self.assertEqual(counts, [0, 0, 4, 4])
        self.assertTrue(cheapest_changes(profile, counts, 4))

    def test_startup_choice_drops_a_mismatched_arm(self):
        measured = [
            {"depth": 0, "length": 0, "joules": 70.0, "matched": True},
            {"depth": 0, "length": 0, "joules": 72.0, "matched": True},
            {"depth": 4, "length": 4, "joules": 48.0, "matched": True},
            {"depth": 4, "length": 4, "joules": 50.0, "matched": True},
            {"depth": 8, "length": 2, "joules": 20.0, "matched": False},
            {"depth": 8, "length": 2, "joules": 21.0, "matched": True},
        ]
        choice = startup_choice(measured)
        self.assertEqual((choice["depth"], choice["length"]), (4, 4))


if __name__ == "__main__":
    unittest.main()
