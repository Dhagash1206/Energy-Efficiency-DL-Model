import time
import unittest

from hardware_aware_selective_feedback.pelm_round import (
    FEATURE_COUNT,
    RoundNet,
    RoundRuntime,
    action_cost,
    cheapest_action,
    menu,
    nudge,
    pack_features,
    train_network,
)


def _teacher():
    fits = {
        4: {"verify_j": 1.18, "accepted_j": 0.63, "rejected_j": 0.44},
        8: {"verify_j": 1.40, "accepted_j": 0.70, "rejected_j": 0.55},
    }
    speeds = {"0/0": 40.0, "4/1": 50.0, "4/2": 55.0, "4/4": 60.0, "8/1": 45.0, "8/2": 48.0, "8/4": 52.0}
    return {
        "fits": fits,
        "varies": True,
        "speeds": speeds,
        "speed_floor": 40.0,
        "shared_verify_j": 1.18,
    }


class PelmRoundTests(unittest.TestCase):
    def test_features_are_telemetry_only(self):
        features = pack_features(50.0, 45.0, 1800.0, 60.0, 4)
        self.assertEqual(tuple(features.shape), (FEATURE_COUNT,))
        self.assertEqual(RoundNet(menu()).hidden.in_features, FEATURE_COUNT)

    def test_heat_shortens_the_draft(self):
        teacher = _teacher()
        actions = menu()
        speeds = {tuple(int(part) for part in key.split("/")): value for key, value in teacher["speeds"].items()}

        def winner(temp):
            costs = [
                action_cost(
                    teacher["fits"], teacher["shared_verify_j"], speeds, teacher["speed_floor"],
                    action, kept=4, temp_c=temp, tokens_per_s=60.0,
                )
                for action in actions
            ]
            return cheapest_action(costs, actions)

        cool = winner(45.0)
        hot = winner(86.0)
        self.assertEqual(cool[2], 16)
        self.assertGreater(cool[1], hot[1])

    def test_training_follows_the_joule_labels(self):
        net, record = train_network(_teacher(), steps=200)
        self.assertGreater(record["label_match"], 0.9)
        self.assertGreater(len(record["distinct_labels"]), 1)
        self.assertGreater(record["cool_action"]["length"], record["hot_action"]["length"])
        self.assertIsInstance(net, RoundNet)

    def test_nudge_follows_a_measured_round(self):
        profile = {"verify_j": 1.0, "accepted_j": 0.5, "rejected_j": 0.4}
        nudge(profile, joules=3.0, accepted=2, rejected=1)
        self.assertGreater(profile["verify_j"], 1.0)

    def test_a_slow_last_round_can_still_draft(self):
        teacher = _teacher()
        actions = menu()
        speeds = {tuple(int(part) for part in key.split("/")): value for key, value in teacher["speeds"].items()}
        costs = [
            action_cost(
                teacher["fits"], teacher["shared_verify_j"], speeds, teacher["speed_floor"],
                action, kept=4, temp_c=45.0, tokens_per_s=10.0,
            )
            for action in actions
        ]
        self.assertGreater(cheapest_action(costs, actions)[1], 0)

    def test_energy_counter_stays_outside_the_round(self):
        class Meter:
            def counter(self):
                raise AssertionError("the energy counter was polled inside a round")

        teacher = _teacher()
        teacher["locked"] = {"depth": 4, "length": 4}
        net, _ = train_network(teacher, steps=1)
        runtime = RoundRuntime(net, teacher, Meter(), hold_s=10.0)
        runtime.last_time = time.perf_counter()
        self.assertEqual(runtime.last_kept, 4)
        self.assertEqual(runtime.choose(None, None), (4, 4, 16))
        runtime.on_cycle(
            {"accepted": 0, "draft": [1, 2, 3, 4], "committed": [9], "chosen_depth": 4, "chosen_drafts": 4}
        )
        self.assertEqual(runtime.last_kept, 1)
        before = runtime.fits[4]["verify_j"]
        runtime.nudge_last(4.0)
        self.assertGreater(runtime.fits[4]["verify_j"], before)
        runtime.on_cycle(
            {"accepted": 0, "draft": [], "committed": [8], "chosen_depth": 0, "chosen_drafts": 0}
        )
        self.assertEqual(runtime.last_kept, 1)


if __name__ == "__main__":
    unittest.main()
