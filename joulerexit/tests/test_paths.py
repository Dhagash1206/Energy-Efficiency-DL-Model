import unittest

import torch

from joulerexit.controller import fit_controller
from joulerexit.paths import hard_paths, soft_paths, straight_through
from joulerexit.profile_io import energy_scale, load_profile
from joulerexit.skip_forward import _attention_mask
from joulerexit.stats import joules_per_correct, paired_savings
from joulerexit.tasks import gsm8k_gold, normalize_number, yes_no
import numpy as np


class PathTests(unittest.TestCase):
    def test_soft_paths_sum_to_one_and_prompt_mass_stays_at_the_last_exit(self):
        early = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 1.0]]])
        paths = soft_paths(early)
        self.assertTrue(torch.allclose(paths.sum(dim=-1), torch.ones(1, 2)))
        self.assertAlmostEqual(float(paths[0, 0, -1]), 1.0)
        self.assertAlmostEqual(float(paths[0, 1, 0]), 0.5)

    def test_straight_through_matches_the_hard_sample_and_trains_the_soft_path(self):
        early = torch.full((1, 1, 1), 0.25, requires_grad=True)
        soft = soft_paths(early)
        hard = hard_paths(torch.zeros(1, 1, 1))
        mixed = straight_through(hard, soft)
        self.assertTrue(torch.equal(mixed, hard))
        weights = torch.tensor([1.0, 5.0])
        (mixed[0, 0] * weights).sum().backward()
        self.assertIsNotNone(early.grad)
        self.assertGreater(float(early.grad.abs().sum()), 0.0)

    def test_exited_token_is_not_a_key_and_keeps_a_finite_self_row(self):
        reached = torch.tensor([[True, False, True]])
        mask = _attention_mask(reached, torch.float32)
        self.assertEqual(float(mask[0, 0, 2, 0]), 0.0)
        self.assertLess(float(mask[0, 0, 2, 1]), 0.0)
        self.assertEqual(float(mask[0, 0, 1, 1]), 0.0)


class ProfileTests(unittest.TestCase):
    def test_saved_4060_profile_is_real_and_depth_gap_sets_the_scale(self):
        profile = load_profile("joulerexit/results/rtx4060-smollm-probe.json")
        self.assertTrue(profile.residual_is_real)
        scale = energy_scale(profile, (8, 15, 23, 30))
        self.assertGreater(scale, 0.0)
        gap = profile.predicted_joules(30, 64, "decode") - profile.predicted_joules(8, 64, "decode")
        self.assertAlmostEqual(scale * gap, 1.5, places=5)


class ScoreTests(unittest.TestCase):
    def test_numbers_and_yes_no(self):
        self.assertEqual(gsm8k_gold("work\n#### 1,024"), "1024")
        self.assertEqual(normalize_number("the answer is 3.0"), "3")
        self.assertEqual(yes_no("I think yes"), "yes")

    def test_controller_learns_a_low_entropy_halt(self):
        features = [(0.1, 0.0, 8.0)] * 10 + [(5.0, 0.0, 8.0)] * 10
        labels = [1] * 10 + [0] * 10
        controller = fit_controller(features, labels)
        self.assertGreater(controller.probability(0.1, 0.0, 8.0), controller.probability(5.0, 0.0, 8.0))

    def test_ratio_and_paired_sign(self):
        joules = np.asarray([2.0, 2.0])
        correct = np.asarray([1.0, 1.0])
        self.assertAlmostEqual(joules_per_correct(joules, correct), 2.0)
        diff = paired_savings(joules, correct, np.asarray([1.0, 1.0]), correct, draws=200)
        self.assertGreater(diff["low"], 0.0)


if __name__ == "__main__":
    unittest.main()
