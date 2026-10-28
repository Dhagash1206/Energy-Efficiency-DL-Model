import unittest

from joulerexit.exits import exit_blocks
from joulerexit.residual import ProbeRow, fit_profile


def _rows_from_law(law, repeats=3):
    rows = []
    for depth in (8, 16, 24, 30):
        for seq_len in (32, 128):
            for phase in ("prefill", "decode"):
                joules = law(depth, seq_len, phase)
                for repeat in range(repeats):
                    # Repeat noise of about 0.01 J, identical across cells.
                    noise = 0.01 * ((repeat - 1) / 1.0)
                    rows.append(ProbeRow(depth, seq_len, phase, joules + abs(noise) * 0 + noise + 0.02))
    return rows


class ResidualTests(unittest.TestCase):
    def test_pure_depth_is_not_a_real_residual(self):
        rows = []
        for depth in (8, 16, 24, 30):
            for seq_len in (32, 128):
                for phase in ("prefill", "decode"):
                    for extra in (0.0, 0.02):
                        rows.append(ProbeRow(depth, seq_len, phase, 0.5 + 0.1 * depth + extra))
        profile = fit_profile(rows)
        self.assertFalse(profile.residual_is_real)
        self.assertAlmostEqual(profile.residual(16, 128, "decode"), 0.0, places=6)

    def test_phase_shift_is_a_real_residual(self):
        rows = []
        for depth in (8, 16, 24, 30):
            for seq_len in (32, 128):
                for phase in ("prefill", "decode"):
                    base = 0.4 + 0.05 * depth + (1.5 if phase == "decode" else 0.0)
                    for extra in (0.0, 0.02):
                        rows.append(ProbeRow(depth, seq_len, phase, base + extra))
        profile = fit_profile(rows)
        self.assertTrue(profile.residual_is_real)
        gap_shallow = profile.residual(8, 32, "decode") - profile.residual(8, 32, "prefill")
        gap_deep = profile.residual(30, 32, "decode") - profile.residual(30, 32, "prefill")
        self.assertAlmostEqual(gap_shallow, 1.5, delta=0.15)
        self.assertAlmostEqual(gap_deep, gap_shallow, places=2)

    def test_exit_maps_match_known_depths(self):
        self.assertEqual(exit_blocks("HuggingFaceTB/SmolLM2-135M", 30), (8, 15, 23, 30))
        self.assertEqual(exit_blocks("Qwen/Qwen3-0.6B", 28), (7, 14, 21, 28))
        self.assertEqual(exit_blocks("facebook/layerskip-llama3.2-1B", 16), (4, 8, 12, 16))

    def test_exit_map_rejects_a_mismatched_checkpoint(self):
        with self.assertRaises(ValueError):
            exit_blocks("Qwen/Qwen3-0.6B", 32)


if __name__ == "__main__":
    unittest.main()
