import unittest

from hardware_aware_selective_feedback.reject_energy import RejectRow, fit_reject


def _rows(rejected_cost: float, noise: float) -> list[RejectRow]:
    rows = []
    for committed, rejected, setting in (
        (16, 0, "greedy"),
        (16, 4, "draft"),
        (12, 8, "long"),
        (20, 1, "short"),
    ):
        for repeat, delta in enumerate((0.0, noise)):
            joules = 0.2 + 0.5 * committed + rejected_cost * rejected + delta
            rows.append(RejectRow(committed, rejected, joules, setting))
    return rows


class RejectEnergyTests(unittest.TestCase):
    def test_extra_reject_cost_is_real(self):
        fitted = fit_reject(_rows(rejected_cost=2.0, noise=0.01))
        self.assertTrue(fitted["reject_is_real"])
        self.assertGreater(fitted["rejected_j"], fitted["committed_j"])

    def test_equal_token_cost_is_not_an_energy_residual(self):
        fitted = fit_reject(_rows(rejected_cost=0.5, noise=0.01))
        self.assertFalse(fitted["reject_is_real"])


if __name__ == "__main__":
    unittest.main()
