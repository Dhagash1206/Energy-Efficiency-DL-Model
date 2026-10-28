import unittest

import torch

from joulerexit.gate import ExitGates


class GateTests(unittest.TestCase):
    def test_last_exit_is_certain_and_early_exit_depends_on_residual(self):
        gates = ExitGates(hidden_size=32, n_exits=4)
        hidden = torch.zeros(2, 32)
        entropy = torch.zeros(2)
        low = torch.zeros(2)
        high = torch.full((2,), 5.0)
        last = gates.probability(3, hidden, entropy, low)
        self.assertTrue(torch.equal(last, torch.ones(2)))
        early_low = gates.probability(0, hidden, entropy, low)
        early_high = gates.probability(0, hidden, entropy, high)
        self.assertEqual(tuple(early_low.shape), (2,))
        self.assertFalse(torch.allclose(early_low, early_high))
        self.assertTrue(torch.all((early_low > 0) & (early_low < 1)))

    def test_gate_count_excludes_the_final_block(self):
        gates = ExitGates(hidden_size=8, n_exits=4)
        self.assertEqual(len(gates.gates), 3)


if __name__ == "__main__":
    unittest.main()
