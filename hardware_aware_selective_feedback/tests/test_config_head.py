import unittest

import torch

from hardware_aware_selective_feedback.config_head import ConfigHead, action_costs, head_loss


class ConfigHeadTests(unittest.TestCase):
    def test_loss_falls_when_the_label_is_learned(self):
        torch.manual_seed(0)
        head = ConfigHead(8, actions=((0, 0), (4, 4)))
        hidden = torch.randn(16, 8)
        labels = torch.zeros(16, dtype=torch.long)
        costs = torch.tensor([[0.2, 1.5]]).expand(16, -1).contiguous()
        optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
        first = None
        for _ in range(30):
            loss = head_loss(head(hidden), labels, costs)
            if first is None:
                first = float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), first)
        self.assertEqual(head.action_at(hidden[0]), (0, 0))

    def test_full_accept_is_cheaper_than_greedy_under_the_profile(self):
        profile = {"verify_j": 1.0, "accepted_j": 0.2, "rejected_j": 0.9}
        costs = action_costs(profile, accepted=4)
        self.assertLess(costs[3], costs[0])


if __name__ == "__main__":
    unittest.main()
