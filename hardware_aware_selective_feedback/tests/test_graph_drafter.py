import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hardware_aware_selective_feedback.engine import generate, native_greedy
from hardware_aware_selective_feedback.graph_drafter import GraphDrafter
from hardware_aware_selective_feedback.vendor.self_speculation.llama_model_utils import (
    forward, forward_early,
)


class GraphDrafterTests(unittest.TestCase):
    """CPU checks of the static-cache draft step (graph capture itself needs CUDA)."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            attn_implementation="eager")).eval()

    def test_draft_steps_match_forward_early(self):
        prompt = list(range(1, 15))
        pending = torch.tensor([[prompt[-1]]])
        with torch.inference_mode():
            prefill = forward(self.model, torch.tensor([prompt[:-1]]), None)
            first = forward_early(self.model, pending, prefill.past_key_values, 3, None)
            second = forward_early(self.model, first.logits[:, -1].argmax(-1).reshape(1, 1),
                                   first.past_key_values, 3, first.exit_query_cache)
        drafter = GraphDrafter(self.model, 3, 64, use_graph=False)
        drafter.reset()
        proposed, cache, query = drafter.draft(prefill.past_key_values, pending, 2, None)
        self.assertEqual([t.item() for t in proposed],
                         [first.logits[:, -1].argmax(-1).item(), second.logits[:, -1].argmax(-1).item()])
        torch.testing.assert_close(query, second.exit_query_cache)
        for (key, value), (expected_key, expected_value) in zip(cache, second.past_key_values, strict=True):
            torch.testing.assert_close(key, expected_key)
            torch.testing.assert_close(value, expected_value)

    def test_generate_with_drafter_matches_greedy_and_plain_drafts(self):
        for prompt in ([1, 3, 5], list(range(1, 21)), [7, 8, 9, 10]):
            reference = native_greedy(self.model, prompt, 25)
            for depth, length in ((2, 1), (3, 3), (6, 2)):
                for adaptive in (False, True):
                    with self.subTest(prompt=prompt[:3], depth=depth, length=length, adaptive=adaptive):
                        drafter = GraphDrafter(self.model, depth, 80, use_graph=False)
                        graphed = generate(self.model, prompt, 25, depth, length,
                                           adaptive_drafts=adaptive, graph_drafter=drafter)
                        plain = generate(self.model, prompt, 25, depth, length, adaptive_drafts=adaptive)
                        self.assertEqual(graphed["tokens"], reference)
                        self.assertEqual([c["draft"] for c in graphed["cycles"]],
                                         [c["draft"] for c in plain["cycles"]])

    def test_invalid_use_rejected(self):
        drafter = GraphDrafter(self.model, 3, 16, use_graph=False)
        with self.assertRaises(ValueError):
            generate(self.model, [1, 2], 4, 2, 1, graph_drafter=drafter)  # depth mismatch
        with self.assertRaises(ValueError):
            generate(self.model, [1, 2], 4, 3, 1, (1,), "all", graph_drafter=drafter)
        with self.assertRaises(ValueError):
            generate(self.model, list(range(1, 20)), 4, 3, 1, graph_drafter=drafter)  # max_len
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):
                GraphDrafter(self.model, 3, 16, use_graph=True)


if __name__ == "__main__":
    unittest.main()
