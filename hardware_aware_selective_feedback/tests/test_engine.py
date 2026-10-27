import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from hardware_aware_selective_feedback.engine import (
    FALLBACK_BELOW, PROBE_EVERY, SHORTEN_BELOW, _adaptive_k, batched_greedy, generate,
    native_greedy,
)
from hardware_aware_selective_feedback.feedback import validate_feedback, LearningState
from hardware_aware_selective_feedback.vendor.self_speculation.llama_model_utils import forward


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32,
                        intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128, attn_implementation="eager")).eval()

    def test_native_agreement_and_observer_alignment(self):
        prompt = [1, 3, 5]
        for limit in (0, 1, 2, 9):
            reference = native_greedy(self.model, prompt, limit)
            for depth, length in ((0, 0), (1, 1), (2, 3), (3, 3)):
                for mode in ("none", "selected", "all", "periodic"):
                    with self.subTest(limit=limit, depth=depth, mode=mode):
                        result = generate(self.model, prompt, limit, depth, length, (1, 2, 3), mode)
                        self.assertEqual(result["tokens"], reference)
                        # Independent teacher-forced full prefixes check ALL observed positions.
                        for cycle in result["cycles"]:
                            for row in cycle["observations"]:
                                with torch.inference_mode():
                                    ids = torch.tensor([row["prefix_ids"]])
                                    native = self.model(ids, output_hidden_states=True)
                                    hidden = native.hidden_states[row["depth"]][:, -1]
                                    predicted = self.model.lm_head(self.model.model.norm(hidden)).argmax(-1).item()
                                self.assertEqual(row["prediction"], predicted)
                                self.assertEqual(row["target"], native.logits[:, -1].argmax(-1).item())

    def test_eos(self):
        prompt = [1, 3]
        baseline = native_greedy(self.model, prompt, 12)
        for eos in baseline[:4]:
            result = generate(self.model, prompt, 12, 2, 3, (1, 3), "all", eos=[eos])
            self.assertEqual(result["tokens"], native_greedy(self.model, prompt, 12, [eos]))

    def test_last_prompt_logits_preserve_cache_and_prediction(self):
        prompt = torch.tensor([[1, 3, 5, 7]])
        with torch.inference_mode():
            full = forward(self.model, prompt, None)
            compact = forward(self.model, prompt, None, logits_to_keep=1)
        self.assertEqual(compact.logits.shape, (1, 1, 32))
        torch.testing.assert_close(compact.logits, full.logits[:, -1:])
        for original_layer, compact_layer in zip(full.past_key_values, compact.past_key_values):
            for original_tensor, compact_tensor in zip(original_layer, compact_layer):
                torch.testing.assert_close(original_tensor, compact_tensor, rtol=0, atol=0)

    def test_rejected_branch_not_failure(self):
        rows = validate_feedback([10], [1, 2, 3], [1, 9, 3, 4], {2: [1, 9, 8, 5]}, 1, 2)
        self.assertEqual([r["valid"] for r in rows], [True, True, False, False])
        self.assertEqual([r["match"] for r in rows], [True, True, None, None])
        state = LearningState()
        state.update(rows)
        self.assertEqual(state.export()["2"]["observations"], 2)

    def test_eos_excludes_later_observations(self):
        rows = validate_feedback([10], [1, 2, 3], [1, 2, 3, 4], {2: [1, 2, 3, 4]}, 3, 1)
        self.assertEqual([r["valid"] for r in rows], [True, False, False, False])

    def test_rejection_rolls_back_cache_and_continues(self):
        found_rejection = False
        for prompt in ([1, 3, 5], [2, 4, 6], [7, 8, 9]):
            result = generate(self.model, prompt, 14, 1, 3)
            self.assertEqual(result["tokens"], native_greedy(self.model, prompt, 14))
            if any(c["accepted"] < len(c["draft"]) for c in result["cycles"]):
                found_rejection = True
                self.assertGreater(len(result["cycles"]), 1)
        self.assertTrue(found_rejection, "Tiny test model must exercise a rejected draft")

    def test_adaptive_draft_length_thresholds(self):
        drafts, limit = 3, 100
        self.assertEqual(_adaptive_k(drafts, limit, 0, []), 3)
        self.assertEqual(_adaptive_k(drafts, limit, 0, [SHORTEN_BELOW]), 3)
        self.assertEqual(_adaptive_k(drafts, limit, 0, [FALLBACK_BELOW]), 2)
        self.assertEqual(_adaptive_k(1, limit, 0, [FALLBACK_BELOW]), 1)
        # Below the fallback threshold: ordinary steps, then a 1-token re-probe.
        self.assertEqual(_adaptive_k(drafts, limit, 0, [0.0], ordinary_streak=0), 0)
        self.assertEqual(_adaptive_k(drafts, limit, 0, [0.0], ordinary_streak=PROBE_EVERY), 1)
        # Never drafts past the remaining token budget.
        self.assertEqual(_adaptive_k(drafts, 5, 3, []), 1)

    def test_adaptive_fallback_keeps_greedy_output(self):
        found_fallback = False
        for prompt in ([1, 3, 5], [2, 4, 6], [7, 8, 9]):
            reference = native_greedy(self.model, prompt, 30)
            for depth, length in ((1, 3), (2, 2), (3, 1)):
                with self.subTest(prompt=prompt, depth=depth, length=length):
                    result = generate(self.model, prompt, 30, depth, length, adaptive_drafts=True)
                    self.assertEqual(result["tokens"], reference)
                    drafts = [len(c["draft"]) for c in result["cycles"]]
                    if 0 in drafts[1:]:
                        found_fallback = True
                        # Fallback must not be permanent: a later probe drafts again.
                        first_fallback = drafts.index(0)
                        if len(drafts) > first_fallback + PROBE_EVERY + 1:
                            self.assertTrue(any(drafts[first_fallback:]))
        self.assertTrue(found_fallback, "Tiny test model must trigger the ordinary fallback")

    def test_batched_greedy_matches_per_prompt_greedy(self):
        prompts = [[1, 3, 5], [2, 4, 6, 8, 10, 12], [7], [9, 9, 9, 9]]
        for limit in (0, 1, 15):
            with self.subTest(limit=limit):
                expected = [native_greedy(self.model, p, limit) for p in prompts]
                self.assertEqual(batched_greedy(self.model, prompts, limit), expected)
        # Rows stop at their own EOS while the rest of the batch continues.
        eos = (native_greedy(self.model, prompts[0], 15)[3],)
        self.assertEqual(batched_greedy(self.model, prompts, 15, eos),
                         [native_greedy(self.model, p, 15, eos) for p in prompts])
        with self.assertRaises(ValueError):
            batched_greedy(self.model, [[1], []], 4)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            generate(self.model, [1], depth=4, drafts=1)
        with self.assertRaises(ValueError):
            generate(self.model, [1], exits=(0,))

    def test_forward_remainder_position_ids_deep_model(self):
        """Regression: forward_remainder used full_past_key_values_length=0 for upper
        layers on their first call, anchoring position IDs at 0 instead of the actual
        sequence position.  This caused argmax flips on prompts where early-layer
        draft acceptance pushed the sequence position well past 0.

        Requires a model deeper than the exit depth (16 layers, exit at depth 4)
        and a prompt long enough that the wrong positional anchor produces a
        different argmax (prompt length 20, 25 generated tokens).
        """
        torch.manual_seed(42)
        deep_model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=64,
            intermediate_size=128, num_hidden_layers=16,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, attn_implementation="eager",
        )).eval()
        prompt = list(range(1, 21))  # 20 tokens — long enough to trigger wrong positions
        limit = 25
        reference = native_greedy(deep_model, prompt, limit)
        for depth, length in ((4, 1), (4, 2), (8, 1)):
            with self.subTest(depth=depth, length=length):
                result = generate(deep_model, prompt, limit, depth, length)
                self.assertEqual(
                    result["tokens"], reference,
                    msg=(f"depth={depth} length={length}: token mismatch at index "
                         f"{next((i for i,(a,b) in enumerate(zip(result['tokens'],reference)) if a!=b), len(reference))}"),
                )


if __name__ == "__main__":
    unittest.main()
