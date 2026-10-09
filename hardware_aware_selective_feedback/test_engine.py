import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from hardware_aware_selective_feedback.engine import generate, native_greedy
from hardware_aware_selective_feedback.feedback import validate_feedback, LearningState
from hardware_aware_selective_feedback.llama_model_utils import forward


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

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            generate(self.model, [1], depth=4, drafts=1)
        with self.assertRaises(ValueError):
            generate(self.model, [1], exits=(0,))

if __name__ == "__main__":
    unittest.main()
