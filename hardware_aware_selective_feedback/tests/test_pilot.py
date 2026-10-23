import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from ..engine import generate, native_greedy
from ..feedback import validate_feedback, LearningState


class PilotTests(unittest.TestCase):
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

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            generate(self.model, [1], depth=4, drafts=1)
        with self.assertRaises(ValueError):
            generate(self.model, [1], exits=(0,))

if __name__ == "__main__":
    unittest.main()

