import unittest
import torch

from hardware_aware_selective_feedback.stopping import (
    StopCriteria,
    apply_stop_decision_to_block,
    crop_kv_cache,
    detect_repetition,
    evaluate_committed_stopping,
)


class DummyTokenizer:
    def __init__(self, vocab_map=None):
        self.vocab = vocab_map or {
            1: "apple",
            2: "banana",
            3: "cherry",
            10: "\nQuestion:",
            99: "<eos>",
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.vocab.get(t, f"[{t}]") for t in tokens)

    def encode(self, text, add_special_tokens=False):
        res = []
        for word in ("apple", "banana", "cherry", "\nQuestion:", "<eos>"):
            if word in text:
                res.append(self.inv_vocab[word])
        return res or [1] * len(text)


class TestPhaseP2Stopping(unittest.TestCase):
    def setUp(self):
        self.criteria = StopCriteria(
            eos_token_ids=(99, 100),
            stop_sequences=((10, 11), (5, 6)),
            stop_strings=("\nQuestion:",),
            max_tokens=20,
            repetition_ngram_size=4,
            repetition_threshold=3,
        )
        self.tokenizer = DummyTokenizer()

    def test_rejected_draft_does_not_stop(self):
        draft = [1, 2, 99, 3]
        decision = evaluate_committed_stopping(
            draft, self.criteria, tokenizer=self.tokenizer, is_committed=False
        )
        self.assertFalse(decision.stopped)

    def test_committed_eos_stops_and_crops(self):
        committed = [1, 2, 99, 3, 4]
        decision = evaluate_committed_stopping(
            committed, self.criteria, tokenizer=self.tokenizer, is_committed=True
        )
        self.assertTrue(decision.stopped)
        self.assertEqual(decision.finish_reason, "eos")
        self.assertEqual(decision.crop_index, 3)  # [1, 2, 99]

    def test_regression_earlier_stop_sequence_selected(self):
        # Criteria has ((10, 11), (5, 6)) in that order.
        # But (5, 6) occurs earlier (index 2) than (10, 11) (index 8).
        tokens = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
        decision = evaluate_committed_stopping(tokens, self.criteria, is_committed=True)
        self.assertTrue(decision.stopped)
        self.assertEqual(decision.finish_reason, "stop_sequence")
        self.assertEqual(decision.matched_boundary, (5, 6))
        self.assertEqual(decision.crop_index, 2)  # Earliest stop wins

    def test_regression_eos_beyond_max_tokens_capped_at_length(self):
        # max_tokens is 20. Tokens has 30 items. EOS occurs at index 25.
        tokens = [1] * 25 + [99] + [1] * 4  # 30 items
        decision = evaluate_committed_stopping(tokens, self.criteria, is_committed=True)
        self.assertTrue(decision.stopped)
        # Must finish for length at index 20, NOT returning 26 tokens with "eos"
        self.assertEqual(decision.finish_reason, "length")
        self.assertEqual(decision.crop_index, 20)

    def test_regression_stop_sequence_spanning_blocks(self):
        # Stop sequence is (10, 11).
        # Prior tokens end with 10. Newly committed block starts with 11.
        prior = [1, 2, 10]
        block = [11, 3, 4]
        cropped, decision = apply_stop_decision_to_block(
            prior, block, criteria=self.criteria, tokenizer=self.tokenizer
        )
        self.assertTrue(decision.stopped)
        self.assertEqual(decision.finish_reason, "stop_sequence")
        self.assertEqual(decision.matched_boundary, (10, 11))
        # Since 10 was at prior index 2, stop occurs at index 2 (prior[2])
        self.assertEqual(cropped, [])

    def test_mid_block_stop_cropping(self):
        prior = [1, 2]
        block = [3, 99, 4]  # EOS at index 1 of block -> total index 3
        cropped, decision = apply_stop_decision_to_block(
            prior, block, criteria=self.criteria, tokenizer=self.tokenizer
        )
        self.assertTrue(decision.stopped)
        self.assertEqual(cropped, [3, 99])

    def test_repetition_detection(self):
        looping = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        is_rep, pattern = detect_repetition(looping, ngram_size=4, threshold=3)
        self.assertTrue(is_rep)
        self.assertEqual(pattern, [1, 2, 3, 4])

        decision = evaluate_committed_stopping(
            looping, self.criteria, tokenizer=self.tokenizer, is_committed=True
        )
        self.assertTrue(decision.stopped)
        self.assertEqual(decision.finish_reason, "repetition")

    def test_crop_kv_cache(self):
        # 2 layers of (k, v) tensors with shape [1, 4, 30, 16]
        k1 = torch.randn(1, 4, 30, 16)
        v1 = torch.randn(1, 4, 30, 16)
        k2 = torch.randn(1, 4, 30, 16)
        v2 = torch.randn(1, 4, 30, 16)
        cache = ((k1, v1), (k2, v2))

        cropped = crop_kv_cache(cache, 18)
        self.assertEqual(len(cropped), 2)
        for k, v in cropped:
            self.assertEqual(k.shape, (1, 4, 18, 16))
            self.assertEqual(v.shape, (1, 4, 18, 16))


if __name__ == "__main__":
    unittest.main()
