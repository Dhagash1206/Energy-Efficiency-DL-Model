import unittest

import torch

from hardware_aware_selective_feedback.run import resolve_dtype


class ModelPrecisionTests(unittest.TestCase):
    def test_fp32_supported_on_cpu_and_cuda(self):
        self.assertIs(resolve_dtype("fp32", "cpu"), torch.float32)
        self.assertIs(resolve_dtype("fp32", "cuda"), torch.float32)

    def test_fp16_supported_on_cuda(self):
        self.assertIs(resolve_dtype("fp16", "cuda"), torch.float16)

    def test_fp16_rejected_on_cpu(self):
        with self.assertRaisesRegex(ValueError, "only on CUDA"):
            resolve_dtype("fp16", "cpu")

    def test_unknown_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported precision"):
            resolve_dtype("bf16", "cuda")


if __name__ == "__main__":
    unittest.main()
