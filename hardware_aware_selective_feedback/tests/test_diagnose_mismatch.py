import json
import tempfile
import unittest
from pathlib import Path

from hardware_aware_selective_feedback.diagnose_mismatch import cases_from_report


class ReportCaseTests(unittest.TestCase):
    def _cases(self, report):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            return cases_from_report(path)

    def test_run_report_mismatches_are_deduplicated(self):
        mismatch = {"prompt_id": "gsm8k-test-1", "depth": 4, "draft_length": 3, "token_index": 9}
        cases, prompt_set, limit, precision = self._cases({
            "arguments": {"calibration": 8, "test": 16, "seed": 42, "tokens": 64, "precision": "fp16"},
            "mismatches": [mismatch, {**mismatch, "token_index": 12}],
        })
        self.assertEqual(cases, [{"prompt_id": "gsm8k-test-1", "mismatch_token": 9, "depth": 4, "draft": 3}])
        self.assertEqual(prompt_set, {"calibration": 8, "test": 16, "seed": 42})
        self.assertEqual((limit, precision), (64, "fp16"))

    def test_sweep_report_uses_one_calibration_prompt(self):
        _, prompt_set, _, _ = self._cases({
            "arguments": {"test": 4, "seed": 44, "tokens": 16, "precision": "fp16"},
            "mismatches": [],
        })
        self.assertEqual(prompt_set["calibration"], 1)


if __name__ == "__main__":
    unittest.main()
