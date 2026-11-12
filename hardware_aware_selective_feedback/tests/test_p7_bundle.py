import json
import tempfile
import unittest
from pathlib import Path

from hardware_aware_selective_feedback.reproducibility_bundle import (
    audit_publication_readiness,
    generate_reproducibility_bundle,
)


class TestPhaseP7ReproducibilityBundle(unittest.TestCase):
    def test_audit_publication_readiness_valid(self):
        valid_report = {
            "environment_provenance": {
                "git": {"commit": "abc", "working_tree_sha256": "1234567890abcdef"},
                "torch": {"version": "2.9.0"},
            },
            "rows": [
                {
                    "prompt_id": "p1",
                    "policy": "fixed",
                    "split": "test",
                    "tokens": [1, 2],
                    "greedy_match": True,
                }
            ],
            "aggregates": {"fixed": {"greedy_mismatches": 0, "all_greedy_match": True}},
            "publication_break_even": {
                "fixed": {"calibration_energy_j": 10.0, "saving_per_request_j": 1.0}
            },
        }
        audit = audit_publication_readiness(valid_report)
        self.assertTrue(audit["passed_all_gates"])
        self.assertEqual(audit["failures"], [])

    def test_regression_rejects_empty_provenance_and_missing_calibration(self):
        # Empty provenance, invalid row, unhandled mismatch, empty calibration
        bad_report = {
            "environment_provenance": {},  # Empty provenance
            "rows": [{"corrupt_key": 1}],  # Invalid row
            "aggregates": {
                "adaptive": {"greedy_mismatches": 2, "all_greedy_match": True}  # False match claim
            },
            "publication_break_even": {},  # Empty calibration
        }
        audit = audit_publication_readiness(bad_report)
        self.assertFalse(audit["passed_all_gates"])
        self.assertFalse(audit["gates"]["environment_provenance"])
        self.assertFalse(audit["gates"]["valid_request_rows"])
        self.assertFalse(audit["gates"]["exact_output_validity"])
        self.assertFalse(audit["gates"]["calibration_accounted"])
        self.assertGreaterEqual(len(audit["failures"]), 4)

    def test_generate_reproducibility_bundle_marks_unmet_items(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            report_file = tmp_path / "incomplete_report.json"
            # Report missing calibration accounting
            report_data = {
                "environment_provenance": {
                    "git": {"commit": "abc", "diff_sha256": "def"},
                    "platform": {},
                },
                "rows": [{"prompt_id": "p1", "policy": "fixed", "split": "test", "tokens": [1]}],
                "aggregates": {"fixed": {"greedy_mismatches": 0, "all_greedy_match": True}},
                "publication_break_even": {},  # Missing!
            }
            report_file.write_text(json.dumps(report_data), encoding="utf-8")

            bundle_dir = tmp_path / "bundle"
            manifest = generate_reproducibility_bundle(report_file, bundle_dir)

            self.assertFalse(manifest["audit"]["passed_all_gates"])
            md_content = (bundle_dir / "reproducibility_summary.md").read_text(encoding="utf-8")
            self.assertIn("REJECTED / INCOMPLETE", md_content)
            # The calibration_accounted checklist item must be UNCHECKED
            self.assertIn("- [ ] **calibration_accounted**", md_content)


if __name__ == "__main__":
    unittest.main()
