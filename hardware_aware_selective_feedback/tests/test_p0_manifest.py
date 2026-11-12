import tempfile
import unittest
from pathlib import Path

from hardware_aware_selective_feedback.protocol import (
    SplitConfig,
    compute_prompt_hash,
    get_known_exposed_ids,
    load_dataset_splits,
)
from hardware_aware_selective_feedback.run_manifest import (
    assert_unique_output_path,
    build_exposure_registry,
    create_results_manifest,
    get_environment_provenance,
    verify_environment,
)


class TestPhaseP0Manifest(unittest.TestCase):
    def test_environment_provenance(self):
        env = get_environment_provenance()
        self.assertIn("timestamp_utc", env)
        self.assertIn("platform", env)
        self.assertIn("git", env)
        git = env["git"]
        self.assertIn("commit", git)
        self.assertIn("dirty", git)
        self.assertIn("diff_sha256", git)
        self.assertEqual(len(git["diff_sha256"]), 64)
        self.assertIn("working_tree_sha256", git)
        self.assertEqual(len(git["working_tree_sha256"]), 64)
        self.assertIn("untracked_files", git)
        self.assertGreater(git["untracked_files_count"], 0)

        # Verification passes on self
        matches, mismatches = verify_environment(env)
        self.assertTrue(matches)
        self.assertEqual(mismatches, [])

        # Verification fails on mismatch
        fake_manifest = {"git": {"commit": "0000000000000000000000000000000000000000"}}
        matches, mismatches = verify_environment(fake_manifest)
        self.assertFalse(matches)
        self.assertGreater(len(mismatches), 0)

    def test_results_manifest_and_unique_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = tmp_path / "sub" / "report1.json"
            f1.parent.mkdir(parents=True)
            f1.write_text('{"status": "ok"}', encoding="utf-8")

            manifest = create_results_manifest(tmp_path)
            self.assertIn("sub/report1.json", manifest["files"])
            entry = manifest["files"]["sub/report1.json"]
            self.assertIn("sha256", entry)
            self.assertEqual(entry["extension"], ".json")

            # Unique path assertion: existing file raises FileExistsError
            with self.assertRaises(FileExistsError):
                assert_unique_output_path(f1)

            # File matching manifest filename raises ValueError
            fresh_f1 = tmp_path / "other" / "report1.json"
            with self.assertRaises(ValueError):
                assert_unique_output_path(fresh_f1, historical_manifest=manifest)

            # Fresh unique path passes
            fresh_unique = tmp_path / "fresh_report.json"
            assert_unique_output_path(fresh_unique, historical_manifest=manifest)

    def test_exposure_registry(self):
        registry = build_exposure_registry()
        counts = registry["summary"]["counts"]
        # Audit established 22 HumanEval, 17 GSM8K-test, 8 GSM8K-train in the 29 reports
        self.assertEqual(counts["humaneval"], 22)
        self.assertEqual(counts["gsm8k_test"], 17)
        self.assertEqual(counts["gsm8k_train"], 8)
        self.assertEqual(counts["total_exposed"], 47)

        exposed_ids = get_known_exposed_ids()
        self.assertEqual(len(exposed_ids), 47)
        self.assertIn("HumanEval/154", exposed_ids)
        self.assertIn("gsm8k-test-208", exposed_ids)

    def test_decoupled_prompt_loader_independent_counts(self):
        cfg = SplitConfig(
            seed=123,
            gsm8k_calibration=3,
            humaneval_calibration=1,
            gsm8k_test=4,
            humaneval_test=2,
            fewshot_k=2,
        )
        splits = load_dataset_splits(cfg)
        calib = splits["calibration_prompts"]
        test = splits["test_prompts"]
        exemplars = splits["exemplars"]

        self.assertEqual(len(calib), 4)
        self.assertEqual(sum(1 for r in calib if r["dataset"] == "gsm8k"), 3)
        self.assertEqual(sum(1 for r in calib if r["dataset"] == "humaneval"), 1)

        self.assertEqual(len(test), 6)
        self.assertEqual(sum(1 for r in test if r["dataset"] == "gsm8k"), 4)
        self.assertEqual(sum(1 for r in test if r["dataset"] == "humaneval"), 2)

        self.assertEqual(len(exemplars), 2)

        # Check prompt hashes exist and are valid SHA256
        for row in calib + test + exemplars:
            self.assertIn("prompt_hash", row)
            self.assertEqual(row["prompt_hash"], compute_prompt_hash(row["text"]))

    def test_five_shot_gsm8k_prompt_rendering(self):
        cfg = SplitConfig(
            seed=42,
            gsm8k_test=2,
            humaneval_test=1,
            fewshot_k=5,
        )
        splits = load_dataset_splits(cfg)
        exemplars = splits["exemplars"]
        self.assertEqual(len(exemplars), 5)

        gsm_tests = [r for r in splits["test_prompts"] if r["dataset"] == "gsm8k"]
        self.assertEqual(len(gsm_tests), 2)
        for t in gsm_tests:
            # Must contain all 5 exemplar questions in order
            for ex in exemplars:
                self.assertIn(ex["text"], t["text"])
            # And finish with the target question
            self.assertTrue(t["text"].endswith("\nAnswer:"))

        # HumanEval must not have fewshot exemplars prepended
        he_tests = [r for r in splits["test_prompts"] if r["dataset"] == "humaneval"]
        self.assertEqual(len(he_tests), 1)
        self.assertNotIn("gsm8k-train", he_tests[0]["text"])

    def test_prompt_reproducibility(self):
        cfg = SplitConfig(
            seed=42,
            gsm8k_calibration=2,
            humaneval_calibration=2,
            gsm8k_test=3,
            humaneval_test=3,
        )
        run1 = load_dataset_splits(cfg)
        run2 = load_dataset_splits(cfg)

        calib1_ids = [r["id"] for r in run1["calibration_prompts"]]
        calib2_ids = [r["id"] for r in run2["calibration_prompts"]]
        self.assertEqual(calib1_ids, calib2_ids)

        test1_hashes = [r["prompt_hash"] for r in run1["test_prompts"]]
        test2_hashes = [r["prompt_hash"] for r in run2["test_prompts"]]
        self.assertEqual(test1_hashes, test2_hashes)

    def test_disjoint_check_and_duplicate_rejection(self):
        cfg = SplitConfig(
            seed=42,
            calibration_ids=["HumanEval/0"],
            test_ids=["HumanEval/0"],
            strict_disjoint_check=True,
        )
        with self.assertRaisesRegex(ValueError, "Strict disjoint check failed"):
            load_dataset_splits(cfg)

    def test_explicit_id_selection(self):
        cfg = SplitConfig(
            calibration_ids=["HumanEval/1", "HumanEval/2"],
            test_ids=["HumanEval/3", "HumanEval/4"],
        )
        splits = load_dataset_splits(cfg)
        self.assertEqual(
            [r["id"] for r in splits["calibration_prompts"]], ["HumanEval/1", "HumanEval/2"]
        )
        self.assertEqual([r["id"] for r in splits["test_prompts"]], ["HumanEval/3", "HumanEval/4"])


if __name__ == "__main__":
    unittest.main()
