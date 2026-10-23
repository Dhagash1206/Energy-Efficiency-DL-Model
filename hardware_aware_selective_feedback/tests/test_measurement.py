import tempfile
import unittest
from pathlib import Path

from hardware_aware_selective_feedback.measurement import Meter


class MeasurementTests(unittest.TestCase):
    def test_rapl_package_wrap_and_no_nested_double_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "intel-rapl" / "package0"
            package.mkdir(parents=True)
            (package / "name").write_text("package-0")
            (package / "max_energy_range_uj").write_text("1000000")
            energy = package / "energy_uj"
            energy.write_text("950000")
            nested = package / "core0"
            nested.mkdir()
            (nested / "energy_uj").write_text("900000")
            (nested / "max_energy_range_uj").write_text("1000000")
            meter = Meter("cpu", energy_source="rapl", rapl_root=root)
            self.assertTrue(meter.available)
            self.assertEqual(meter.describe()["rapl_zones"], ["package-0"])
            def work():
                energy.write_text("100000")
                return 7
            value, measured = meter.run(work)
            self.assertEqual(value, 7)
            self.assertAlmostEqual(measured["energy_j"], 0.15)
            self.assertEqual(measured["boundary"], "cpu_packages")

    def test_external_counter_requires_explicit_scope_and_monotonicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counter.txt"
            path.write_text("1000")
            with self.assertRaises(ValueError):
                Meter("cpu", energy_source="file", counter_file=path)
            meter = Meter("cpu", energy_source="file", counter_file=path,
                          counter_unit="millijoules", energy_scope="whole_device")
            def work():
                path.write_text("1250")
            _, measured = meter.run(work)
            self.assertAlmostEqual(measured["energy_j"], 0.25)
            self.assertEqual(measured["boundary"], "whole_device")
            path.write_text("1200")
            with self.assertRaises(RuntimeError):
                meter.run(lambda: path.write_text("1100"))

    def test_missing_counter_is_null_with_reason(self):
        meter = Meter("cpu", energy_source="none")
        _, measured = meter.run(lambda: 1)
        self.assertIsNone(measured["energy_j"])
        self.assertEqual(measured["boundary"], "unavailable")
        self.assertTrue(measured["energy_reason"])


if __name__ == "__main__":
    unittest.main()
