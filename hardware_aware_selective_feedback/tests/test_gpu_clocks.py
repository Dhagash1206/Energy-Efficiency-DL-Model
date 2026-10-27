import subprocess
import unittest
from unittest.mock import patch

from hardware_aware_selective_feedback.gpu_clocks import pinned_gpu


class FakeNvidiaSmi:
    """Records nvidia-smi calls; optionally fails on a given flag."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, command, **kwargs):
        arguments = command[1:]
        self.calls.append(arguments)
        if self.fail_on in arguments:
            raise subprocess.CalledProcessError(1, command, stderr="Insufficient Permissions")
        stdout = "80.00\n" if "--query-gpu=power.limit" in arguments else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


@patch("hardware_aware_selective_feedback.gpu_clocks.shutil.which", return_value="nvidia-smi")
class GpuClockTests(unittest.TestCase):
    def test_nothing_requested_runs_nothing(self, _which):
        runner = FakeNvidiaSmi()
        with pinned_gpu(runner=runner) as applied:
            self.assertEqual(applied, {})
        self.assertEqual(runner.calls, [])

    def test_applies_and_restores_settings(self, _which):
        runner = FakeNvidiaSmi()
        with pinned_gpu(1500, 60, runner=runner) as applied:
            self.assertEqual(applied["graphics_clock_mhz"], 1500)
            self.assertEqual(applied["original_power_limit_w"], 80.0)
        self.assertIn(["-i", "0", "-pl", "60"], runner.calls)
        self.assertIn(["-i", "0", "-lgc", "1500,1500"], runner.calls)
        self.assertEqual(runner.calls[-2:], [["-i", "0", "-rgc"], ["-i", "0", "-pl", "80"]])

    def test_restores_power_limit_when_clock_lock_fails(self, _which):
        runner = FakeNvidiaSmi(fail_on="-lgc")
        with self.assertRaisesRegex(RuntimeError, "administrator/root"):
            with pinned_gpu(1500, 60, runner=runner):
                self.fail("block must not run when pinning fails")
        self.assertEqual(runner.calls[-1], ["-i", "0", "-pl", "80"])
        self.assertNotIn(["-i", "0", "-rgc"], runner.calls)

    def test_restores_after_error_inside_block(self, _which):
        runner = FakeNvidiaSmi()
        with self.assertRaises(ValueError):
            with pinned_gpu(1500, runner=runner):
                raise ValueError("experiment failed")
        self.assertEqual(runner.calls[-1], ["-i", "0", "-rgc"])

    def test_missing_driver_is_reported(self, _which):
        _which.return_value = None
        with self.assertRaisesRegex(RuntimeError, "nvidia-smi was not found"):
            with pinned_gpu(1500, runner=FakeNvidiaSmi()):
                pass


if __name__ == "__main__":
    unittest.main()
