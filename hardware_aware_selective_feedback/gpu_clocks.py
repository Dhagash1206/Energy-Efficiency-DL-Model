"""Optional NVIDIA clock and power-limit pinning for repeatable energy benchmarks.

Boost clocks drift with temperature, so the same configuration can draw
different energy across runs. Locking the graphics clock (``nvidia-smi -lgc``)
and/or capping board power (``nvidia-smi -pl``) removes most of that drift and
lets a sweep compare energy per token at a fixed operating point.

Both settings need administrator (Windows) or root (Linux) rights, and some
laptop GPUs reject them. Settings are always restored on exit, even on error.
"""
from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from typing import Callable, Generator, Optional


def _nvidia_smi(runner: Callable, *arguments: str) -> str:
    """Run one nvidia-smi command and return stdout; raise a clear error on failure."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi was not found; GPU clock pinning requires an NVIDIA driver.")
    try:
        completed = runner([executable, *arguments], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"nvidia-smi {' '.join(arguments)} failed ({detail or error.returncode}). "
            "Run as administrator/root, or check that this GPU supports the setting."
        ) from error
    return completed.stdout


@contextmanager
def pinned_gpu(
    graphics_clock_mhz: Optional[int] = None,
    power_limit_w: Optional[float] = None,
    gpu_index: int = 0,
    runner: Callable = subprocess.run,
) -> Generator[dict, None, None]:
    """Lock the GPU graphics clock and/or power limit for the duration of the block.

    Yields a dict describing what was applied (empty when nothing was requested),
    suitable for storing in an experiment report.
    """
    if graphics_clock_mhz is None and power_limit_w is None:
        yield {}
        return

    index = str(gpu_index)
    applied: dict = {"gpu_index": gpu_index}
    original_power_limit_w: Optional[float] = None
    clock_locked = False
    try:
        if power_limit_w is not None:
            original_power_limit_w = float(_nvidia_smi(
                runner, "-i", index, "--query-gpu=power.limit", "--format=csv,noheader,nounits"
            ).strip())
            _nvidia_smi(runner, "-i", index, "-pl", f"{power_limit_w:g}")
            applied.update(power_limit_w=power_limit_w,
                           original_power_limit_w=original_power_limit_w)
        if graphics_clock_mhz is not None:
            _nvidia_smi(runner, "-i", index, "-lgc", f"{graphics_clock_mhz},{graphics_clock_mhz}")
            clock_locked = True
            applied["graphics_clock_mhz"] = graphics_clock_mhz
        yield applied
    finally:
        # Restore every setting that was changed, even if a later step failed.
        restore_errors = []
        if clock_locked:
            try:
                _nvidia_smi(runner, "-i", index, "-rgc")
            except RuntimeError as error:
                restore_errors.append(str(error))
        if original_power_limit_w is not None:
            try:
                _nvidia_smi(runner, "-i", index, "-pl", f"{original_power_limit_w:g}")
            except RuntimeError as error:
                restore_errors.append(str(error))
        if restore_errors:
            raise RuntimeError("Could not restore GPU settings: " + "; ".join(restore_errors))
