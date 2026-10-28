"""NVIDIA GPU energy from the NVML cumulative counter.

The counter is millijoules for the GPU only. It does not include the CPU
or the rest of the laptop.
"""

from __future__ import annotations

import pynvml


class NvmlEnergy:
    """Read cumulative GPU energy in joules for one NVIDIA device."""

    def __init__(self, index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.name = pynvml.nvmlDeviceGetName(self.handle)
        # Fail at construction when the counter is missing.
        self.read_millijoules()

    def read_millijoules(self) -> int:
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def read_joules(self) -> float:
        return self.read_millijoules() / 1000.0


class EnergyWindow:
    """Joules consumed between entering and exiting the window."""

    def __init__(self, meter: NvmlEnergy):
        self.meter = meter
        self.start_mj = 0
        self.joules = 0.0

    def __enter__(self) -> "EnergyWindow":
        self.start_mj = self.meter.read_millijoules()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.joules = (self.meter.read_millijoules() - self.start_mj) / 1000.0
