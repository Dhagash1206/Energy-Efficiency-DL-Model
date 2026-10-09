"""Measured request latency and optional cumulative hardware energy counters.

A counter's scope is explicit. CPU package and GPU energy must not be described
as whole-device energy, and missing sensors never become estimated joules.
"""
import argparse
import ctypes
import json
import platform
import struct
import time
from pathlib import Path

import torch


class Meter:
    def __init__(self, device, energy_source="auto", counter_file=None,
                 counter_unit="joules", energy_scope=None, rapl_root=None):
        self.device = torch.device(device)
        self.backend = None
        self.handle = None
        self.nvml = None
        self.zones = ()
        self.emi_channels = ()
        self.emi_channel_names = ()
        self.counter_file = None
        self.unit_scale = None
        self.scope = None
        self.source = None
        self.reason = None
        if energy_source not in ("auto", "none", "nvml", "rapl", "emi", "file"):
            raise ValueError("Unknown energy source")
        if energy_source == "file" or counter_file is not None:
            if counter_file is None or not energy_scope:
                raise ValueError("A counter file requires its path and measured scope")
            units = {"joules": 1.0, "millijoules": 1e-3, "microjoules": 1e-6}
            if counter_unit not in units:
                raise ValueError("Unsupported counter unit")
            self.counter_file = Path(counter_file)
            self.unit_scale = units[counter_unit]
            self.scope = energy_scope
            self.source = "cumulative counter file: " + str(self.counter_file)
            self.backend = "file"
            self._validate_counter()
        elif energy_source == "none":
            self.reason = "Energy measurement disabled"
        elif energy_source == "nvml" or (energy_source == "auto" and self.device.type == "cuda"):
            self._init_nvml()
        elif energy_source == "rapl" or (energy_source == "auto" and self.device.type == "cpu"
                                          and platform.system() == "Linux"):
            self._init_rapl(Path(rapl_root) if rapl_root is not None
                            else Path("/sys/class/powercap"))
        elif energy_source == "emi" or (energy_source == "auto" and self.device.type == "cpu"
                                         and platform.system() == "Windows"):
            self._init_emi()
        else:
            self.reason = "No supported cumulative energy counter for this device/OS"

    @property
    def available(self):
        return self.backend is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _validate_counter(self):
        try:
            self.counter()
        except (OSError, ValueError) as error:
            self.backend = None
            self.reason = "Counter unreadable: " + str(error)

    def _init_nvml(self):
        if self.device.type != "cuda":
            self.reason = "NVML measures a CUDA NVIDIA GPU, not this device"
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.backend = "nvml"
            self.scope = "nvidia_gpu"
            self.source = "NVML cumulative GPU energy counter (mJ)"
        except Exception as error:
            self.reason = "NVML counter unavailable: " + str(error)
            self.handle = None

    def _init_rapl(self, root):
        if self.device.type != "cpu":
            self.reason = "RAPL package energy cannot represent this accelerator's total energy"
            return
        try:
            # Only top-level package zones. Adding nested core/DRAM zones would
            # double-count energy already included by a parent package zone.
            zones = []
            for control in (root / "intel-rapl", root / "amd-rapl"):
                if not control.is_dir():
                    continue
                for zone in control.iterdir():
                    if not zone.is_dir():
                        continue
                    energy = zone / "energy_uj"
                    maximum = zone / "max_energy_range_uj"
                    if energy.is_file() and maximum.is_file():
                        name = (zone / "name").read_text().strip() if (zone / "name").is_file() else zone.name
                        zones.append((energy, int(maximum.read_text().strip()), name))
            if not zones:
                raise FileNotFoundError("no readable top-level RAPL package zones")
            self.zones = tuple(zones)
            for energy, maximum, _ in self.zones:
                if maximum <= 0 or not 0 <= int(energy.read_text().strip()) < maximum:
                    raise ValueError("invalid RAPL counter or range")
            self.backend = "rapl"
            self.scope = "cpu_packages"
            self.source = "Linux powercap RAPL package energy_uj"
        except (OSError, ValueError) as error:
            self.reason = "RAPL counter unavailable: " + str(error)

    @staticmethod
    def _emi_packages(metadata):
        if len(metadata) < 68:
            raise ValueError("short EMI V2 metadata")
        count = struct.unpack_from("<H", metadata, 66)[0]
        offset, packages = 68, []
        for index in range(count):
            if offset + 6 > len(metadata):
                raise ValueError("truncated EMI channel")
            unit, length = struct.unpack_from("<IH", metadata, offset)
            offset += 6
            if unit != 0 or length < 2 or length % 2 or offset + length > len(metadata):
                raise ValueError("invalid EMI channel")
            name = metadata[offset:offset + length].decode("utf-16-le").split("\0", 1)[0]
            if name.endswith("_PKG"):
                packages.append((index, name))
            offset += length
        return count, packages

    def _emi_ioctl(self, handle, code, size):
        output = ctypes.create_string_buffer(size)
        returned = ctypes.c_uint32()
        if not self.emi_kernel.DeviceIoControl(handle, code, None, 0, output, size,
                                               ctypes.byref(returned), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if returned.value != size:
            raise ValueError("unexpected EMI response size")
        return output.raw

    def _init_emi(self):
        if self.device.type != "cpu" or platform.system() != "Windows":
            self.reason = "Windows EMI package counter requires Windows CPU inference"
            return
        try:
            import winreg
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_void_p]
            kernel.CreateFileW.restype = ctypes.c_void_p
            kernel.DeviceIoControl.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                               ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                                               ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
            kernel.DeviceIoControl.restype = ctypes.c_int
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            self.emi_kernel = kernel
            path = ("SYSTEM\\CurrentControlSet\\Control\\DeviceClasses\\"
                    "{45bd8344-7ed6-49cf-a440-c276c933b053}")
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                names, i = [], 0
                while True:
                    try:
                        names.append(winreg.EnumKey(key, i))
                        i += 1
                    except OSError:
                        break
            for name in names:
                if not name.startswith("##?#"):
                    continue
                handle = kernel.CreateFileW("\\\\?\\" + name[4:], 0x80000000, 3,
                                            None, 3, 0, None)
                if handle in (None, ctypes.c_void_p(-1).value):
                    continue
                try:
                    if struct.unpack("<H", self._emi_ioctl(handle, 0x224000, 2))[0] != 2:
                        continue
                    size = struct.unpack("<I", self._emi_ioctl(handle, 0x224004, 4))[0]
                    if not 68 <= size <= 1_000_000:
                        continue
                    count, packages = self._emi_packages(self._emi_ioctl(handle, 0x224008, size))
                    if not packages:
                        continue
                    self._emi_ioctl(handle, 0x22400C, count * 16)
                    self.emi_channels = tuple((handle, count, index, label)
                                              for index, label in packages)
                    self.emi_channel_names = tuple(label for _, label in packages)
                    self.backend, self.scope = "emi", "cpu_packages"
                    self.source = "Windows EMI CPU package cumulative energy (pWh)"
                    return
                finally:
                    if not self.emi_channels:
                        kernel.CloseHandle(handle)
            raise FileNotFoundError("no readable EMI V2 CPU package channel")
        except (OSError, ValueError, struct.error) as error:
            self.reason = "Windows EMI counter unavailable: " + str(error)

    def counter(self):
        if self.backend == "nvml":
            return int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        if self.backend == "rapl":
            return tuple(int(path.read_text().strip()) for path, _, _ in self.zones)
        if self.backend == "emi":
            samples = {}
            for handle, count, _, _ in self.emi_channels:
                if handle not in samples:
                    samples[handle] = self._emi_ioctl(handle, 0x22400C, count * 16)
            return tuple(struct.unpack_from("<Q", samples[handle], index * 16)[0]
                         for handle, _, index, _ in self.emi_channels)
        if self.backend == "file":
            value = float(self.counter_file.read_text(encoding="utf-8").strip())
            if not 0 <= value < float("inf"):
                raise ValueError("counter must be a finite, nonnegative number")
            return value
        return None

    def _joules(self, before, after):
        if self.backend == "nvml":
            if after < before:
                raise RuntimeError("NVML energy counter reset during request")
            return (after - before) / 1000
        if self.backend == "rapl":
            total = 0
            for previous, current, (_, maximum, _) in zip(before, after, self.zones):
                if not 0 <= current < maximum:
                    raise RuntimeError("RAPL counter outside its declared range")
                total += current - previous if current >= previous else maximum - previous + current
            return total / 1_000_000
        if self.backend == "emi":
            if any(current < previous for previous, current in zip(before, after)):
                raise RuntimeError("Windows EMI energy counter reset during request")
            return sum(current - previous for previous, current in zip(before, after)) * 3.6e-9
        if self.backend == "file":
            if after < before:
                raise RuntimeError("External energy counter decreased during request")
            return (after - before) * self.unit_scale
        return None

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "xpu":
            torch.xpu.synchronize(self.device)

    def run(self, function):
        self.sync()
        before = self.counter()
        start = time.perf_counter()
        try:
            result = function()
        finally:
            self.sync()
        elapsed = time.perf_counter() - start
        after = self.counter()
        return result, {
            "latency_s": elapsed,
            "energy_j": self._joules(before, after) if self.available else None,
            "boundary": self.scope if self.available else "unavailable",
            "energy_reason": self.reason,
            "source": self.source,
        }

    def describe(self):
        return {"device": str(self.device), "energy_available": self.available,
                "energy_boundary": self.scope, "energy_source": self.source,
                "energy_reason": self.reason,
                "rapl_zones": [name for _, _, name in self.zones],
                "emi_channels": list(self.emi_channel_names)}

    def close(self):
        if self.emi_channels:
            for handle in {channel[0] for channel in self.emi_channels}:
                self.emi_kernel.CloseHandle(handle)
            self.emi_channels = ()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None


def main():
    parser = argparse.ArgumentParser(description="Probe available energy measurement without loading a model")
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu", "mps"), default="cpu")
    parser.add_argument("--energy-source", choices=("auto", "none", "nvml", "rapl", "emi", "file"), default="auto")
    parser.add_argument("--counter-file")
    parser.add_argument("--counter-unit", choices=("joules", "millijoules", "microjoules"), default="joules")
    parser.add_argument("--energy-scope")
    args = parser.parse_args()
    meter = Meter(args.device, args.energy_source, args.counter_file, args.counter_unit, args.energy_scope)
    print(json.dumps(meter.describe(), indent=2))
    meter.close()


if __name__ == "__main__":
    main()
