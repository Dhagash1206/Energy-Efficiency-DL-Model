"""Capability detection without requiring PyTorch for portable demo mode."""

import ctypes
import importlib.util
import platform
import shutil
import sys


def detect():
    result = {"os": platform.platform(), "python": sys.version.split()[0],
              "python_executable": sys.executable, "cpu": platform.processor(),
              "free_disk_gb": round(shutil.disk_usage(".").free / 1e9, 2),
              "ram_gb": None, "backends": {"simulation": True, "cpu": False,
              "cuda": False, "mps": False, "xpu": False}, "accelerators": [], "notes": []}
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["ram_gb"] = round(status.total_phys / 1024**3, 1)
    if "msys" in sys.executable.lower() or "mingw" in sys.executable.lower():
        result["notes"].append("MSYS/MinGW Python detected. Use standard CPython for official PyTorch wheels.")
    if importlib.util.find_spec("torch") is None:
        result["notes"].append("PyTorch is absent. Simulation works; real model inference needs optional dependencies.")
        return result
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["backends"]["cpu"] = True
        for name in ("cuda", "xpu"):
            api = getattr(torch, name, None)
            available = bool(api and api.is_available())
            result["backends"][name] = available
            if available:
                for index in range(api.device_count()):
                    result["accelerators"].append({"backend": name, "index": index,
                                                   "name": api.get_device_name(index)})
        mps = getattr(torch.backends, "mps", None)
        result["backends"]["mps"] = bool(mps and mps.is_available())
        if result["backends"]["mps"]:
            result["accelerators"].append({"backend": "mps", "name": "Apple Metal"})
    except Exception as exc:
        result["notes"].append(f"PyTorch capability query failed: {exc}")
    result["notes"].append("Availability is not a benchmark validation. Intel Arc detection requires an XPU-enabled PyTorch build.")
    return result


def select_device(requested, capabilities):
    if requested == "auto":
        for name in ("cuda", "mps", "xpu", "cpu"):
            if capabilities["backends"].get(name):
                return name
        raise RuntimeError("No real backend available. Run simulate or install optional dependencies.")
    if not capabilities["backends"].get(requested):
        raise RuntimeError(f"Requested backend {requested!r} is unavailable; run python -m prototype doctor")
    return requested
