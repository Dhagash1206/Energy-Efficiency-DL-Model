"""Check whether the local machine can run the NVIDIA RTX 3050 path."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys


def check_cuda() -> dict:
    result = {
        "cuda_available": False,
        "device_name": None,
        "device_count": 0,
        "torch_version": None,
        "driver_version": None,
        "notes": [],
    }
    if importlib.util.find_spec("torch") is None:
        result["notes"].append("PyTorch is not installed yet.")
        return result

    import torch

    result["torch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    if not result["cuda_available"]:
        result["notes"].append("CUDA is not available. Install the CUDA-enabled PyTorch build and an NVIDIA driver.")
        return result

    result["device_count"] = torch.cuda.device_count()
    result["device_name"] = torch.cuda.get_device_name(0)
    result["driver_version"] = torch.version.cuda
    result["notes"].append("CUDA is available. The project can target NVIDIA GPUs such as the RTX 3050.")
    return result


def main() -> None:
    print("Environment check for RTX 3050 CUDA support")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"nvidia-smi present: {bool(shutil.which('nvidia-smi'))}")

    cuda = check_cuda()
    print(f"CUDA available: {cuda['cuda_available']}")
    print(f"Torch version: {cuda['torch_version']}")
    print(f"CUDA runtime: {cuda['driver_version']}")
    print(f"Device count: {cuda['device_count']}")
    print(f"Device name: {cuda['device_name']}")

    for note in cuda["notes"]:
        print(f"- {note}")

    if not cuda["cuda_available"]:
        print("")
        print("Install the RTX 3050 path with:")
        print("  powershell -ExecutionPolicy Bypass -File scripts\\setup_cuda_3050.ps1")
        print("")
        print("Then verify with:")
        print("  python -m prototype.doctor")


if __name__ == "__main__":
    main()
