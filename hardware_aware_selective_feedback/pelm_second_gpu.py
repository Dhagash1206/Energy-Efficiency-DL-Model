"""Refuse to train or score PELM on a machine that is not the RTX 3050.

The 4060 sweep and the 4060 network are not copied. Run this on the 3050
after its own startup sweep:

    python -m hardware_aware_selective_feedback.pelm_second_gpu
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

STATUS = Path("hardware_aware_selective_feedback/results/rtx3050-pelm-status.json")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    gpu_name = torch.cuda.get_device_name(0)
    if "3050" not in gpu_name:
        status = {
            "status": "not_measured",
            "gpu_visible": gpu_name,
            "required": "NVIDIA GeForce RTX 3050",
            "reason": "The RTX 3050 is not in this machine. The 4060 profile and network are not a 3050 result.",
            "copied_4060_profile": False,
            "copied_4060_network": False,
            "command": "python -m hardware_aware_selective_feedback.pelm_second_gpu",
        }
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise SystemExit(3)
    raise SystemExit(
        "The 3050 is present. Run its own startup sweep, then train a new network on that profile. "
        "Do not load the 4060 checkpoint."
    )


if __name__ == "__main__":
    main()
