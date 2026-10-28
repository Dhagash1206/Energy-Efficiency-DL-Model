"""Repeat the probe and the held-out measurement on the teammate RTX 3050.

This machine's GPU is the RTX 4060. The script stops unless the visible
NVIDIA device name contains 3050, so a 4060 run cannot be saved as the
second device.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from joulerexit.energy import NvmlEnergy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="joulerexit/checkpoints")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    meter = NvmlEnergy(0)
    status_path = Path("joulerexit/results/rtx3050-status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if "3050" not in meter.name:
        status = {
            "status": "not_measured",
            "gpu_visible": meter.name,
            "required": "NVIDIA GeForce RTX 3050",
            "reason": "The RTX 3050 is not in this machine. Copy the checkpoints and run this same command there.",
            "command": "python -m joulerexit.device_repeat --checkpoint-dir joulerexit/checkpoints",
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        raise SystemExit(3)

    probe_out = Path("joulerexit/results/rtx3050-smollm-probe.json")
    probe = subprocess.run(
        [
            sys.executable, "-m", "joulerexit.probe",
            "--model", "HuggingFaceTB/SmolLM2-135M",
            "--output", str(probe_out),
        ],
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(probe.returncode)
    eval_out = Path("joulerexit/results/rtx3050-eval.json")
    evaluation = subprocess.run(
        [
            sys.executable, "-m", "joulerexit.evaluate",
            "--profile", str(probe_out),
            "--checkpoint-dir", args.checkpoint_dir,
            "--output", str(eval_out),
            "--limit", str(args.limit),
            "--repeats", str(args.repeats),
        ],
        check=False,
    )
    status = {
        "status": "measured" if evaluation.returncode == 0 else "eval_failed",
        "gpu_visible": meter.name,
        "probe": str(probe_out),
        "eval": str(eval_out),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    raise SystemExit(evaluation.returncode)


if __name__ == "__main__":
    main()
