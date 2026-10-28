"""Fit the 4060 sweep per exit and train the per-round network.

The locked draft remains the result when one action wins every step.
Example:
    python -m hardware_aware_selective_feedback.train_pelm
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hardware_aware_selective_feedback.pelm_round import teach, train_network

PROFILE = Path("hardware_aware_selective_feedback/results/nvidia-geforce-rtx-4060-laptop-gpu-device-profile.json")
TEACHER = Path("hardware_aware_selective_feedback/results/rtx4060-pelm-teacher.json")
CHECKPOINT = Path("hardware_aware_selective_feedback/checkpoints/pelm-round.pt")


def _public_teacher(teacher: dict) -> dict:
    fits = {
        str(depth): {key: float(value) for key, value in fit.items() if key != "rows"}
        for depth, fit in teacher["fits"].items()
    }
    return {
        "varies": teacher["varies"],
        "reason": teacher["reason"],
        "teacher_depth": teacher["teacher_depth"],
        "fits": fits,
        "speeds": teacher["speeds"],
        "speed_floor": teacher["speed_floor"],
        "shared_verify_j": teacher["shared_verify_j"],
        "repeat_noise_j": teacher["repeat_noise_j"],
        "locked": teacher["locked"],
    }


def main():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    teacher = teach(profile)
    public = _public_teacher(teacher)
    TEACHER.parent.mkdir(parents=True, exist_ok=True)
    if not teacher["varies"]:
        public["trained"] = False
        TEACHER.write_text(json.dumps(public, indent=2), encoding="utf-8")
        print(public["reason"])
        return
    net, record = train_network(teacher)
    public["trained"] = True
    public["train"] = record
    TEACHER.write_text(json.dumps(public, indent=2), encoding="utf-8")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": net.state_dict(),
            "actions": net.actions,
            "train": record,
            "repeat_noise_j": teacher["repeat_noise_j"],
            "locked": teacher["locked"],
        },
        CHECKPOINT,
    )
    print(json.dumps(record, indent=2))
    print(f"wrote {TEACHER}")
    print(f"wrote {CHECKPOINT}")


if __name__ == "__main__":
    main()
