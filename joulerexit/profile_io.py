"""Load the frozen NVML residual saved by the probe."""

from __future__ import annotations

import json
from pathlib import Path

from joulerexit.residual import EnergyProfile


def load_profile(path: str | Path) -> EnergyProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    coef = data["coefficients"]
    profile = EnergyProfile(
        intercept=float(coef["intercept"]),
        depth_slope=float(coef["depth_slope"]),
        log_length_slope=float(coef["log_length_slope"]),
        decode_offset=float(coef["decode_offset"]),
        depth_only_intercept=float(coef["depth_only_intercept"]),
        depth_only_slope=float(coef["depth_only_slope"]),
        depth_only_rmse=float(data["depth_only_rmse_j"]),
        full_rmse=float(data["full_rmse_j"]),
        repeat_noise_j=float(data["repeat_noise_j"]),
    )
    if not profile.residual_is_real:
        raise ValueError(f"Probe {path} says the residual is noise. Training would only learn a depth penalty.")
    return profile


def energy_scale(profile: EnergyProfile, exits: tuple[int, ...], seq_len: int = 64) -> float:
    """Scale joules so the first-to-last exit gap is a fraction of a nat.

Decode joules are almost flat in depth on the 4060. The gap, not the
large decode offset, is what can change an exit decision. The target
gap is 1.5 nats, large enough that easy tokens can leave early while
a wrong early exit still costs more in next-token loss.
    """
    early = profile.predicted_joules(exits[0], seq_len, "decode")
    late = profile.predicted_joules(exits[-1], seq_len, "decode")
    gap = max(late - early, 1e-6)
    return 1.5 / gap
