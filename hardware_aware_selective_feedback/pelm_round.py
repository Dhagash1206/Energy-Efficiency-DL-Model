"""PELM's small network, taught by a startup-sweep joule fit.

Inputs are power, temperature, the clock the GPU is already running, the
speed of the last round, and how many draft tokens that round kept. The
clock is never written. A shallow check is an available action, but it is
not treated as cheaper than a full check until a measurement says so.
"""

from __future__ import annotations

import time
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from hardware_aware_selective_feedback.device_energy import (
    acceptance_steps,
    cheapest_changes,
    cycle_joules_per_token,
    fit_exit_profiles,
)

FULL_STOP = 16
SHALLOW_STOP = 12
THROTTLE_C = 80.0
FEATURE_COUNT = 5


def menu(n_layers: int = FULL_STOP) -> tuple[tuple[int, int, int], ...]:
    """Exit, draft length, and check depth. Full check is listed before the shallow one."""
    shallow = min(SHALLOW_STOP, n_layers - 1)
    actions = [(0, 0, n_layers), (0, 0, shallow)]
    for depth in (4, 8):
        if depth >= n_layers:
            continue
        for length in (1, 2, 4):
            actions.append((depth, length, n_layers))
            actions.append((depth, length, shallow))
    return tuple(actions)


def action_speeds(measured: list[dict]) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in measured:
        latency = float(row["latency_s"])
        if latency <= 0:
            continue
        key = (int(row["depth"]), int(row["length"]))
        grouped.setdefault(key, []).append(float(row["committed"]) / latency)
    if not grouped:
        raise ValueError("the sweep has no timed rows")
    return {key: float(sum(values) / len(values)) for key, values in grouped.items()}


def action_cost(
    fits: dict[int, dict],
    shared_verify: float,
    speeds: dict[tuple[int, int], float],
    floor: float,
    action: tuple[int, int, int],
    kept: int,
    temp_c: float,
    tokens_per_s: float,
    full_stop: int = FULL_STOP,
) -> float:
    """Expected joules of one action, plus heat and speed penalties."""
    depth, length, stop = action
    if length == 0:
        joules = shared_verify
    else:
        joules = cycle_joules_per_token(fits[depth], length, min(length, kept))
    if stop < full_stop:
        joules += 1e-4
    if temp_c >= THROTTLE_C:
        joules += (temp_c - THROTTLE_C) / 10.0 * length * 0.25
    # Penalise an action whose sweep speed is below the floor. The last round's
    # speed stays an input, but it must not tax a longer draft: that lock keeps
    # the next round on greedy after one slow step.
    if speeds.get((depth, length), floor) < floor:
        joules += 0.3 * max(length, 1)
    return float(joules)


def cheapest_action(costs: list[float], actions: tuple[tuple[int, int, int], ...]) -> tuple[int, int, int]:
    index = min(range(len(costs)), key=lambda item: (costs[item], item))
    return actions[index]


class RoundNet(nn.Module):
    """Five telemetry numbers in, one action logit per menu entry. No hidden state."""

    def __init__(self, actions: tuple[tuple[int, int, int], ...] = ()):
        super().__init__()
        self.actions = actions or menu()
        self.hidden = nn.Linear(FEATURE_COUNT, 64)
        self.out = nn.Linear(64, len(self.actions))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != FEATURE_COUNT:
            raise ValueError("features are power, temperature, clock, speed, and kept tokens")
        return self.out(F.gelu(self.hidden(features)))

    def action_at(self, features: torch.Tensor) -> tuple[int, int, int]:
        index = int(self.forward(features.view(1, -1)).argmax(dim=-1))
        return self.actions[index]


def pack_features(power_w: float, temp_c: float, clock_mhz: float, tokens_per_s: float, kept: float) -> torch.Tensor:
    return torch.tensor(
        [power_w / 100.0, temp_c / 100.0, clock_mhz / 3000.0, tokens_per_s / 50.0, kept / 4.0],
        dtype=torch.float32,
    )


def teach(profile: dict) -> dict:
    """Fit each exit and stop if one action wins every recorded step."""
    fits = fit_exit_profiles(profile["rows"])
    depth = 4 if 4 in fits else next(iter(fits))
    counts = acceptance_steps(profile["acceptance_histogram"], depth)
    varies = cheapest_changes(fits[depth], counts, depth)
    speeds = action_speeds(profile["rows"])
    floor = speeds.get((0, 0), min(speeds.values()))
    return {
        "fits": fits,
        "varies": varies,
        "teacher_depth": depth,
        "speeds": {f"{depth}/{length}": value for (depth, length), value in speeds.items()},
        "speed_floor": floor,
        "shared_verify_j": float(profile["verify_j"]),
        "repeat_noise_j": float(profile["repeat_noise_j"]),
        "locked": profile["startup_choice"],
        "reason": (
            "the cheapest action changes across steps"
            if varies
            else "one action wins every decode step"
        ),
    }


def _examples(teacher: dict, actions: tuple[tuple[int, int, int], ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speeds = {
        tuple(int(part) for part in key.split("/")): value
        for key, value in teacher["speeds"].items()
    }
    features, labels, costs = [], [], []
    # Power and clock do not change the joule label. They still vary here so a
    # live laptop reading is not out of the range the network saw.
    for power in (20.0, 50.0, 75.0):
        for clock in (600.0, 1800.0, 2600.0):
            for kept in range(5):
                for temp in (45.0, 55.0, 70.0, 86.0):
                    for speed in (30.0, 60.0):
                        row = [
                            action_cost(
                                teacher["fits"], teacher["shared_verify_j"], speeds, teacher["speed_floor"],
                                action, kept, temp, speed,
                            )
                            for action in actions
                        ]
                        label = cheapest_action(row, actions)
                        features.append(pack_features(power, temp, clock, speed, kept))
                        labels.append(actions.index(label))
                        costs.append(row)
    return torch.stack(features), torch.tensor(labels), torch.tensor(costs, dtype=torch.float32)


def train_network(
    teacher: dict,
    steps: int = 200,
    seed: int = 0,
    net: RoundNet | None = None,
) -> tuple[RoundNet, dict]:
    if not teacher["varies"]:
        raise ValueError("one action wins every step, so the locked draft stays")
    fresh = net is None
    actions = menu() if fresh else net.actions
    features, labels, costs = _examples(teacher, actions)
    generator = torch.Generator().manual_seed(seed)
    if fresh:
        net = RoundNet(actions)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2 if fresh else 1e-3)
    final = None
    for _ in range(steps):
        optimizer.zero_grad()
        order = torch.randperm(features.shape[0], generator=generator)
        logits = net(features[order])
        expected = (logits.softmax(dim=-1) * costs[order]).sum(dim=-1).mean()
        loss = F.cross_entropy(logits, labels[order]) + expected
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    with torch.no_grad():
        predicted = net(features).argmax(dim=-1)
        match = float((predicted == labels).float().mean())
    cool = net.action_at(pack_features(50.0, 45.0, 1800.0, 60.0, 4))
    hot = net.action_at(pack_features(50.0, 86.0, 1800.0, 60.0, 4))
    live = net.action_at(pack_features(30.0, 60.0, 2595.0, 50.0, 4))
    return net, {
        "final_loss": final,
        "label_match": match,
        "cool_action": {"depth": cool[0], "length": cool[1], "stop": cool[2]},
        "hot_action": {"depth": hot[0], "length": hot[1], "stop": hot[2]},
        "live_action": {"depth": live[0], "length": live[1], "stop": live[2]},
        "distinct_labels": sorted({int(item) for item in labels.tolist()}),
    }


def nudge(profile: dict, joules: float, accepted: int, rejected: int, rate: float = 0.02) -> None:
    """Move this exit's three costs toward the joules just measured for one round."""
    predicted = (
        float(profile["verify_j"])
        + float(profile["accepted_j"]) * accepted
        + float(profile["rejected_j"]) * rejected
    )
    error = joules - predicted
    profile["verify_j"] = max(0.0, float(profile["verify_j"]) + rate * error)
    if accepted:
        profile["accepted_j"] = max(0.0, float(profile["accepted_j"]) + rate * error / accepted)
    if rejected:
        profile["rejected_j"] = max(0.0, float(profile["rejected_j"]) + rate * error / rejected)


def read_signals(meter) -> tuple[float, float, float]:
    """Power in watts, temperature in C, and the graphics clock already running."""
    power = meter.nvml.nvmlDeviceGetPowerUsage(meter.handle) / 1000.0
    temp = meter.nvml.nvmlDeviceGetTemperature(meter.handle, meter.nvml.NVML_TEMPERATURE_GPU)
    clock = meter.nvml.nvmlDeviceGetClockInfo(meter.handle, meter.nvml.NVML_CLOCK_SM)
    return float(power), float(temp), float(clock)


class RoundRuntime:
    """One decision per speculative round. A positive hold_s keeps the choice for that long.

    The NVML energy counter is not read here. On this laptop, polling that
    counter during a request inflates it, so request joules stay a single
    before/after pair around the whole answer. A measured round can still
    update the fit through ``nudge_last``, using joules taken outside this loop.
    """

    def __init__(self, net: RoundNet, teacher: dict, meter, hold_s: float = 0.0, force_stop: int | None = None):
        self.net = net
        self.fits = deepcopy(teacher["fits"])
        self.shared_verify = float(teacher["shared_verify_j"])
        self.meter = meter
        self.hold_s = hold_s
        self.force_stop = force_stop
        locked = teacher.get("locked") or {}
        self.last_action = (
            int(locked.get("depth", 4)),
            int(locked.get("length", 4)),
            FULL_STOP,
        )
        self.last_time = 0.0
        self.last_kept = int(locked.get("length", 4))
        self.last_tps = float(teacher.get("speed_floor", 40.0))
        self._started = None
        self._open_round = None

    def choose(self, pending, cache) -> tuple[int, int, int]:
        now = time.perf_counter()
        if self.hold_s and now - self.last_time < self.hold_s:
            action = self.last_action
        else:
            power, temp, clock = read_signals(self.meter)
            features = pack_features(power, temp, clock, self.last_tps, self.last_kept)
            with torch.no_grad():
                action = self.net.action_at(features)
            self.last_action = action
            self.last_time = now
        self._started = time.perf_counter()
        depth, length, stop = action
        if self.force_stop is not None:
            stop = self.force_stop
        self._open_round = (depth, length, stop)
        return depth, length, stop

    def on_cycle(self, cycle: dict) -> None:
        elapsed = max(time.perf_counter() - (self._started or time.perf_counter()), 1e-6)
        committed = max(len(cycle["committed"]), 1)
        self.last_tps = committed / elapsed
        # A greedy step has no draft to score. Overwriting the last kept count
        # with 0 makes every later round look like a total miss, so the network
        # stays on ordinary writing. A total miss shortens the next feature to
        # one kept token, which is the cheapest speculative action at low acceptance.
        if int(cycle["chosen_drafts"]) > 0:
            accepted = int(cycle["accepted"])
            self.last_kept = accepted if accepted > 0 else 1
        depth = int(cycle["chosen_depth"])
        rejected = max(len(cycle["draft"]) - int(cycle["accepted"]), 0)
        self._open_round = (depth, int(cycle["accepted"]), rejected)

    def nudge_last(self, joules: float) -> None:
        """Move the open round's exit costs toward joules measured outside the request."""
        if self._open_round is None:
            return
        depth, accepted, rejected = self._open_round
        if depth not in self.fits:
            return
        nudge(self.fits[depth], joules, accepted, rejected)
