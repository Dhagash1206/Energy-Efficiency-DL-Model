"""Actual elapsed time and optional NVML cumulative GPU energy; never TDP estimates."""

import time


class EnergyMeter:
    def __init__(self, backend):
        self.nvml = None
        self.handle = None
        self.reason = "Energy unavailable for this backend; no CPU/Intel/Apple energy estimate is substituted."
        if getattr(backend, "device", None) != "cuda":
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            # Match CUDA's visible device by UUID; do not assume NVML index 0 is CUDA index 0.
            props = backend.torch.cuda.get_device_properties(0)
            uuid = getattr(props, "uuid", None)
            if not uuid:
                raise RuntimeError("CUDA device UUID unavailable; cannot safely select the energy meter")
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            self.read()
            self.reason = "NVML cumulative GPU energy; excludes CPU and the rest of the laptop."
        except Exception as exc:
            self.handle = None
            self.reason = f"Energy unavailable: {exc}"

    def read(self):
        if self.handle is None:
            return None
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def close(self):
        if self.nvml:
            self.nvml.nvmlShutdown()


def start_measurement(backend, meter):
    backend.synchronize()
    energy = meter.read()
    return energy, time.perf_counter()


def stop_measurement(backend, meter, start):
    backend.synchronize()
    elapsed = time.perf_counter() - start[1]
    end = meter.read()
    delta = None if end is None or start[0] is None else end - start[0]
    if delta is not None and delta <= 0:
        delta = None
    return delta, elapsed
