"""Total request GPU-counter energy, or explicit unavailable value."""
import time
import torch


class Meter:
    def __init__(self, device):
        self.device, self.handle, self.nvml = device, None, None
        self.reason = 'No CPU/whole-device energy sensor configured'
        if device.type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                uuid = str(torch.cuda.get_device_properties(device).uuid)
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                self.counter()
                self.reason = None
            except Exception as error:
                self.handle = None
                self.reason = str(error)

    def counter(self):
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) if self.handle else None

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def run(self, function):
        self.sync()
        before = self.counter()
        start = time.perf_counter()
        result = function()
        self.sync()
        after = self.counter()
        elapsed = time.perf_counter()-start
        if before is not None and after < before:
            raise RuntimeError('Energy counter decreased')
        return result, dict(latency_s=elapsed, energy_j=(after-before)/1000 if before is not None else None,
                            boundary='gpu' if self.handle else 'unavailable',
                            energy_reason=self.reason, source='NVML cumulative counter' if self.handle else None)

    def close(self):
        if self.nvml:
            self.nvml.nvmlShutdown()
