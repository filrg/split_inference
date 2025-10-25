import torch
import time
import pandas as pd
from ultralytics import YOLO
from src.Utils import write_partial


class LayerProfiler:
    def __init__(self, config):
        self.model_path = config["model"]
        self.num_runs = config["time_layer"]["num_round"]
        self.input_shape = config["time_layer"]["input_shape"]

        # Load YOLO model
        self.model = YOLO(self.model_path).model
        self.model.eval()

        # Random input tensor
        self.x = torch.randn(*self.input_shape)

        # Storage
        self.layer_start = {}
        self.layer_times = {}

        # Register hooks
        for m in self.model.model:
            m.register_forward_pre_hook(self._pre_hook)
            m.register_forward_hook(self._hook_fn)

    def _pre_hook(self, m, inp):
        """Store start time for each layer."""
        self.layer_start[m.i] = time.perf_counter()

    def _hook_fn(self, m, inp, out):
        """Measure elapsed time for each layer."""
        elapsed = (time.perf_counter() - self.layer_start[m.i]) * 1e6  # μs
        name = f"{m.__class__.__name__}_{m.i}"
        if name not in self.layer_times:
            self.layer_times[name] = []
        self.layer_times[name].append(elapsed)

    def run(self):
        # Warm-up run
        with torch.no_grad():
            self.model(self.x)

        # Benchmark runs
        for _ in range(self.num_runs):
            with torch.no_grad():
                self.model(self.x)

        # Compute average per-layer times
        avg_times = {
            name: round(sum(times) / len(times), 2)
            for name, times in self.layer_times.items()
        }

        times_store = list(avg_times.values())

        # Total time
        total_time = round(sum(avg_times.values()), 2)
        avg_times["total_time"] = total_time

        return times_store