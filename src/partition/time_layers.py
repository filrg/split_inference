import torch
import time
from ultralytics import YOLO


class LayerProfiler:
    def __init__(self, config, mode="time", unit="MB"):
        """
        mode : "time" | "shape"
        unit : "KB" | "MB"   (for shape)
        """
        assert mode in ["time", "shape"]
        assert unit in ["KB", "MB"]

        self.mode = mode
        self.unit = unit
        self.num_runs = config["time_layer"]["num_round"]
        self.input_shape = config["time_layer"]["input_shape"]

        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Profiling on device: {self.device}")

        # -------------------------
        # Load model
        # -------------------------
        # We load the base model from the YOLO object
        self.model = YOLO(config["server"]["model"]).model
        self.model.eval()

        # Handle Precision based on Device
        if self.device == "cuda":
            # GPUs handle FP16 (Half) very well for speed
            self.model.half().to(self.device)
            self.x = torch.randn(*self.input_shape, device=self.device).half()
        else:
            # CPU usually requires FP32 (Float) for many ops like Upsample
            self.model.float().to(self.device)
            self.x = torch.randn(*self.input_shape, device=self.device).float()

        # -------------------------
        # Storage
        # -------------------------
        self.num_layers = len(self.model.model)

        # time_per_layer[layer_idx] = [t1, t2, ...] (in microseconds)
        self.time_per_layer = [[] for _ in range(self.num_layers)]

        # shape_list[layer_idx] = MB / KB
        self.shape_list = [None for _ in range(self.num_layers)]

        # Events storage (Handles both CUDA Events and CPU timestamps)
        self._start_events = {}
        self._end_events = {}

        # Register hooks
        for idx, m in enumerate(self.model.model):
            m._layer_idx = idx
            m.register_forward_pre_hook(self._pre_hook)
            m.register_forward_hook(self._post_hook)

    # --------------------------------------------------
    # Hooks
    # --------------------------------------------------
    def _pre_hook(self, m, inp):
        if self.mode == "time":
            if self.device == "cuda":
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._start_events[m._layer_idx] = ev
            else:
                # Use high-resolution CPU timer
                self._start_events[m._layer_idx] = time.perf_counter()

    def _post_hook(self, m, inp, out):
        idx = m._layer_idx

        # -------- TIME MODE --------
        if self.mode == "time":
            if self.device == "cuda":
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._end_events[idx] = ev
            else:
                # CPU timing is synchronous, calculate immediately
                start_time = self._start_events[idx]
                # Result in microseconds (us)
                duration_us = (time.perf_counter() - start_time) * 1_000_000
                self.time_per_layer[idx].append(duration_us)

        # -------- SHAPE MODE --------
        if self.mode == "shape" and self.shape_list[idx] is None:
            total_bytes = 0
            if isinstance(out, (list, tuple)):
                for o in out:
                    if torch.is_tensor(o):
                        total_bytes += o.numel() * o.element_size()
            elif torch.is_tensor(out):
                total_bytes = out.numel() * out.element_size()

            denom = 1024 if self.unit == "KB" else (1024 ** 2)
            self.shape_list[idx] = round(total_bytes / denom, 3)

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(self):
        # -------------------------
        # Warm-up (Important for JIT/CUDNN)
        # -------------------------
        with torch.no_grad():
            self.model(self.x)

        # -------------------------
        # Benchmark
        # -------------------------
        for _ in range(self.num_runs):
            with torch.no_grad():
                self.model(self.x)

            if self.mode == "time" and self.device == "cuda":
                # Wait for GPU to finish all scheduled tasks
                torch.cuda.synchronize()

                for i in range(self.num_layers):
                    # elapsed_time returns milliseconds (ms), convert to us
                    t_us = self._start_events[i].elapsed_time(self._end_events[i]) * 1000
                    self.time_per_layer[i].append(t_us)

            # If CPU, times are already appended in _post_hook

        # -------------------------
        # Return Results
        # -------------------------
        if self.mode == "time":
            # Return mean time per layer in microseconds
            return [
                round(sum(t) / len(t), 2) if len(t) > 0 else 0.0
                for t in self.time_per_layer
            ]
        else:
            return self.shape_list