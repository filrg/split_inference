import torch
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -------------------------
        # Load model (FP16)
        # -------------------------
        self.model = YOLO(config["model"]).model
        self.model.eval().half().to(self.device)

        self.x = torch.randn(*self.input_shape, device=self.device).half()

        # -------------------------
        # Storage
        # -------------------------
        self.num_layers = len(self.model.model)

        # time_per_layer[layer_idx] = [t1, t2, ...]
        self.time_per_layer = [[] for _ in range(self.num_layers)]

        # shape_list[layer_idx] = MB / KB
        self.shape_list = [None for _ in range(self.num_layers)]

        # CUDA events
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
        if self.mode == "time" and self.device == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._start_events[m._layer_idx] = ev

    def _post_hook(self, m, inp, out):
        idx = m._layer_idx

        # -------- TIME MODE --------
        if self.mode == "time" and self.device == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._end_events[idx] = ev

        # -------- SHAPE MODE (chỉ lấy 1 lần) --------
        if self.mode == "shape" and self.shape_list[idx] is None:
            total_bytes = 0

            if isinstance(out, (list, tuple)):
                for o in out:
                    if torch.is_tensor(o):
                        total_bytes += o.numel() * o.element_size()
            elif torch.is_tensor(out):
                total_bytes = out.numel() * out.element_size()

            if self.unit == "KB":
                self.shape_list[idx] = round(total_bytes / 1024, 3)
            else:
                self.shape_list[idx] = round(total_bytes / (1024 ** 2), 3)

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(self):
        # -------------------------
        # Warm-up
        # -------------------------
        with torch.no_grad():
            self.model(self.x)

        # -------------------------
        # Benchmark
        # -------------------------
        for _ in range(self.num_runs):
            with torch.no_grad():
                self.model(self.x)

            # 🔥 synchronize 1 LẦN / forward
            if self.mode == "time" and self.device == "cuda":
                torch.cuda.synchronize()

                # collect times for this round
                for i in range(self.num_layers):
                    t_us = (
                        self._start_events[i]
                        .elapsed_time(self._end_events[i])
                        * 1000
                    )
                    self.time_per_layer[i].append(t_us)

        # -------------------------
        # Return
        # -------------------------
        if self.mode == "time":
            # mean per layer
            return [
                round(sum(t) / len(t), 2) if len(t) > 0 else 0.0
                for t in self.time_per_layer
            ]
        else:
            return self.shape_list
