import numpy as np
import torch , os , yaml, gc , time
import torch.nn as nn
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops , nms
from ultralytics.nn.tasks import DetectionModel

from src.partition.tools import extract_input_layer , load_weights_optimized
class YOLOHeadInference:
    def __init__(
        self,
        cfg_yaml: str,
        sys_cfg: str,
        weight_path: str,
        device: str,
    ):
        # Device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # print(f"[DEVICE] {self.device}")

        # System config
        with open(sys_cfg) as f:
            self.config = yaml.safe_load(f)

        # Output layers
        self.output_layers = extract_input_layer("yolo11n.yaml")["output"]
        # print(f"[Output layers] {self.output_layers}")

        # Load model
        with open(cfg_yaml, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)

        self.model = DetectionModel(model_cfg, verbose=False)

        # Load weights
        load_weights_optimized(self.model, weight_path)

        # Finalize model
        self.model.to(self.device)
        self.model.eval()
        self.model.half()   # model FP16

        self.time_layers = []
        self.time_start = time.perf_counter_ns()
        # print(self.time_start)

    # Input preparation (FP16 + device sync)
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure input tensor is on correct device and FP16
        """
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        if x.dtype != torch.float16:
            x = x.half()

        return x

    # Optimized forward using hooks
    def forward(self, x: torch.Tensor):
        """
        x: input tensor (any dtype/device)
        """
        x = self._prepare_input(x)

        feature_maps = {}

        def hook_fn(layer_id):
            def fn(_, __, out):
                feature_maps[layer_id] = out
            return fn

        handles = [
            self.model.model[i].register_forward_hook(hook_fn(i))
            for i in self.output_layers
        ]

        with torch.inference_mode():
            _ = self.model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        for h in handles:
            h.remove()

        print((time.perf_counter_ns() - self.time_start) / 1000)
        return feature_maps

if __name__ == "__main__":
    dummy_image = torch.randn(1, 3, 640, 640)
    run = YOLOHeadInference(
        cfg_yaml="cfg/yolo11n.yaml",
        sys_cfg="cfg/config.yaml",
        weight_path="part.pt",
        device="cuda"
    )
    run.forward(dummy_image)