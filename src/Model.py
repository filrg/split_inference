import numpy as np
import torch , os , yaml, gc
import torch.nn as nn
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops , nms
from ultralytics.nn.tasks import DetectionModel

from src.partition.tools import extract_input_layer , load_weights_optimized

class SplitDetectionModel(nn.Module):
    def __init__(self, cfg=YOLO('yolo11n.pt').model, split_layer=-1):
        super().__init__()
        self.model = cfg.model
        self.save = cfg.save
        self.names = cfg.names
        self.stride = cfg.stride
        self.inplace = cfg.inplace
        self.yaml = cfg.yaml
        self.nc = len(self.names)  # cfg.nc
        self.task = cfg.task
        self.pt = True

        if split_layer > 0:
            self.head = self.model[:split_layer]
            self.tail = self.model[split_layer:]

        self.output = extract_input_layer("yolo11n.yaml")



    def forward_head(self, x, output_from=()):
        # print(self.output)
        print(f"[DEBUG] [check output] {output_from} [check save] {self.save}")
        y, dt = [], []  # outputs
        for i, m in enumerate(self.head):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            if (m.i in self.save) or (i in output_from):
                y.append(x)
            else:
                y.append(None)

        for mi in range(len(y)):
            if mi not in output_from:
                y[mi] = None

        if y[-1] is None:
            y[-1] = x
        return {"layers_output": y, "last_layer_idx": len(y) - 1}

    def forward_tail(self, x):
        y = x["layers_output"]
        x = y[x["last_layer_idx"]]
        for m in self.tail:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x if m.i in self.save else None)

        y = x
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0] if len(y) == 1 else [self.from_numpy(x) for x in y])
        else:
            return self.from_numpy(y)

    def _predict_once(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def forward(self, x):
        return self._predict_once(x)

    def from_numpy(self, x):
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x


class SplitDetectionPredictor(DetectionPredictor):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        model.fp16 = self.args.half
        self.model = model

    def postprocess(self, preds, img_shape=None, orig_shape=None, orig_imgs=None):
        """Post-processes predictions and returns a list of Results objects."""
        preds = nms.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if orig_imgs is not None and not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            if orig_imgs is None:
                orig_img = np.empty([0, 0, 0, 0])
                img_path = ""
            else:
                orig_img = orig_imgs[i]
                img_path = ""

            pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], orig_shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

class BoundingBox(DetectionPredictor):
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        coco_yaml_path = os.path.join(current_dir, "coco.yaml")
        with open(coco_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.names = data["names"]

    def postprocess(self, preds, img_shape=None, orig_shape=None, orig_imgs=None):
        """Post-processes predictions and returns a list of Results objects."""
        preds = nms.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if orig_imgs is not None and not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            if orig_imgs is None:
                orig_img = np.empty([0, 0, 0, 0])
                img_path = ""
            else:
                orig_img = orig_imgs[i]
                img_path = ""

            pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], orig_shape)
            results.append(Results(orig_img, path=img_path, names=self.names, boxes=pred))
        return results



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

        return feature_maps


class YOLOTailInference:
    def __init__(
        self,
        cfg_yaml: str,
        sys_cfg: str,
        weight_path: str,
        device: str ,
    ):
        # =============================
        # Device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[DEVICE] {self.device}")

        # System config
        with open(sys_cfg) as f:
            self.config = yaml.safe_load(f)

        # Load architecture (tail / full)
        yaml_file = cfg_yaml

        print(f"[YAML] {yaml_file}")

        with open(yaml_file, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)

        self.model = DetectionModel(model_cfg, verbose=False)

        # =============================
        # Tail meta info
        # =============================
        info = extract_input_layer("yolo11n.yaml")
        self.res_tail = info["res_tail"]
        self.output_layers = info["output"]
        self.cut_layer = self.output_layers[-1] + 1

        # =============================
        # Load tail weights
        # =============================
        load_weights_optimized(self.model, weight_path)

        # =============================
        # Finalize model
        # =============================
        self.model.to(self.device)
        self.model.eval()
        self.model.half()

    # ==================================================
    # Prepare input feature map
    # ==================================================
    def _prepare_state(self, state_dict: dict) -> dict:
        """
        Ensure all tensors are FP16 and on correct device
        """
        out = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                if v.device != self.device:
                    v = v.to(self.device, non_blocking=True)
                if v.dtype != torch.float16:
                    v = v.half()
            out[k] = v
        return out

    # ==================================================
    # Forward tail
    # ==================================================
    def forward(self, state_dict: dict):
        """
        state_dict: feature maps from head
        """
        y = self._prepare_state(state_dict)

        # last output from head
        y[-1] = y[self.cut_layer -  1]

        with torch.inference_mode():
            for layer in self.model.model[self.cut_layer:]:
                if isinstance(layer.f, int):
                    x_in = y[layer.f]
                else:
                    x_in = [y[j] for j in layer.f]

                x_out = layer(x_in)

                if layer.i in self.res_tail:
                    y[layer.i] = x_out

                y[-1] = x_out

        return y[-1]