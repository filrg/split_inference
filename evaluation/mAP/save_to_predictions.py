from ultralytics.utils import ops, nms
import os
import torch


class Predictions:
    def __init__(
        self,
        conf=0.1,  # 0.25
        iou=0.7,
        agnostic_nms=False,
        max_det=300,
        classes=None,
        output_dir="evaluation/datasets/predictions",
        save=True,
    ):
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.classes = classes
        self.output_dir = output_dir
        self.save = save

        if self.save:
            os.makedirs(self.output_dir, exist_ok=True)

    def postprocess_v2(
        self,
        preds,
        img=(640, 640),
        frame_idx=0,
        orig_img_shape=None,
    ):
        assert orig_img_shape is not None, "orig_img_shape must be (h, w)"
        h, w = orig_img_shape

        preds = nms.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            classes=self.classes,
        )

        results = []

        for i, pred in enumerate(preds):
            frame_id = frame_idx + i

            if pred is None or len(pred) == 0:
                results.append([])
                continue

            # Clone to avoid modifying original tensor
            pred = pred.clone()

            square_size = max(h, w)
            pred[:, :4] = ops.scale_boxes(img, pred[:, :4], (square_size, square_size))

            pred[:, 0].clamp_(0, w)
            pred[:, 2].clamp_(0, w)
            pred[:, 1].clamp_(0, h)
            pred[:, 3].clamp_(0, h)

            frame_results = []

            for box in pred:
                x1, y1, x2, y2, conf, cls = box.tolist()

                # YOLO format
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                det = {
                    "class": int(cls),
                    "conf": float(conf),
                    "bbox": [cx, cy, bw, bh],
                }

                frame_results.append(det)

            results.append(frame_results)

            if self.save:
                output_file = os.path.join(
                    self.output_dir, f"frame_{frame_id:06d}.txt"
                )
                with open(output_file, "w") as f:
                    for det in frame_results:
                        cls = det["class"]
                        cx, cy, bw, bh = det["bbox"]
                        conf = det["conf"]
                        f.write(f"{cls} {cx} {cy} {bw} {bh} {conf}\n")

        return results