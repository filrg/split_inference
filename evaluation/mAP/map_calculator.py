import os
import numpy as np


class DirectoryMAPCalculator:
    """
    input : path of ground truth and predictions\
        GT:         class center_x center_y width height
        Prediction: class center_x center_y width height conf
        Coordinates are normalized (0–1)
    output :
        self.compute_map(threshold)
        self.compute_map_coco : return threshold from 0.5 to 0.95
    """

    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def reset(self):
        self.predictions = []
        self.ground_truths = []

    @staticmethod
    def calculate_iou(box1, box2):

        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter = max(0, xB - xA) * max(0, yB - yA)

        if inter <= 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def yolo_to_xyxy(cx, cy, w, h):

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return [x1, y1, x2, y2]

    def load_ground_truth_folder(self, folder):

        frame_id = 1

        for file in sorted(os.listdir(folder)):

            path = os.path.join(folder, file)

            if not os.path.isfile(path):
                continue

            with open(path) as f:
                for line in f:

                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    cls, cx, cy, w, h = map(float, parts)

                    box = self.yolo_to_xyxy(cx, cy, w, h)

                    self.ground_truths.append({
                        "frame_id": frame_id,
                        "class_id": int(cls),
                        "box": box
                    })

            frame_id += 1

    def load_prediction_folder(self, folder):

        frame_id = 1

        for file in sorted(os.listdir(folder)):

            path = os.path.join(folder, file)

            if not os.path.isfile(path):
                continue

            with open(path) as f:
                for line in f:

                    parts = line.strip().split()

                    if len(parts) < 6:
                        continue

                    cls, cx, cy, w, h, conf = map(float, parts)

                    box = self.yolo_to_xyxy(cx, cy, w, h)

                    self.predictions.append({
                        "frame_id": frame_id,
                        "class_id": int(cls),
                        "box": box,
                        "conf": conf
                    })

            frame_id += 1

    def _calculate_ap_per_class(self, class_id, iou_threshold):

        preds = [p for p in self.predictions if p["class_id"] == class_id]
        gts = [g for g in self.ground_truths if g["class_id"] == class_id]

        if len(gts) == 0:
            return 0

        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        gt_pool = {}

        for g in gts:
            fid = g["frame_id"]
            gt_pool.setdefault(fid, []).append({
                "box": g["box"],
                "matched": False
            })

        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))

        for i, pred in enumerate(preds):

            fid = pred["frame_id"]

            best_iou = 0
            best_idx = -1

            for j, gt in enumerate(gt_pool.get(fid, [])):

                iou = self.calculate_iou(pred["box"], gt["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0:

                if not gt_pool[fid][best_idx]["matched"]:
                    TP[i] = 1
                    gt_pool[fid][best_idx]["matched"] = True
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        acc_TP = np.cumsum(TP)
        acc_FP = np.cumsum(FP)

        recalls = acc_TP / len(gts)
        precisions = acc_TP / (acc_TP + acc_FP + 1e-9)

        mrec = np.concatenate(([0], recalls, [1]))
        mpre = np.concatenate(([1], precisions, [0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        return ap

    def compute_map(self, iou_threshold=0.5):

        classes = set([g["class_id"] for g in self.ground_truths] +
                      [p["class_id"] for p in self.predictions])

        aps = []

        for c in classes:
            aps.append(self._calculate_ap_per_class(c, iou_threshold))

        return np.mean(aps) if aps else 0

    def compute_coco_map(self):

        thresholds = np.arange(0.5, 1.0, 0.05)

        scores = [self.compute_map(t) for t in thresholds]

        return np.mean(scores)

pred_dir = "../dataset/predictions"
gt_dir = "../datasets/groundtruth"

calc = DirectoryMAPCalculator()

calc.load_prediction_folder(pred_dir)
calc.load_ground_truth_folder(gt_dir)

map50 = calc.compute_map(0.5)
map5095 = calc.compute_coco_map()

print(f"mAP@0.5:{ map50:.6f}")
print(f"mAP@0.5:0.95:{map5095:.6f}")