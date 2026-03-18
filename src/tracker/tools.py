import os, yaml, csv
import numpy as np
import pandas as pd

from datetime import datetime
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops, nms
from ultralytics.engine.results import Results


# ---- Bounding Box Post-processing ----

class BoundingBox(DetectionPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load class names from coco.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        coco_yaml_path = os.path.join(current_dir, "coco.yaml")

        with open(coco_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.names = data["names"]

    def postprocess(self, preds, img_shape=None, orig_shape=None, orig_imgs=None):
        """
        Post-process model predictions and return a list of Results objects.
        """

        # Apply Non-Max Suppression (NMS)
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes
        )

        # Convert tensor batch → numpy if needed
        if orig_imgs is not None and not isinstance(orig_imgs, list):
            # Input images are torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []

        for i, pred in enumerate(preds):

            # Handle missing original images
            if orig_imgs is None:
                orig_img = np.empty([0, 0, 0, 0])
                img_path = ""
            else:
                orig_img = orig_imgs[i]
                img_path = ""

            # Rescale bounding boxes to original image size
            pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], orig_shape)

            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.names,
                    boxes=pred
                )
            )

        return results


# ---- Utility Functions ----

class Tools:
    def __init__(self):
        pass

    def visual_time(self, data, title):
        # Plot time-series data for visualization
        plt.plot(data, marker='o', linestyle='-', label="My Data")
        plt.title(title)
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_datatime(self, millisecond=False):
        # Get current time as a formatted string
        now = datetime.now()

        hour = now.hour
        minute = now.minute
        second = now.second
        millisecond = now.microsecond // 1000  # Convert microseconds → milliseconds

        if millisecond:
            return f"{hour}:{minute}:{second}.{millisecond}"
        else:
            return f"{hour}:{minute}:{second}"


# ---- CSV Logging ----

# Convert a list of column names into a dict initialized with -1
def list_to_dict_with_minus_one(lst):
    return {key: -1 for key in lst}


# Default CSV columns
cols = [
    "Time", "PointCut",
    "[T]totalTM", "[T]FPSR",
    "[1]totalFr", "[2]totalTm", "[1]totalTm",
    "[1]outSze[T]", "[1]outSze[2]", "[2]outSize",
    "[1]GPU_time", "[2]GPU_time",
    "[1]peak_RAM", "[2]peak_RAM",
    "[1]peak_VRAM", "[2]peak_VRAM",
]

dict_data = list_to_dict_with_minus_one(cols)
file_path = "res/output.csv"

row_buffer = {}


def write_partial(partial_data, flush=False):
    global row_buffer, cols

    # Ensure CSV header exists / is updated
    update_csv_header(file_path, cols)

    # Update buffer with new partial data
    row_buffer.update(partial_data)

    # Detect new columns dynamically
    new_cols = [c for c in partial_data.keys() if c not in cols]

    if new_cols:
        cols.extend(new_cols)  # Add new columns

        # Reload CSV and append new empty columns for old rows
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for c in new_cols:
                df[c] = ""
            df.to_csv(file_path, index=False)

    # If all columns are filled → flush row
    if all(col in row_buffer for col in cols):
        flush = True

    if flush:
        row_df = pd.DataFrame([row_buffer], columns=cols)

        if not os.path.exists(file_path):
            row_df.to_csv(file_path, index=False)
        else:
            row_df.to_csv(file_path, mode='a', index=False, header=False)

        row_buffer = {}
        print("[CSV] Write successful.")


def update_csv_header(filename, new_headers):
    # Update only the header row of an existing CSV file

    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError("CSV file is empty!")

    # Replace the first row (header)
    reader[0] = new_headers

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(reader)