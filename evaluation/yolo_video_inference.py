import os
import cv2
import torch
from ultralytics import YOLO
import time , yaml

with open('../cfg/config.yaml') as file:
    config = yaml.safe_load(file)

start = time.time()

MODEL_PATH = "../" + config["server"]["model"]
VIDEO_PATH = "../" + config["data"]
OUTPUT_DIR = "dataset/predictions"
BATCH_SIZE = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)


def post_process(results, frame_ids, output_dir):
    for result, frame_idx in zip(results, frame_ids):

        h, w = result.orig_shape

        output_file = os.path.join(
            output_dir, f"frame_{frame_idx:06d}.txt"
        )

        with open(output_file, "w") as f:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cls in zip(boxes, classes):

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


frame_id = 1
batch_frames = []
batch_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    batch_frames.append(frame)
    batch_ids.append(frame_id)

    if len(batch_frames) == BATCH_SIZE:
        results = model(batch_frames, device=device, verbose=False)

        post_process(results, batch_ids, OUTPUT_DIR)

        batch_frames.clear()
        batch_ids.clear()

    frame_id += 1


# handle last batch
if batch_frames:
    results = model(batch_frames, device=device, verbose=False)
    post_process(results, batch_ids, OUTPUT_DIR)

cap.release()

print("Done.")
print(f"time: {time.time() - start:.2f}s")