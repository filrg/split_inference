import torch
import time
from ultralytics import YOLO

# -------------------------
# Config
# -------------------------
MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 640
WARMUP = 20
RUNS = 100
DEVICE = "cuda"

# -------------------------
# Load model
# -------------------------
model = YOLO(MODEL_PATH)
model.to(DEVICE)
model.model.half()          # FP16
model.model.eval()

# Dummy input
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE).half()

# -------------------------
# Warm-up (important!)
# -------------------------
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model.model(dummy)

torch.cuda.synchronize()

# -------------------------
# Benchmark
# -------------------------
times = []

with torch.no_grad():
    for _ in range(RUNS):
        start = time.time()
        _ = model.model(dummy)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time

print("=================================")
print(f"Device        : {torch.cuda.get_device_name(0)}")
print(f"Precision     : FP16")
print(f"Input size    : {IMG_SIZE}x{IMG_SIZE}")
print(f"Avg time/frame: {avg_time*1000:.2f} ms")
print(f"FPS           : {fps:.2f}")
print("=================================")
