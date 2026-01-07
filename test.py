# import time
#
# import torch
# from ultralytics import YOLO
#
# # -------------------------
# # Config
# # -------------------------
# MODEL_PATH = "yolo11n.pt"
# IMG_SIZE = 640
# WARMUP = 30
# RUNS = 200
# DEVICE = "cuda"
#
# # -------------------------
# # Load model
# # -------------------------
# model = YOLO(MODEL_PATH)
# model.to(DEVICE)
# model.model.half()      # FP16
# model.model.eval()
#
# # Dummy input
# dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE).half()
#
# # -------------------------
# # Warm-up
# # -------------------------
# print('start warm-up !!!')
# with torch.no_grad():
#     for _ in range(WARMUP):
#         _ = model.model(dummy)
#
# torch.cuda.synchronize()
#
# # -------------------------
# # CUDA Event timing
# # -------------------------
# starter = torch.cuda.Event(enable_timing=True)
# ender = torch.cuda.Event(enable_timing=True)
#
# times_ms = []
#
# print('start inference !')
# start = time.time()
# with torch.no_grad():
#     for _ in range(RUNS):
#         starter.record()
#         _ = model.model(dummy)
#         ender.record()
#         torch.cuda.synchronize()
#         times_ms.append(starter.elapsed_time(ender))  # milliseconds
#
# print(time.time() - start)
# print('inference done !')
# avg_ms = sum(times_ms) / len(times_ms)
# fps = 1000.0 / avg_ms
#
# # -------------------------
# # Report
# # -------------------------
# print("=================================")
# print(f"Device         : {torch.cuda.get_device_name(0)}")
# print(f"Precision      : FP16")
# print(f"Input size     : {IMG_SIZE}x{IMG_SIZE}")
# print(f"Avg GPU time   : {avg_ms:.2f} ms")
# print(f"FPS            : {fps:.2f}")
# print("=================================")

from src.clustering.clustering import Clustering
import numpy as np

device_names = [
    "Jetson Nano",
    "Jetson TX2",
    # "Jetson Xavier",
    # "RTX 3060"
]

features = np.array([
    [0.5, 0.25, 4 , 5],
    [1.3, 1.0, 8 , 6],
    # [30, 10, 16 , 8],
    # [170, 13, 12 , 9]
])

cluster = Clustering(features, device_names, n_clusters=2)

print(cluster.run())

