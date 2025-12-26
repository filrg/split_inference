# import cv2
# import torch
# import time
# from ultralytics import YOLO
# import torchvision.transforms as T
#
# VIDEO_PATH = "video.mp4"
# IMG_SIZE = 640
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# yolo = YOLO("yolo11n.pt")
# model = yolo.model.to(DEVICE)
# model.eval()
#
# transform = T.ToTensor()
# cap = cv2.VideoCapture(VIDEO_PATH)
#
# times = []
#
# with torch.no_grad():
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         x = transform(img).unsqueeze(0).to(DEVICE)
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#
#         t0 = time.perf_coun ter()
#         _ = model(x)
#
#         if DEVICE == "cuda":
#             torch.cuda.synchronize()
#
#         t1 = time.perf_counter()
#         times.append((t1 - t0) * 1000)
#
# cap.release()
#
# avg_time = sum(times) / len(times)
# print(f"Average inference time: {avg_time:.2f} ms")
# print(f"FPS: {1000 / avg_time:.2f}")
import torch
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Lấy state_dict (chỉ weights)
state_dict = model.model.state_dict()

# Lưu ra file chỉ chứa weights
torch.save(state_dict, "part.pt")

print("✅ Saved weights-only file: part.pt")
