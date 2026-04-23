import torch
import time
from ultralytics import YOLO

class Edge:
    def __init__(self , n_flops = 100):
        self.n_flops = n_flops

    def run(self):



    def measuringFLOPS(self):
        model = YOLO("yolo11n.pt").model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        model.to(device)

        model.eval()
        model.fuse()  # Optional but recommended: fuses Conv2d + BatchNorm2d for speed

        dummy = torch.randn(1, 3, 640, 640).to(device)

        print(f"Bắt đầu đo trên thiết bị: {device}")

        with torch.no_grad():
            # Warm-up (Important to wake up GPU clocks)
            for _ in range(50):
                _ = model(dummy)

            # Measure 100 iterations
            times = []
            for _ in range(self.n_flops):
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()  # perf_counter is more precise than time.time()
                _ = model(dummy)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        # 5. Results
        avg_time = sum(times) / len(times)
        # YOLO11n @ 640x640 has approx 6.5 - 6.6 GFLOPs
        achieved_gflops = 6.6 / avg_time

        return achieved_gflops