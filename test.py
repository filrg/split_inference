import cv2
import torch
from src.Model import YOLOHeadInference , YOLOTailInference
from src.Compress import Encoder , Decoder


# =====================================================
# 1. DEVICE CHECK (ENTRY POINT)
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using device: {DEVICE}")


# =====================================================
# 2. CONFIG
# =====================================================
VIDEO_PATH = "video.mp4"
BATCH_SIZE = 30


# =====================================================
# 3. INIT MODEL
# =====================================================
runner = YOLOHeadInference(
    cfg_yaml="cfg/yolo11n.yaml",
    sys_cfg="cfg/config.yaml",
    weight_path="part1.pt",
    device=DEVICE
)


# =====================================================
# 4. VIDEO LOOP
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Cannot open video"

input_batch = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # CPU preprocess (per frame)
    # -----------------------------
    frame = cv2.resize(frame, (640, 640))
    frame = frame.astype("float32") / 255.0
    tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (3,640,640)

    input_batch.append(tensor)

    # -----------------------------
    # INFER WHEN BATCH FULL
    # -----------------------------
    if len(input_batch) == BATCH_SIZE:
        # stack -> (B,3,640,640)
        x = torch.stack(input_batch, dim=0)

        # CHECK INPUT (before model)
        print("[CHECK INPUT]", x.device, x.dtype, x.shape)

        # forward (model will move + FP16 internally)
        y = runner.forward_head(x)
        print(f"[DEBUG] type of y {type(y)}")
        print(f"[KEYS of Y ] {y.keys()}")
        print(f"[TYPE of 4 ] {type(y[4])}")
        for key , val in y.items():
            if isinstance(val , torch.Tensor):
                y[key] = val.cpu().numpy()


        for key , val in y.items():
            print(f"[KEY] {key} [TYPE VAL] {type(val)} [DEVICE VAL] {val.device}")

        out , shape = Encoder(data_output= [y[4]] , num_bits=8)
        print(f"[SHAPE] {shape}")

        # CHECK OUTPUT (confirm GPU + FP16)
        feat = next(iter(y.values()))
        print("[CHECK OUTPUT]", feat.device, feat.dtype)

        input_batch.clear()


# =====================================================
# 5. HANDLE LAST INCOMPLETE BATCH
# =====================================================
if input_batch:
    x = torch.stack(input_batch, dim=0)
    print("[CHECK INPUT LAST]", x.device, x.dtype, x.shape)

    y = runner.forward_head(x)
    feat = next(iter(y.values()))
    print("[CHECK OUTPUT LAST]", feat.device, feat.dtype)


cap.release()
print("[DONE] Video inference finished")

