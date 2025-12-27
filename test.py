import time
import torch
from ultralytics import YOLO


def profile_yolo_layers(
    model_path="yolo11n.pt",
    input_shape=(1, 3, 640, 640),
    num_runs=30,
    warmup=5,
    device=None,
    fp16=True,
    move_output_to_cpu=True,
):
    """
    Returns:
        layer_times  : list[float]  # ms, prefix-delta
        layer_outputs: list         # output từng layer
        layer_shapes : list         # shape từng layer
    """

    # -------------------------
    # Device
    # -------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Load YOLO model
    # -------------------------
    model = YOLO(model_path).model
    model.eval().to(device)
    if fp16:
        model = model.half()

    # -------------------------
    # Dummy input (fake image)
    # -------------------------
    x0 = torch.randn(*input_shape, device=device)
    if fp16:
        x0 = x0.half()

    # ============================================================
    # 1) FORWARD 1 LẦN → LẤY OUTPUT + SHAPE TỪNG LAYER (YOLO GRAPH)
    # ============================================================
    layer_outputs = []
    layer_shapes = []

    outputs_cache = []

    with torch.no_grad():
        x = x0
        for i, m in enumerate(model.model):
            # xử lý graph (from)
            if m.f != -1:
                if isinstance(m.f, int):
                    x = outputs_cache[m.f]
                else:
                    x = [outputs_cache[j] for j in m.f]

            x = m(x)
            outputs_cache.append(x)

            # ----- lưu output -----
            if isinstance(x, (list, tuple)):
                out = tuple(
                    o.detach().cpu() if move_output_to_cpu else o
                    for o in x
                    if torch.is_tensor(o)
                )
                layer_outputs.append(out)
                layer_shapes.append(
                    [tuple(o.shape) for o in x if torch.is_tensor(o)]
                )
            else:
                out = x.detach().cpu() if move_output_to_cpu else x
                layer_outputs.append(out)
                layer_shapes.append(tuple(x.shape))

    # ============================================================
    # 2) PREFIX–DELTA LATENCY (ĐO THỜI GIAN TỪNG LAYER – CHUẨN NHẤT)
    # ============================================================
    def run_prefix(end_idx):
        outputs = []
        x = x0
        for i, m in enumerate(model.model[: end_idx + 1]):
            if m.f != -1:
                if isinstance(m.f, int):
                    x = outputs[m.f]
                else:
                    x = [outputs[j] for j in m.f]
            x = m(x)
            outputs.append(x)
        return x

    def measure_prefix_latency(end_idx):
        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                run_prefix(end_idx)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_runs):
                run_prefix(end_idx)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        return (t1 - t0) * 1000 / num_runs  # ms

    layer_times = []
    prev_prefix_time = 0.0

    for i in range(len(model.model)):
        t_prefix = measure_prefix_latency(i)
        delta = max(0 , t_prefix - prev_prefix_time)
        layer_times.append(round(delta, 4))

        print(
            f"Layer {i:02d} | "
            f"time = {delta:7.3f} ms | "
            f"shape = {layer_shapes[i]}"
        )

        prev_prefix_time = t_prefix

    return layer_times, layer_outputs, layer_shapes


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    layer_times, layer_outputs, layer_shapes = profile_yolo_layers(
        model_path="yolo11n.pt",
        input_shape=(1, 3, 640, 640),
        num_runs=30,
        fp16=True,
        move_output_to_cpu=True,  # BẮT BUỘC nếu Jetson
    )

    # print("\n=== SUMMARY ===")
    # for i in range(len(layer_times)):
    #     print(
    #         f"Layer {i:02d}: "
    #         f"time = {layer_times[i]:6.3f} ms | "
    #         f"shape = {layer_shapes[i]}"
    #     )
