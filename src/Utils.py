import pika
from requests.auth import HTTPBasicAuth
import requests
import os
import numpy as np
import cv2
import pandas as pd
import csv
import yaml


def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "result") or queue_name.startswith("rpc_queue"):

                http_channel.queue_delete(queue=queue_name)

            else:
                http_channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    else:
        return False

def compute_iou(box1, box2):
    """Compute IoU"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0.0

def compute_ap(tp, fp, total_gt):
    tp = np.array(tp)
    fp = np.array(fp)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (total_gt + 1e-6)
    ap = 0.0
    for i in range(len(precision)):
        if i == 0 or recall[i] != recall[i - 1]:
            delta_r = recall[i] - recall[i - 1] if i > 0 else recall[i]
            ap += precision[i] * delta_r
    return ap

def compute_map(preds, gts, iou_threshold=0.1):
    from collections import defaultdict
    preds_by_class = defaultdict(list)
    gts_by_class = defaultdict(lambda: defaultdict(list))

    for img_id, cls, x1, y1, x2, y2 in gts:
        gts_by_class[int(cls)][img_id].append([x1, y1, x2, y2])

    for img_id, cls, x1, y1, x2, y2, conf in preds:
        preds_by_class[int(cls)].append((img_id, [x1, y1, x2, y2], float(conf)))

    ap_list = []
    for cls in sorted(preds_by_class.keys()):
        # sắp xếp lại theo confidence
        detections = sorted(preds_by_class[cls], key=lambda x: -x[2])
        gt_class = gts_by_class[cls]
        tp, fp = [], []
        matched = defaultdict(set)
        # tổng tất cả các ground truth của cls
        total_gt = sum(len(boxes) for boxes in gt_class.values())
        for img_id, box_pred, _ in detections:
            matched_gt_boxes = gt_class.get(img_id, [])
            # match_gt_boxes: tất cả các gt có nhãn là cls trong img_id
            ious = [compute_iou(box_pred, gt_box) for gt_box in matched_gt_boxes]
            #ious: list các IoU tính được giữa dự đoán và gt trong img_id
            if ious:
                max_iou = max(ious)
                max_idx = np.argmax(ious)
                if max_iou >= iou_threshold and max_idx not in matched[img_id]:
                    tp.append(1)
                    fp.append(0)
                    matched[img_id].add(max_idx)
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                # trong img_id không có nhãn đó
                tp.append(0)
                fp.append(1)
        ap = compute_ap(tp, fp, total_gt)
        ap_list.append(ap)
    return np.mean(ap_list) if ap_list else 0.0

def load_ground_truth(label_dir, image_dir):
    """Tạo gts từ thư mục label"""
    gts = []
    for file in sorted(os.listdir(label_dir)):
        if not file.endswith(".txt"):
            continue
        image_id = os.path.splitext(file)[0]
        label_path = os.path.join(label_dir, file)
        img_path = os.path.join(image_dir, image_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gts.append([image_id, int(cls), x1, y1, x2, y2])
    return gts


""" write to csv file """
def list_to_dict_with_minus_one(lst):
    return {key: -1 for key in lst}

cols = [
    "Time" , "PointCut" ,
    "[T]totalTM" , "[T]FPSR" ,
    "[1]totalFr" , "[2]totalTm" , "[1]totalTm",
    "[1]outSze[T]" , "[1]outSze[2]" ,"[2]outSize",
    "[1]GPU_time" , "[2]GPU_time" ,
    "[1]peak_RAM" , "[2]peak_RAM" ,
    "[1]peak_VRAM" , "[2]peak_VRAM" ,
]
# cols = [
#  'time', 'pc',
#  '[T]tm', '[T]fps',
#  '[1]fr', '[2]tm', '[1]tm',
#  '[1]outT', '[1]out2', '[2]out',
#  '[1]gpu', '[2]gpu',
#  '[1]ram', '[2]ram',
#  '[1]vram', '[2]vram'
# ]


dict_data = list_to_dict_with_minus_one(cols)
file_path = "output.csv"

row_buffer = {}

def write_partial(partial_data, flush=False):
    global row_buffer, cols

    # exist output.csv = run this function
    update_csv_header(file_path ,cols)

    row_buffer.update(partial_data)

    new_cols = [c for c in partial_data.keys() if c not in cols]
    if new_cols:
        cols.extend(new_cols)  # add new columns
        # reload CSV with new header
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for c in new_cols:
                df[c] = ""  # add empty column for past rows
            df.to_csv(file_path, index=False)

    if all(col in row_buffer for col in cols):
        flush = True

    if flush:
        row_df = pd.DataFrame([row_buffer], columns=cols)

        if not os.path.exists(file_path):
            row_df.to_csv(file_path, index=False)
        else:
            row_df.to_csv(file_path, mode='a', index=False, header=False)

        row_buffer = {}
        print("[CSV] write csv successfully !")

def update_csv_header(filename, new_headers):
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError("CSV file is empty!")

    # Replace only the first row
    reader[0] = new_headers

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(reader)

def format_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"

def get_layer_output(cut_point , yaml_file = "cfg/yolo11n.yaml"):
    """
    Parse YOLO config YAML and return list of 'from' indices for all layers.
    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    res = [cut_point - 1]
    from_idx = []
    # backbone and head sections
    for section in ["backbone", "head"]:
        for i, layer in enumerate(config.get(section, [])):
            from_idx.append(layer[0])

    # return from_idx

    for i in range(cut_point , len(from_idx)):
        # print(type(from_idx[0]))
        if isinstance(from_idx[i] , list):
            for j in from_idx[i]:
                if j != -1 and j not in res and j < cut_point :
                    res.append(j)
    res.sort()
    return tuple((cut_point , res))

def get_output_sizes(cfg_path, img_size=(640, 640)):
    """
    Estimate output tensor sizes (MB) for each layer defined in YOLOv11 YAML.
    - Works without torch.
    - Assumes float32 (4 bytes).
    - Detect layer returns 0.0 placeholder unless anchors/nc are specified.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dtype_size = 4
    H, W = img_size
    C = 3  # RGB input
    sizes = []
    saved = {}

    # YOLOv11: layers are in backbone + head
    for idx, layer in enumerate(cfg["backbone"] + cfg["head"]):
        from_idx, repeats, module, args = layer
        module = str(module)

        if module in ("Conv", "ConvBnAct"):
            C = args[0]
            stride = args[2] if len(args) > 2 else 1
            H //= stride
            W //= stride
        elif module in ("C2f", "C3", "C3k", "C3k2", "C3TR", "C2PSA"):
            C = args[0]
        elif module == "SPPF":
            C = args[0]
        elif "Upsample" in module:
            scale = args[1] if len(args) > 1 else 2
            H *= scale
            W *= scale
        elif module == "Concat":
            total_C = 0
            for src in from_idx:
                if src == -1:
                    total_C += saved[idx - 1][0]
                elif isinstance(src, int):
                    total_C += saved[src][0]
                else:
                    raise ValueError(f"Concat source not handled: {src}")
            C = total_C
        elif module == "Detect":
            sizes.append(0.0)  # can't infer from YAML
            saved[idx] = (C, H, W)
            continue
        else:
            # unknown block → keep previous channels
            pass

        size_mb = round(C * H * W * dtype_size / (1024 * 1024), 2)
        sizes.append(size_mb)
        saved[idx] = (C, H, W)

    for i in range(0 ,len(sizes)):
        from_idx = get_layer_output(i)[1]
        if len(from_idx) == 1 :
            continue
        temp = 0
        for j in range(len(from_idx)):
            temp = temp + sizes[from_idx[j] + 1]
        sizes[i] = temp
    return sizes



