import cv2
import cv2
from torchvision.ops import nms

save_yolo26 = [4,6,10,13,16,19,22]
input_yolo26 = [None,None,None,None,None,None,None,None,None,None,None,None,[-1,6],None,None,[-1,4],None,None
    ,[-1,13],None,None,[-1,10],None,[16, 19, 22]]

def inference(model, x, y, cut, input_cfg=input_yolo26, save_cfg=save_yolo26):
    """
    model: list of layers
    x: input tensor
    y: list storing intermediate outputs (auto-resized if needed)
    cut: starting index offset
    """

    # ---- Ensure y is large enough ----
    required_len = len(input_cfg) + cut
    if len(y) < required_len:
        y.extend([None] * (required_len - len(y)))

    for i, layer in enumerate(model):
        idx = i + cut

        # ---- Handle multi-input routing ----
        if input_cfg[idx] is not None:
            inputs = input_cfg[idx]

            if inputs[0] == -1:
                prev = y[inputs[1]]
                assert prev is not None, f"Layer {inputs[1]} not saved!"
                x = [x, prev]
            else:
                prevs = []
                for j in inputs:
                    assert y[j] is not None, f"Layer {j} not saved!"
                    prevs.append(y[j])
                x = prevs

        # ---- Fix FP16 + channels_last ----
        if isinstance(x, list):
            x = [t.contiguous() if hasattr(t, "contiguous") else t for t in x]
        else:
            x = x.contiguous()

        # ---- Forward ----
        x = layer(x)

        # ---- Save outputs ----
        y[idx] = x if idx in save_cfg else None

    return x, y
def postprocess_yolo(output, conf_thres=0.1, iou_thres=0.1):
    pred_tensor = output[0]   # [B,N,6]
    batch_results = []
    B = pred_tensor.shape[0]

    for b in range(B):
        pred = pred_tensor[b]      # [N,6]

        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = pred[:, 5].long()

        mask = scores > conf_thres

        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # NMS
        keep = nms(boxes, scores, iou_thres)

        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        batch_results.append({
            "boxes": boxes,
            "scores": scores,
            "classes": classes
        })

    return batch_results

def draw_img(img, r):
    for box, score, cls in zip(r["boxes"], r["scores"], r["classes"]):
        x1, y1, x2, y2 = box.int().tolist()

        conf = score.item()
        cls_id = cls.item()

        label = f"{cls_id}:{conf:.2f}"

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img

