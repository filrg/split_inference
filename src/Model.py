import cv2
from torchvision.ops import nms

save_yolo26 = [4,6,10,13,16,19,22]
input_yolo26 = [None,None,None,None,None,None,None,None,None,None,None,None,[-1,6],None,None,[-1,4],None,None
    ,[-1,13],None,None,[-1,10],None,[16, 19, 22]]

def inference(model, x, y, cut):
    for i, layer in enumerate(model):
        idx = i + cut
        if input_yolo26[idx] is not None:
            if input_yolo26[idx][0] == -1:
                x = [x, y[input_yolo26[idx][1]]]
            else:
                x = [y[input_yolo26[idx][0]], y[input_yolo26[idx][1]], y[input_yolo26[idx][2]]]

        x = layer(x)

        if idx in save_yolo26:
            y.append(x)
        else:
            y.append(None)
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

