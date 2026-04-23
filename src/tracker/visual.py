import cv2
import numpy as np


class Visual_tracker:
    def __init__(self, fps=25 , red_point = True):
        self.fps = fps
        self.frame_id = 1
        self.red_point = red_point

    def yolo_to_xyxy(self, cx, cy, w, h, img_w, img_h):
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        return x1, y1, x2, y2

    def run(self, origin_frame_test, raw_prediction_tensor):
        if origin_frame_test is None:
            return False

        frame = np.ascontiguousarray(origin_frame_test, dtype=np.uint8)
        img_h, img_w = frame.shape[:2]

        bboxes = raw_prediction_tensor
        drawn_count = 0

        if bboxes is not None and isinstance(bboxes, list):
            for item in bboxes:
                try:
                    if isinstance(item, dict):
                        cls = item.get('class', -1)
                        conf = item.get('conf', 0.0)
                        bbox = item.get('bbox', [])

                        if len(bbox) >= 4:
                            cx = float(bbox[0])
                            cy = float(bbox[1])
                            w = float(bbox[2])
                            h = float(bbox[3])

                            x1, y1, x2, y2 = self.yolo_to_xyxy(cx, cy, w, h, img_w, img_h)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (70 , 130, 180), 2)

                            # red point at center
                            if self.red_point:
                                center_x = int(cx * img_w)
                                center_y = int(cy * img_h)
                                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                            label = f"ID:{int(cls)} {conf:.2f}"
                            cv2.putText(frame, label, (x1, max(y1 - 5, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                            drawn_count += 1
                except Exception as e:
                    pass

        info_text = f"Frame: {self.frame_id} | Boxes: {drawn_count}"
        cv2.putText(frame, info_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Visual from Tracker", frame)
        self.frame_id += 1

        delay = int(1000 / self.fps) if drawn_count > 0 else 1
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False

        return True