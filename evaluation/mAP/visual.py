import cv2
import os

VIDEO_PATH = "../../videos/video.mp4"
GT_DIR = "../datasets/groundtruth"
PRED_DIR = "../datasets/predictions"

FPS = 25
PREDICT = True
GT = False
BATCH_SIZE = 8  # Define your batch size here


class Visual:
    def __init__(self, fps, gt, batch_size=1):
        self.gt = gt
        self.fps = fps
        self.batch_size = batch_size  # Added batch size

    def yolo_to_xyxy(self, cx, cy, w, h, img_w, img_h):
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        return x1, y1, x2, y2

    def load_boxes(self, file_path, img_w, img_h, is_prediction=False):
        boxes = []
        if not os.path.exists(file_path):
            return boxes

        with open(file_path) as f:
            for line in f:
                parts = line.strip().split()
                if is_prediction:
                    cls, cx, cy, w, h, conf = map(float, parts)
                else:
                    cls, cx, cy, w, h = map(float, parts)

                x1, y1, x2, y2 = self.yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                boxes.append((int(cls), x1, y1, x2, y2))
        return boxes

    def run_mAP(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        frame_id = 1

        while True:
            batch_frames = []

            for _ in range(self.batch_size):
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                img_h, img_w = frame.shape[:2]

                gt_file = os.path.join(GT_DIR, f"frame_{frame_id:06d}.txt")
                pred_file = os.path.join(PRED_DIR, f"frame_{frame_id:06d}.txt")

                gt_boxes = self.load_boxes(gt_file, img_w, img_h, False)
                pred_boxes = self.load_boxes(pred_file, img_w, img_h, True)

                # draw ground truth (green)
                if self.gt:
                    for cls, x1, y1, x2, y2 in gt_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # draw predictions (blue)
                if PREDICT:
                    for cls, x1, y1, x2, y2 in pred_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{cls}", (x1 + 20, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.putText(frame, f"Frame: {frame_id}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                batch_frames.append(frame)
                frame_id += 1

            # If the batch is empty, we reached the end of the video
            if not batch_frames:
                break

            # Display the batch
            quit_playback = False
            for annotated_frame in batch_frames:
                cv2.imshow("GT vs Prediction", annotated_frame)

                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    quit_playback = True
                    break

            if quit_playback:
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_tracker(self):
        pass


visual = Visual(FPS, gt=GT, batch_size=5)
visual.run_mAP()