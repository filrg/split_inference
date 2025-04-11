import pickle
import time
import torch
import src.Log
import cv2
from PIL import Image
import torchvision.transforms as transforms
from src.Model import SplitDetectionPredictor


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device

    def send_next_layer(self, data, save_output=False):
        intermediate_queue = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(intermediate_queue, durable=False)
        if save_output:
            message = pickle.dumps({
                "action": "SAVE",
                "data": data
            })
        else:
            message = pickle.dumps({
                "action": "OUTPUT",
                "data": data
            })
            bit_size = len(message)
            print(f"Size: {bit_size} bites.")

        self.channel.basic_publish(
            exchange='',
            routing_key=intermediate_queue,
            body=message
        )

    def first_layer(self, model, save_layers, batch_frame, save_output):
        input_image = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)
        video_path = "video.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            src.Log.print_with_color(f"Not open video", "yellow")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        path = None

        self.send_next_layer({"fps": fps, "width": width, "height": height}, True)

        try:
            while True:
                # queue = self.channel.queue_declare(queue=f"intermediate_queue_{self.layer_id}", passive=True)
                # message_count = queue.method.message_count
                # if message_count > 50:
                #     time.sleep(0.1)
                #     continue
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    src.Log.print_with_color(f"Not read from video", "yellow")
                    return False
                frame = cv2.resize(frame, (640, 640))
                tensor = torch.from_numpy(frame).float().permute(2, 0, 1)  # shape: (3, 640, 640)
                tensor /= 255.0
                input_image.append(tensor)
                # input_image = tensor.unsqueeze(0)
                if len(input_image) == batch_frame:
                    input_image = torch.stack(input_image)
                    # Prepare data
                    predictor.setup_source(input_image)
                    for predictor.batch in predictor.dataset:
                        path, input_image, _ = predictor.batch

                    # Preprocess
                    preprocess_image = predictor.preprocess(input_image)
                    if isinstance(input_image, list):
                        input_image = np.array([np.moveaxis(img, -1, 0) for img in input_image])

                    # Head predict
                    y = model.forward_head(preprocess_image, save_layers)
                    if save_output:
                        y["img"] = preprocess_image
                        y["orig_imgs"] = input_image
                        y["path"] = path

                    stop = time.time()
                    print(stop - start)
                    self.send_next_layer(y, False)
                    input_image = []
                else:
                    continue
        except Exception as e:
            src.Log.print_with_color(f"Error: {e}", "yellow")
        finally:
            cap.release()

    def last_layer(self, model, save_output=False):
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = None
        width = None
        height = None
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=10)

        try:
            while True:
                method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True)
                if method_frame and body:
                    start = time.time()
                    received_data = pickle.loads(body)
                    if received_data['action'] == "SAVE":
                        data = received_data["data"]
                        fps = data['fps']
                        width = data['width']
                        height = data['height']
                        video = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
                    else:
                        y = received_data["data"]

                        # Tail predict
                        predictions = model.forward_tail(y)

                        # Postprocess
                        if save_output:
                            results = predictor.postprocess(predictions, y["img"], y["orig_imgs"], y["path"])
                            for res in results:
                                annotated_frame = res.plot()
                                # cv2.imshow("YOLOv8n - Object Detection", annotated_frame)
                                video.write(cv2.resize(annotated_frame, (width, height)))
                        stop = time.time()
                        print(stop - start)

                else:
                    continue

        except Exception as e:
            src.Log.print_with_color(f"Error: {e}", "yellow")
        finally:
            video.release()
            cv2.destroyAllWindows()

    def middle_layer(self, model):
        pass

    def inference_func(self, model, num_layers, save_layers, batch_frame, save_output):
        if self.layer_id == 1:
            self.first_layer(model, save_layers, batch_frame, save_output)
        elif self.layer_id == num_layers:
            self.last_layer(model, save_output)
        else:
            self.middle_layer(model)
