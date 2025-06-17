import threading
import queue
from queue import Full
import pickle
import time
from tqdm import tqdm
import torch
import cv2
from src.Model import SplitDetectionPredictor
from src.Communication import sender_thread, receiver_thread


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

    def send_next_layer(self, intermediate_queue, data):
        if data != 'STOP':
            data["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in data["layers_output"]]
            message = pickle.dumps({
                "action": "OUTPUT",
                "data": data
            })

            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )
        else:
            message = pickle.dumps(data)
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )

    def first_layer(self, model, data, save_layers, batch_frame, logger):
        queue_out = queue.Queue(maxsize=10)
        t_send = threading.Thread(target=sender_thread, args=(queue_out,self.intermediate_queue, self.channel,))
        t_send.daemon = True
        t_send.start()

        time_inference = 0
        input_image = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)
        video_path = data
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.log_error(f"Not open video")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.log_info(f"FPS input: {fps}")
        path = None
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                y = 'STOP'
                while True:
                    try:
                        queue_out.put(y, block=False)
                        break
                    except Full:
                        time.sleep(0.5)
                break
            frame = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1)  # shape: (3, 640, 640)
            tensor /= 255.0
            input_image.append(tensor)

            if len(input_image) == batch_frame:
                input_image = torch.stack(input_image)
                input_image = input_image.to(self.device)
                # Prepare data
                predictor.setup_source(input_image)
                for predictor.batch in predictor.dataset:
                    path, input_image, _ = predictor.batch

                # Preprocess
                preprocess_image = predictor.preprocess(input_image)

                # Head predict
                y = model.forward_head(preprocess_image, save_layers)
                # if save_output:
                #     y["img"] = preprocess_image
                #     y["orig_imgs"] = input_image
                #     y["path"] = path
                time_inference += (time.time() - start)
                # self.send_next_layer(self.intermediate_queue, y)
                while True:
                    try:
                        queue_out.put(y, block=False)
                        break
                    except Full:
                        time.sleep(0.5)
                input_image = []
                pbar.update(batch_frame)
            else:
                continue

        cap.release()
        pbar.close()
        logger.log_info(f"End Inference.")
        return time_inference

    def last_layer(self, model, batch_frame, logger):
        previous_queue = f"intermediate_queue_{self.layer_id - 1}"
        queue_in = queue.Queue(maxsize=10)
        t_send = threading.Thread(target=receiver_thread, args=(queue_in, previous_queue,))
        t_send.daemon = True
        t_send.start()

        time_inference = 0
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)

        # self.channel.queue_declare(queue=last_queue, durable=False)
        # self.channel.basic_qos(prefetch_count=50)
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            if not queue_in.empty():
                received_data = queue_in.get()
                if received_data != 'STOP':
                    y = received_data["data"]
                    y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]
                    start = time.time()
                    # Tail predict
                    predictions = model.forward_tail(y)

                    # Postprocess
                    # if save_output:
                    #     results = predictor.postprocess(predictions, y["img"], y["orig_imgs"], y["path"])
                    time_inference += (time.time() - start)
                    pbar.update(batch_frame)
                else:
                    break
            else:
                time.sleep(0.01)
        pbar.close()
        logger.log_info(f"End Inference.")
        return time_inference

    def middle_layer(self, model):
        pass

    def inference_func(self, model, data, num_layers, save_layers, batch_frame, logger):
        time_inference = 0
        if self.layer_id == 1:
            time_inference = self.first_layer(model, data, save_layers, batch_frame, logger)
        elif self.layer_id == num_layers:
            time_inference = self.last_layer(model, batch_frame, logger)
        else:
            self.middle_layer(model)
        return time_inference
