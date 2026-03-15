import pickle
from tqdm import tqdm
import torch
import cv2
from src.Compress import Encoder,Decoder
import src.Log as Log
import copy
from src.Model import inference, postprocess_yolo, draw_img

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.size_message = None

        self.current_time = None
        self.previous_time = None

        # partition and clustering
        self.my_cluster = 1
        self.intermediate_queue = f"intermediate_queue_{self.my_cluster}"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

        self.rpc_queue = f"rpc_queue"
        self.channel.queue_declare(self.rpc_queue, durable=False)

    def send_next_layer(self, intermediate_queue, data, compress):
        if data != 'STOP':
            if compress["enable"]:
                data["data"] = [t.cpu().numpy() if isinstance(t, torch.Tensor) else None for t in
                                         data["data"]]
                data["data"], data["shape"] = Encoder(data_output=data["data"], num_bits=compress["num_bit"])

                # frame index
                # cluster index

            else:
                data["data"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in
                                         data["data"]]
            message = pickle.dumps({
                "action": "OUTPUT",
                "data": data
            })
            if self.size_message is None:
                self.size_message = len(message)
        else:
            message = pickle.dumps(data)

        self.channel.basic_publish(
            exchange='',
            routing_key=intermediate_queue,
            body=message,
        )

    def send_to_server(self , mess):
        message = {
            'action':mess
        }
        message = pickle.dumps(message)

        self.channel.basic_publish(
            exchange='',
            routing_key=self.rpc_queue,
            body=message,
        )

    def first_layer(self, model, data, batch_size, logger, compress):
        orig_images = []
        input_image = []
        model.eval()
        model.to(self.device)

        video_path = data
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            Log.print_with_color(f"Not open video", "red")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                self.send_to_server('STOP')
                break
            frame = cv2.resize(frame, (640, 640))
            orig_images.append(copy.deepcopy(frame))
            frame = frame.astype('float16') / 255.0
            tensor = torch.from_numpy(frame).permute(2, 0, 1)  # shape: (3, 640, 640)
            input_image.append(tensor)

            if len(input_image) == batch_size:
                input_image = torch.stack(input_image)
                input_image = input_image.to(self.device)
                y = []
                x, y = inference(model, input_image, y, 0)
                y[-1] = x

                y = {"data": y,
                     "width": width,
                     "height": height
                     }

                self.send_next_layer(self.intermediate_queue, y, compress)
                input_image = []
                orig_images = []
                pbar.update(batch_size)
            else:
                continue
        print(f'size message: {self.size_message} bytes.')
        cap.release()
        pbar.close()


    def last_layer(self, model, batch_size, splits, logger, compress):
        num_last = 1
        count = 0

        model.eval()
        model.to(self.device)

        self.channel.basic_qos(prefetch_count=10)
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=self.intermediate_queue, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"]
                    if compress["enable"]:
                        y["data"] = Decoder(y["data"], y["shape"])
                        y["data"] = [torch.from_numpy(t) if t is not None else None for t in y["data"]]

                    y["data"] = [t.to(self.device) if t is not None else None for t in y["data"]]
                    list_output = y["data"]
                    x = list_output[-1]
                    x, _ = inference(model, x, list_output, splits)

                    results = postprocess_yolo(x)

                    pbar.update(batch_size)
                else:
                    self.send_to_server('STOPPED')
                    count += 1
                    if count == num_last:
                        break
                    continue
            else:
                continue

        # out.release()
        cv2.destroyAllWindows()
        pbar.close()

    def middle_layer(self, model):
        pass

    def inference_func(self, model, data, num_layers, splits , batch_size, logger, compress):
        if self.layer_id == 1:
            self.first_layer(model, data, batch_size, logger, compress)
        elif self.layer_id == num_layers:
            self.last_layer(model, batch_size, splits , logger, compress)
        else:
            self.middle_layer(model)
