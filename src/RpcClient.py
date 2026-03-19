import pickle
import time
import base64
import os

import torch
import src.Log as Log

class RpcClient:
    def __init__(self, client_id, layer_id, channel, logger ,inference_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.logger = logger
        self.inference_func = inference_func
        self.device = device

        self.channel = channel
        self.response = None

    def wait_response(self):
        status = True
        reply_queue_name = f"reply_{self.client_id}"
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]

        if action == "START":
            model_name = self.response["model_name"]
            num_layers = self.response["num_layers"]
            splits = self.response["splits"]
            batch_size = self.response["batch_size"]
            model = self.response["model"]
            data = self.response["data"]
            compress = self.response["compress"]

            if model is not None:
                file_path = f'{model_name}.pt'
                if os.path.exists(file_path):
                    Log.print_with_color(f"Exist {model_name}.pt", "green")
                else:
                    decoder = base64.b64decode(model)
                    with open(f"{model_name}.pt", "wb") as f:
                        f.write(decoder)
                    Log.print_with_color(f"Loaded {model_name}.pt", "green")
            else:
                Log.print_with_color(f"Can't load model.", "yellow")

            ckpt = torch.load("yolo26n.pt", map_location=self.device, weights_only=False)
            model = ckpt["model"].to(self.device)
            model = model.float()
            layers = model.model
            if self.layer_id == 1:
                client = layers[:splits]
            else:
                client = layers[splits:]

            Log.print_with_color(f"Start Inference", "green")

            self.inference_func(client, data, num_layers, splits, batch_size, self.logger, compress)

            return False
        else:
            return False

    def send_to_server(self, message):

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

