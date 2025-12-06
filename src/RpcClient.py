import pickle
import time
import base64
import os

import pika
import torch
import torch.nn as nn

import src.Log
from src.Model import SplitDetectionModel
from ultralytics import YOLO

class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, virtual_host, inference_func, check_compress_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.inference_func = inference_func
        self.check_compress_func = check_compress_func
        self.device = device

        self.channel = None
        self.connection = None
        self.response = None
        self.model = None
        self.data = None
        self.logger = None
        self.connect()

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
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]

        if action == "START":
            model_name = self.response["model_name"]
            num_layers = self.response["num_layers"]
            splits = self.response["splits"]
            save_layers = self.response["save_layers"]
            batch_frame = self.response["batch_frame"]
            model = self.response["model"]
            data = self.response["data"]
            compress = self.response["compress"]
            cal_map = self.response["cal_map"]

            debug_mode = self.response["debug_mode"]

            self.logger = src.Log.Logger(f"res/result.log", debug_mode)
            if model is not None:
                file_path = f'{model_name}.pt'
                if os.path.exists(file_path):
                    src.Log.print_with_color(f"Exist {model_name}.pt", "green")
                else:
                    decoder = base64.b64decode(model)
                    with open(f"{model_name}.pt", "wb") as f:
                        f.write(decoder)
                    src.Log.print_with_color(f"Loaded {model_name}.pt", "green")
            else:
                src.Log.print_with_color(f"Do not load model.", "yellow")

            pretrain_model = YOLO(f"{model_name}.pt").model
            self.model = SplitDetectionModel(pretrain_model, split_layer=splits)
            start = time.time()
            self.logger.log_info(f"Start Inference")
            if cal_map["enable"] is False:
                self.inference_func(self.model, data, num_layers, save_layers, batch_frame, self.logger, compress)
            else:
                self.check_compress_func(self.model, data, num_layers, save_layers, batch_frame, self.logger, compress, cal_map)
            all_time = time.time() - start
            src.Log.print_with_color(f"All time: {all_time}s", 'green')
            # Stop or Error
            return False
        else:
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, self.virtual_host, credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

