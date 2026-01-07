import os
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn

import src.Model
import src.Log
from ultralytics import YOLO
from dataclasses import dataclass , field
import numpy as np
from src.clustering.clustering import Clustering
from collections import defaultdict
# from src.Log import log_debug


@dataclass
class Cluster:
    device_names: list = field(default_factory=list)
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    data: dict = field(default_factory=dict)
    data_names: dict = field(default_factory=dict)
    num_edges: dict = field(default_factory=dict)

class Server:
    def __init__(self, config , split_point ):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_frame = config["server"]["batch-frame"]
        self.split_point = split_point

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.data = config["data"]
        self.debug_mode = config["debug-mode"]
        self.compress = config["compress"]
        self.cal_map = config["cal_map"]

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log" , debug_mode = self.debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

        # Handle message for clustering
        self.cluster = Cluster()
        self.cluster.data_names = defaultdict(list)
        self.result = {}
        self.n_cluster = config["clustering"]["num_clusters"]



    def on_request(self, ch: object, method: object, props: object, body: object) -> object:
        message = pickle.loads(body)
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

                # Handle message for Clustering
                self.cluster.data_names[layer_id].append(message['client_id'])
                # 1. Extract the data into a list
                lst_data = [val for _, val in message['device'].items()]
                new_row = np.asarray(lst_data)
                layer_id = int(message["layer_id"])
                self.logger.log_debug("New row data " + str(new_row))
                self.logger.log_debug("Layer id " + str( layer_id))

                if layer_id not in self.cluster.data:
                    # Use [new_row] or reshape to make it 2D (1 row, N columns)
                    self.cluster.data[layer_id] = np.array([new_row]).reshape(1 , -1)
                else:
                    # Stack the new row vertically
                    self.cluster.data[layer_id] = (
                        np.vstack((self.cluster.data[layer_id], new_row))
                    )

            self.logger.log_debug("Check output cluster data " , self.cluster.data)
            self.logger.log_debug("Check output cluster features " , self.cluster.features)
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")


            # Save messages from clients
            self.register_clients[layer_id-1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                # notify get info from devices successfully
                self.logger.log_debug('Check data output :')
                self.logger.log_debug('device names : ', self.cluster.data_names)
                self.logger.log_debug('data : ' , self.cluster.data)

                for layer , features in self.cluster.data.items():
                    cluster = Clustering(features ,
                                         self.cluster.data_names[layer] ,
                                         n_clusters = self.n_cluster
                                         )
                    res = cluster.run()
                    self.logger.log_debug('result of cluster ' , res )
                    for key , lst_id in res.items():
                        for id in lst_id :
                            self.result[id.hex] = key

                    if layer == 1 :
                        self.cluster.num_edges[int(layer)] = len(features)
                self.logger.log_debug("[result]")
                self.logger.log_debug(self.result)
                self.logger.log_debug("[num edges]")
                self.logger.log_debug(self.cluster.num_edges)

                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()




        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.logger.log_debug(f"[SPLIT_POINT] {self.split_point}")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def start(self):
        self.channel.start_consuming()

    def notify_clients(self):
        default_splits = {
            "a": (4, [3]),
            "b": (11, [4, 6, 10]),
            "c": (17, [10, 13, 16]),
            "d": (23, [16, 19, 22])
        }
        # model = YOLO(f"{self.model_name}.pt")
        # splits = default_splits[self.cut_layer]
        splits = self.split_point
        if splits == -1 :
            splits = default_splits[self.cut_layer]
        self.logger.log_debug(f"splits : {splits}")
        file_path = f"{self.model_name}.pt"
        if os.path.exists(file_path):
            src.Log.print_with_color(f"Load model {self.model_name}.", "green")
            with open(f"{self.model_name}.pt", "rb") as f:
                file_bytes = f.read()
                encoded = base64.b64encode(file_bytes).decode('utf-8')
        else:
            src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
            sys.exit()

        for (client_id, layer_id) in self.list_clients:

            lst_keys = list(self.result.keys())
            self.logger.log_debug(f"check type list key 0 {type(lst_keys[0])} of {lst_keys[0]}")
            self.logger.log_debug(f"client id {client_id} of {type(client_id)}")
            level = 0
            for key , val in self.result.items():
                if client_id.replace("-", "") in key :
                    level = val

            self.logger.log_debug(f"level {level} , {type(level)}")
            self.logger.log_debug(f"Num edge at layer id 1 {self.cluster.num_edges[1]} type {type(self.cluster.num_edges[1])}")

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "model": encoded,
                        "splits": splits[0],
                        "save_layers": splits[1],
                        "batch_frame": self.batch_frame,
                        "num_layers": len(self.total_clients),
                        "model_name": self.model_name,
                        "data": self.data,
                        "debug_mode": self.debug_mode,
                        "compress": self.compress,
                        "cal_map": self.cal_map,
                        "level": level,
                        "num_edge_layer_1": self.cluster.num_edges[1]}

            self.send_to_response(client_id, pickle.dumps(response))
