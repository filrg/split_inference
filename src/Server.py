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

from src.partition.controller import Controller
from src.partition.handle_data import Data
from src.partition.dijkstra import Dijkstra
from src.Utils import get_layer_output , get_log , save_log , save_partition_cluster , read_partition_cluster
# from src.Log import log_debug


@dataclass
class Cluster:
    device_names: list = field(default_factory=list)
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    data: dict = field(default_factory=dict)
    data_names: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)

class Server:
    def __init__(self, config ):
        self.config = config

        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_frame = config["server"]["batch-frame"]
        self.split_point = {}

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.lst_devices = [0, [0] , [0]]   # [0 , [list stage 1 id ] , [list stage 2 id ]]

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
        self.n_cluster = config["clustering"]["num_clusters"]

        # fine tune code variables
        self.data_clients = {}  # storing all data of clients ( overview )



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
                self.storing_data(message , verbose=False)
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

            # check the number of clients each stage
            self.register_clients[layer_id-1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:

                self.logger.log_debug('DATA USE FOR CLUSTER BEFORE HANDLE ')
                self.logger.log_debug(f'lst_devices {self.lst_devices}')
                self.logger.log_debug(f'data_clients {self.data_clients}')

                self.logger.log_debug('AFTER CLUSTERING  \n')
                cluster = Clustering(
                    lst_devices = self.lst_devices ,
                    data_clients = self.data_clients,
                    n_cluster = self.n_cluster
                )
                res = cluster.run()
                self.logger.log_debug(f'RES {res}')
                for client_id in self.data_clients.keys():
                    self.data_clients[client_id]['cluster'] = res[client_id]

                self.count_num_edges()
                self.count_num_clouds()

                self.logger.log_debug('DATA CLIENTS AFTER CLUSTERING ')
                self.logger.log_debug(self.data_clients)

                # send notify to clients include cluster id and partition point
                self.notify_clients()

                self.logger.log_debug('SENT NOTIFY TO CLIENTS ')

            else:
                print('not matching register clients and total clients ')
                print(f'register clients{self.register_clients}')
                self.logger.log_debug(f'total client {self.total_clients}')

        ch.basic_ack(delivery_tag=method.delivery_tag)

    # [ Utils ] start
    def storing_data(self , dict_data , verbose = True ):
        if verbose :
            self.logger.log_debug(f' === > dict_data before run storing_data function \n {dict_data}')
        new_key = str(dict_data['client_id'])
        info = {}
        info['device'] = dict_data['device']
        info['stage'] = dict_data['layer_id']
        self.data_clients[new_key] = info
        # push to list state
        self.lst_devices[info['stage']].append(new_key)

        if verbose :
            print(f'DATATYPE OF KEY {type(new_key)}')
            print(f'Check data clients \n {self.data_clients}')
            print(f'\nCheck lst devices info \n {self.lst_devices}')

    def count_num_edges(self):
        self.logger.log_debug(f'data clients {self.data_clients}')
        cnt = [0] * (self.n_cluster + 1 )
        for client_id in self.data_clients:
            if self.data_clients[client_id]['stage'] == 1:
                self.logger.log_debug(f'Check cluster {self.data_clients[client_id]['cluster']}')
                cnt[self.data_clients[client_id]['cluster']] += 1

        self.logger.log_debug(f'Check count num edges devices \n {cnt}')

        for client_id in self.data_clients:
            self.data_clients[client_id]['num_edges'] = cnt[self.data_clients[client_id]['cluster']]

        self.logger.log_debug(f'Check num edges devices \n {self.data_clients}')

    def count_num_clouds(self):
        self.logger.log_debug(f'data clients {self.data_clients}')
        cnt = [0] * (self.n_cluster + 1 )
        for client_id in self.data_clients:
            if self.data_clients[client_id]['stage'] == 2:
                self.logger.log_debug(f'Check cluster {self.data_clients[client_id]['cluster']}')
                cnt[self.data_clients[client_id]['cluster']] += 1

        self.logger.log_debug(f'Check count num clouds devices \n {cnt}')

        for client_id in self.data_clients:
            self.data_clients[client_id]['num_clouds'] = cnt[self.data_clients[client_id]['cluster']]

        self.logger.log_debug(f'Check num clouds devices \n {self.data_clients}')

    def update_list(self , raw_list , para):
        for i in range(len(raw_list)):
            raw_list[i] = raw_list[i] * para
        return raw_list

    def assign_response(self , response , client_id ):
        response['cluster_id'] = self.data_clients[client_id]['cluster']
        response['num_edge_layer_1'] = self.data_clients[client_id]['num_edges']
        response['num_clouds'] = self.data_clients[client_id]['num_clouds']

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.logger.log_debug(f'QUEUE_NAME {reply_queue_name}')
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        # self.logger.log_debug(f"[SPLIT_POINT] {self.split_point}")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    # end
    def start(self):
        self.channel.start_consuming()

    def notify_clients(self):
        file_path = f"{self.model_name}.pt"
        if os.path.exists(file_path):
            src.Log.print_with_color(f"Load model {self.model_name}.", "green")
            with open(f"{self.model_name}.pt", "rb") as f:
                file_bytes = f.read()
                encoded = base64.b64encode(file_bytes).decode('utf-8')
        else:
            src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
            sys.exit()

        response = {"action": "START",
                    "message": "Server accept the connection",
                    "model": encoded,
                    "splits": "None",
                    "save_layers": "None",
                    "batch_frame": self.batch_frame,
                    "num_layers": len(self.total_clients),
                    "model_name": self.model_name,
                    "data": self.data,
                    "debug_mode": self.debug_mode,
                    "compress": self.compress,
                    "cal_map": self.cal_map,
                    "cluster_id": 0 ,    # setup below
                    "num_edge_layer_1": 0 ,
                    "num_clouds" : 0
                    }

        self.logger.log_debug(
            f'\n RESULT \n {self.cluster.result} \n --------------- \n'
        )
        self.logger.log_debug(
            f'\n LIST CLIENT  \n {self.list_clients} \n --------------- \n '
            f'{type(self.list_clients[0][0])}'
        )

        if self.config["partition"]["auto"]:
            if self.config["partition"]["re-measure"]:
                """
                Remeasure mode
                After clustering :
                if re-measure mode : 
                    1. Send respond to clients with ['splits'] : 'remeasure' . Then clients run measure mode 
                    2. Get data and run dijkstra algorithm .
                    3. Send only split point and save layers to clients .
                else :
                    1. Get from json file 
                    2. Send response to each client 
                """
                # 1.a choose 2 clients are 1 edge and 1 cloud each cluster to partition .
                # 1.b send 'remeasure' for clients remeasure , 'wait' for clients
                self.logger.log_debug('RE-MEASURE MODE ! ')
                checker = [0] * (self.n_cluster + 1)    # 0 , 1 for edges , 2 for cloud
                for client_id in self.data_clients.keys():
                    cluster_id = self.data_clients[client_id]['cluster']
                    if checker[cluster_id] == 0 or checker[cluster_id] + self.data_clients[client_id]['stage'] == 3:
                        response['splits'] = 'remeasure'
                        checker[cluster_id] += self.data_clients[client_id]['stage']
                    else :
                        response['splits'] = 'wait'

                    self.assign_response(response , client_id)
                    self.send_to_response(str(client_id), pickle.dumps(response))

                # 2. cluster for each pair of each cluster
                n_edges_cluster = [0] * (self.n_cluster + 1 )
                n_clouds_cluster = [0] * (self.n_cluster + 1 )
                for client_id in self.data_clients:
                    n_edges_cluster[self.data_clients[client_id]['cluster']] = self.data_clients[client_id]['num_edges']
                    n_clouds_cluster[self.data_clients[client_id]['cluster']] = self.data_clients[client_id]['num_clouds']

                for cluster_id in range(1 , self.n_cluster + 1):
                    self.logger.log_debug(f'START RE-MEASURE MODE for cluster {cluster_id}')
                    data = Controller(self.config, level=cluster_id).run()

                    layer_times = data["layer_times"]
                    layer_times[1] = self.update_list(layer_times[1] , n_edges_cluster[cluster_id] / n_clouds_cluster[cluster_id])

                    comm_times = data["comm_times"]
                    tmp_comm_times = self.update_list(comm_times , n_clouds_cluster[cluster_id] / n_edges_cluster[cluster_id])
                    comm_times = min(comm_times , tmp_comm_times)

                    cost = Data(layer_times, comm_times, data["name_devices"], verbose=False).run()
                    dijkstra_app = Dijkstra(cost, data["name_devices"])
                    splits = get_layer_output(dijkstra_app.run())
                    src.Log.print_with_color(f"[***] Split point {splits} of {cluster_id}", "blue")
                    res = {
                        "splits": splits[0],
                        "save_layers": splits[1]
                    }
                    self.split_point[cluster_id] = res

                # 3
                for client_id in self.data_clients.keys():
                    split_pt = self.split_point[self.data_clients[client_id]['cluster']]
                    self.send_to_response(str(client_id), pickle.dumps(split_pt))

                save_partition_cluster(self.split_point)

            else :
                self.split_point = read_partition_cluster()
                # response['splits'] = self.split_point[self.data_clients[client_id]['cluster']]
                for client_id in self.data_clients.keys():
                    self.assign_response(response, client_id)
                    response['splits'] = self.split_point[self.data_clients[client_id]['cluster']]['splits']
                    response['save_layers'] = self.split_point[self.data_clients[client_id]['cluster']]['save_layers']
                    self.send_to_response(str(client_id), pickle.dumps(response))

        else :
            default_splits = {
                        "a": (4, [3]),
                        "b": (11, [4, 6, 10]),
                        "c": (17, [10, 13, 16]),
                        "d": (23, [16, 19, 22])
                    }

            self.split_point = default_splits[self.cut_layer]
            # response['splits'] = self.split_point[self.data_clients[client_id]['cluster']]
            for client_id in self.data_clients.keys():
                self.assign_response(response, client_id)
                response['splits'] = self.split_point[0]
                response['save_layers'] = self.split_point[1]
                self.send_to_response(str(client_id), pickle.dumps(response))

