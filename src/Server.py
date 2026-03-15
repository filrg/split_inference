import os
import sys
import base64
import pika
import pickle

import src.Model
import src.Log
from ultralytics import YOLO
from src.Utils import delete_old_queues

class Server:
    def __init__(self, config):
        # RabbitMQ
        self.address = config["rabbit"]["address"]
        self.username = config["rabbit"]["username"]
        self.password = config["rabbit"]["password"]
        self.virtual_host = config["rabbit"]["virtual-host"]
        
        
        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_size = config["server"]["batch-size"]

        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, f'{self.virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.data = config["data"]
        self.compress = config["compress"]

        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log" , config["debug-mode"])
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

        self.cnt_stop_edges =  0
        self.cnt_stop_clouds = 0

        self.n_cluster =  1
        # for cluster in self.n_cluster :
        self.intermediate_queue = f"intermediate_queue_{self.n_cluster}"
        self.channel.queue_declare(self.intermediate_queue, durable=False)


    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        action = message["action"]

        if action == "REGISTER":
            client_id = message["client_id"]
            stage_id = message["layer_id"]

            if (str(client_id), stage_id) not in self.list_clients:
                self.list_clients.append((str(client_id), stage_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[stage_id-1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()

        elif action == "STOP":
            self.cnt_stop_edges += 1
        
        elif action == "STOPPED":
            self.cnt_stop_clouds += 1
            

        if self.cnt_stop_edges == self.total_clients[0]:
            for _ in range(self.total_clients[1]):
                self.send_stop_signal()

        if self.cnt_stop_clouds == self.total_clients[1]:
            delete_old_queues(self.address, self.username, self.password, self.virtual_host)
            sys.exit(0)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_response(self, client_id, message):
        reply_queue_name = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def start(self):
        self.channel.start_consuming()

    def notify_clients(self):
        default_splits = {
            "a": 4,
            "b": 11,
            "c": 17,
            "d": 23
        }
        if os.path.exists(f"{self.model_name}.pt"):
            src.Log.print_with_color(f"Exist {self.model_name}", "green")
        else:
            src.Log.print_with_color(f"Download {self.model_name}", "yellow")
            model = YOLO(f"{self.model_name}.pt")

        splits = default_splits[self.cut_layer]
        file_path = f"{self.model_name}.pt"
        if os.path.exists(file_path):
            src.Log.print_with_color(f"Send model {self.model_name} to devices.", "green")
            with open(f"{self.model_name}.pt", "rb") as f:
                file_bytes = f.read()
                encoded = base64.b64encode(file_bytes).decode('utf-8')
        else:
            src.Log.print_with_color(f"{self.model_name} does not exist.", "yellow")
            sys.exit()

        for (client_id, stage_id) in self.list_clients:

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "model": encoded,
                        "splits": splits,
                        "batch_size": self.batch_size,
                        "num_layers": len(self.total_clients),
                        "model_name": self.model_name,
                        "data": self.data,
                        "compress": self.compress}

            self.send_to_response(client_id, pickle.dumps(response))

    def send_stop_signal(self):
        message = 'STOP'
        message = pickle.dumps(message)
        self.channel.basic_publish(
            exchange='',
            routing_key=self.intermediate_queue,
            body=message,
        )


