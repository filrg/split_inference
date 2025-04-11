import os
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn

import src.Model
import src.Log


class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_frame = config["server"]["batch-frame"]
        self.save_output = config["server"]["save-output"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug-mode"]
        log_path = config["log-path"]
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id-1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()

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
            "a": (10, [4, 6, 9]),
            "b": (16, [9, 12, 15]),
            "c": (22, [15, 18, 21])
        }
        splits = default_splits[self.cut_layer]
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

            response = {"action": "START",
                        "message": "Server accept the connection",
                        "model": encoded,
                        "splits": splits[0],
                        "save_layers": splits[1],
                        "batch_size": self.batch_frame,
                        "num_layers": len(self.total_clients),
                        "model_name": self.model_name,
                        "save_output": self.save_output}

            self.send_to_response(client_id, pickle.dumps(response))
