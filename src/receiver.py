import os
import json
import time
import yaml
import pika
import pickle
import socket
from src.Utils import write_partial
from pathlib import Path
from src.Utils import get_output_sizes
from src.time_layers import LayerProfiler

class MessageReceiver:
    def __init__(self, config: dict):
        print("Start Receiver ... ")
        self.config = config
        self.queue_device_1 = config["rabbit"]["queue_device_1"]
        self.queue_device_2 = config["rabbit"]["queue_device_2"]

        credentials = pika.PlainCredentials(
            config["rabbit"]["username"], config["rabbit"]["password"]
        )
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                config["rabbit"]["address"],
                5672,
                config["rabbit"]["virtual_host"],
                credentials,
            )
        )

        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_device_1, durable=True)
        self.channel.queue_declare(queue=self.queue_device_2, durable=True)

        self.host_name = socket.gethostname()

    def send_message(self, messages_dict):
        self.channel.basic_publish(exchange='',
                                   routing_key=self.queue_device_2,
                                   body=pickle.dumps(messages_dict)
                                   )
        # print(f"Sent {messages_dict["message"]}")

    def listening(self ):
        device_queue = self.queue_device_1
        method_frame, header_frame, body = self.channel.basic_get(queue=device_queue, auto_ack=True)
        if method_frame and body:
            data = pickle.loads(body)
            return data
        else :
            return None

    def comm_times(self):
        while True:
            data = self.listening()
            if data is not None:
                if data["signal"] == "yes":
                    self.send_message(data)
                else :
                    break

    def run(self):
        while True:
            self.send_message({
                "signal": "REQUEST",
                "message": "receiver"
            })

            data = self.listening()
            if data is not None:
                if data["signal"] == "START":
                    layer_times_app = LayerProfiler(self.config)
                    res = layer_times_app.run()
                    print(len(res))
                    self.send_message({
                        "signal": "time_layer " + str(self.host_name) ,
                        "message": res
                    })
                print("Start comm times function ")
                self.comm_times()
                break

        # self.clean()

    def clean(self):
        """Clean up: close channel and connection safely"""
        try:
            if self.channel.is_open:
                self.channel.close()
            if self.connection.is_open:
                self.connection.close()
            print("Cleaned up RabbitMQ connection.")
        except Exception as e:
            print(f"Error during clean-up: {e}")