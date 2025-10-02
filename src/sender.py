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

MAX_SIZE_QUEUE = 16777216
INFINITY_TIME = 1000000


class MessageSender:
    def __init__(self, config):
        print("Start sender ... ")
        self.config = config
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config["rabbit"]["address"],
                credentials=pika.PlainCredentials(
                    config["rabbit"]["username"],
                    config["rabbit"]["password"]
                ),
                virtual_host=config["rabbit"]["virtual_host"]
            )
        )
        self.queue_device_1 = config["rabbit"]["queue_device_1"]
        self.queue_device_2 = config["rabbit"]["queue_device_2"]
        self.queue_device_3 = "host_queue"
        self.channel = self.connection.channel()
        self.queue = "time_layers"
        self.channel.queue_declare(queue=self.queue_device_1 , durable= True)
        self.channel.queue_declare(queue=self.queue_device_2 , durable= True)
        self.channel.queue_declare(queue=self.queue_device_3, durable=True)

        project_root = Path.cwd()
        cfg_path = project_root / "cfg" / "yolo11n.yaml"
        self.size_data = get_output_sizes(cfg_path)
        self.start_time = time.time()
        self.num_round = self.config["time_layer"]["num_round"]
        self.host_name = socket.gethostname()

    def send_message(self , messages_dict , queue_num = 2 ):
        queue_device = self.queue_device_2
        if queue_num == 1 :
            queue_device = self.queue_device_1
        elif queue_num == 3 :
            queue_device = self.queue_device_3
        self.channel.basic_publish(exchange='',
                              routing_key=queue_device,
                              body=pickle.dumps(messages_dict)
                              )
        # print(f"Sent {messages_dict["message"]}")

    def listening(self , queue_num = 1 ):
        device_queue = self.queue_device_1
        if queue_num == 2 :
            device_queue = self.queue_device_2
        method_frame, header_frame, body = self.channel.basic_get(queue=device_queue, auto_ack=True)
        if method_frame and body:
            data = pickle.loads(body)
            return data
        else :
            return None

    def comm_times(self):
        times = []

        for size in self.size_data :
            size_bytes = int(size * 1e6)
            if size_bytes >= MAX_SIZE_QUEUE:    # chubby size
                times.append(INFINITY_TIME)
                continue

            message = '1' * size_bytes
            avg_time = 0.0
            for _ in range(self.num_round):
                time_old = time.time_ns()
                self.send_message({
                    "signal" : "yes" ,
                    "message" : message,
                } , queue_num= 1)
                while True:
                    # method_frame, header_frame, body = self.channel.basic_get(queue=self.queue_device_1, auto_ack=True)
                    data = self.listening(queue_num=2)
                    if data is not None :
                        time_new = time.time_ns()
                        t = time_new - time_old
                        avg_time += t / 2
                        break
                    else:
                        continue
            avg_time = avg_time / self.num_round
            time_ms = avg_time / 1e6
            times.append(time_ms)
        self.send_message({
            "signal": "no",
            "message": 0,
        }, queue_num=1)

        return times



    def run(self):
        sent_request = True
        sent_comm_times = False
        while True :
            if sent_request :
                self.send_message({
                    "signal" : "REQUEST",
                    "message" : "sender"
                })

            data = self.listening()
            if data is not None :
                if data["signal"] == "START" :
                    layer_times_app = LayerProfiler(self.config)
                    res = layer_times_app.run()
                    print(len(res))
                    self.send_message({
                        "signal" : "time_layer " + str(self.host_name) ,
                        "message" : res
                    })
                print("Start comm times function ")
                times = self.comm_times()
                print(len(times))

                self.send_message({
                    "signal" : "comm_times",
                    "message" : times
                } , queue_num= 3)

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
