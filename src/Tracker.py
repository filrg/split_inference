import pika
import pickle
import yaml
import torch
import numpy as np
import threading
import time
import cv2
import os

from ultralytics.utils import ops
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from src.Utils import write_partial

from queue import Queue
from src.Model import BoundingBox
from datetime import datetime

class Tracker:
    def __init__(self, config):
        self.time_start_process = time.time()
        rabbit_config = config.get("rabbit", {})
        credentials = pika.PlainCredentials(rabbit_config.get("username"), rabbit_config.get("password"))
        params = pika.ConnectionParameters(
            host=rabbit_config.get("address"),
            virtual_host=rabbit_config.get("virtual-host"),
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        print("[Tracker] Connected to RabbitMQ.")

        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"

        self.bbox_buffer_queue = Queue()
        self.image_buffer_queue = Queue()

        self.bbox_buffer = {}
        self.image_buffer = {}

        self.stop_event = threading.Event()
        self.image_stream_stopped = False
        self.bbox_stream_stopped = False

        self.fps = 30
        self.orig_img_size = (0 , 0)

        self.dict_data = {
            "Time" : -1
            ,"PointCut" : config["server"]["cut-layer"]
            ,"[T]totalTM" : -1
            ,"[T]FPSR" : -1
            ,"[1]totalFr" : -1
            ,"[1]totalTm" : -1
            ,"[2]totalTm" : -1
            ,"[1]outSize" : -1
            ,"[2]outSize" : -1
        }

        print("Time start received :")
        print(self.get_datatime(millisecond=True))
        self.digits = 5

        self.prev_frame = time.time()
        self.prev_imshow = time.time()
        self.task_display = threading.Thread(target=self.display)
        self.check_display = False

        self.fps_display = []

        # frame
        self.total_frames = 0
        self.frame_showed = 0
        self.frame_received = 0
        self.frame_start = 0
        self.frame_for_1st = -1
        self.frame_for_2nd = -1
        self.frame_for_3rd = -1

        # fps
        self.fps_mean = -1

        # time
        self.start_time =  -1
        self.time_start_received = time.time()
        self.time_start_display = 0

    def _declare_queues(self):
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)

    def _image_callback(self, ch, method, properties, body):
        """ get image , frame and origin image size from client 1 .
        save image to queue ."""
        try:
            message = pickle.loads(body)
            if message == 'STOP':
                print("[Tracker] STOP signal received from image queue.")
                # print(f"[Origin Image][Time] {time.time() - self.start_receive_origin_image}")
                self.image_stream_stopped = True
                return

            # frame_index = message.get("frame_index")
            frame = message.get("ori_img")
            total_frames = message.get("total_frames")

            if total_frames != -1 :
                self.total_frames = total_frames
            self.orig_img_size = message.get("orig_img_size")

            # just use only queue
            self.image_buffer_queue.put(frame)
            self.handle_data()

        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _bbox_callback(self, ch, method, properties, body):
        """ get result and frame index from client 2 """
        try:
            message = pickle.loads(body)
            if message == 'STOP':
                print("[Tracker] STOP signal received from bbox queue.")
                self.bbox_stream_stopped = True
                return

            # frame_index = message.get("frame_index")
            predictions = message.get("predictions")

            # handle by just only queue
            self.bbox_buffer_queue.put(predictions)
            self.handle_data()
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self._declare_queues()
        self.channel.basic_consume(queue=self.ori_img_queue, on_message_callback=self._image_callback, auto_ack=False)
        self.channel.basic_consume(queue=self.bbox_queue, on_message_callback=self._bbox_callback, auto_ack=False)

        print("[Tracker] Listening for confirmation... Press Ctrl+C to exit.")
        self.start_time = time.time()
        self.prev_frame = time.time()

        while not (self.image_stream_stopped and self.bbox_stream_stopped):
            if self.stop_event.is_set():
                break
            self.connection.process_data_events(time_limit=1)

        total_time = time.time() - self.start_time
        print(f"[Tracker][Time] total time: {total_time:.2f}s")
        print("\n[Tracker] All streams stopped. Loop finished.")

    def run(self):
        self.start_time = time.time()
        try:
            self.start_listening()
            self.task_display.join()
        except KeyboardInterrupt:
            print("\n[Tracker] Interrupted by user.")
            self.stop_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        self.data_for_csv()
        write_partial(self.dict_data)
        print(f"[Frame showed ] {self.frame_showed}")
        print("[Tracker] Cleaning up...")
        if self.connection and self.connection.is_open:
            self.connection.close()
            cv2.destroyAllWindows()
            print("[Tracker] Connection closed.")

    def display(self):
        while (self.image_buffer_queue.qsize() != 0  and self.bbox_buffer_queue.qsize() != 0) or self.frame_showed < self.total_frames:
            if self.frame_showed == 0 :
                self.time_start_display = time.time()
            if self.frame_showed == self.total_frames - 1:
                print("Estimating fps mean !")
                self.fps_mean = round(self.total_frames/(time.time() - self.time_start_display),self.digits)
            predictor = BoundingBox()
            origin_frame_test = self.image_buffer_queue.get()
            raw_prediction_tensor = self.bbox_buffer_queue.get()

            origin_frame_shape = origin_frame_test.shape
            orig_imgs_list = [origin_frame_test]

            results = predictor.postprocess(
                preds=raw_prediction_tensor,
                resized_shape=(640, 640),
                orig_shape=origin_frame_shape[:2],
                orig_imgs=orig_imgs_list
            )
            if results:
                final_result = results[0]
                annotated_image = final_result.plot()
                annotated_image = annotated_image[0:self.orig_img_size[0], 0: self.orig_img_size[1]]

                # get fps reality
                now = time.time()
                if now - self.prev_imshow != 0:
                    fps_a = int(1 / (now - self.prev_imshow))
                else:
                    print("[FPS] [problem] at frame ", frame_index)
                    fps_a = 10000
                self.prev_imshow = now
                self.fps_display.append(fps_a)
                cv2.putText(annotated_image, f"FPS: {fps_a}",
                            (20, 40),  # Position (x, y)
                            cv2.FONT_HERSHEY_SIMPLEX,  # Font
                            1,  # Font scale
                            (0, 255, 0),  # Color (BGR) - Green
                            2,  # Thickness
                            cv2.LINE_AA)  # Line type
                self.prev_imshow = time.time()
                cv2.imshow("Visual Detection Output", annotated_image)
                cv2.waitKey(int(1000 / self.fps))
                self.frame_showed += 1

    def visual_time(self , data , title):

        plt.plot(data, marker='o', linestyle='-', color='b', label="My Data")

        plt.title(title)
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        plt.show()

    def handle_data(self):
        # get number of frame for 1s , 2s and 3s
        self.frame_received = min(self.bbox_buffer_queue.qsize(), self.image_buffer_queue.qsize())
        if self.frame_received == 1:
            self.time_start_received = time.time()

        if (self.frame_start != 0 and self.frame_received == self.frame_start) and self.check_display == False:
            print("Start show output at frame ", self.frame_received)
            self.check_display = True
            self.task_display.start()

        now = time.time()

        if self.frame_for_1st == -1 and (now - self.time_start_received) >= 1:
            print(f"[Frame for 1st ] {self.frame_received}")
            self.frame_for_1st = self.frame_received
        if self.frame_for_2nd == -1 and (now - self.time_start_received) >= 2:
            print(f"[Frame for 2nd ] {self.frame_received}")
            self.frame_for_2nd = self.frame_received
        if self.frame_for_3rd == -1 and (now - self.time_start_received) >= 3:
            print(f"[Frame for 3rd ] {self.frame_received}")
            self.frame_for_3rd = self.frame_received
            fps_reallity = (self.frame_for_3rd - self.frame_for_1st)//2
            self.frame_start = int((self.fps - fps_reallity) * self.total_frames // self.fps)
            print(f"[Frame start] {self.frame_start}")

    def get_datatime(self , millisecond = False):
        now = datetime.now()

        # Extract hour, minute, second, millisecond
        hour = now.hour
        minute = now.minute
        second = now.second
        millisecond = now.microsecond // 1000  # convert microseconds → milliseconds

        if millisecond :
            return f"{hour}:{minute}:{second}.{millisecond}"
        else :
            return f"{hour}:{minute}:{second}"

    def data_for_csv(self):
        self.dict_data["Time"] = self.get_datatime(millisecond=False)
        # self.dict_data["PointCut"]

        self.dict_data["[T]totalTM"] = round(time.time() - self.start_time , self.digits)
        self.dict_data["[T]FPSR"] = self.fps_mean

        self.dict_data["[1]totalFr"] = self.total_frames
        self.dict_data["[1]totalTm"] = -1
        self.dict_data["[1]outSize"] = -1

        self.dict_data["[2]totalTm"] = -1
        self.dict_data["[2]outSize"] = -1
