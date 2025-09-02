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
        self.start_receive_bounding_box = 0
        self.start_receive_origin_image = 0

        self.orig_img_size = (0 , 0)

        self.dict_data = {
            "[T]totalTm" : 0
            ,"[T]totalFr" : 0
            ,"[T]TmRecv" : 0
            ,"[T]FRPS" : 0
            ,"[T]Fr/~1s" : -1
            ,"[T]Fr/~2s" : -1
            ,"[T]Fr/~3s" : -1
        }

        self.time_start_received = time.time()
        print("Time start received :")
        self.show_datetime()
        self.total_frames = 0
        self.digits = 5

        self.prev_frame = time.time()
        self.prev_imshow = time.time()
        self.lst_time = [0]
        self.lst_delay = []

        self.num_testbed = 2
        self.frame_received = 0
        self.frame_start = 0
        self.task_display = threading.Thread(target=self.display)
        self.check_display = False

        self.fps_display = []
        self.frame_showed = 0



    def _declare_queues(self):
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)

    def _image_callback(self, ch, method, properties, body):
        try:
            message = pickle.loads(body)
            if message == 'STOP':
                print("[Tracker] STOP signal received from image queue.")
                # print(f"[Origin Image][Time] {time.time() - self.start_receive_origin_image}")
                self.image_stream_stopped = True
                return

            frame_index = message.get("frame_index")
            frame = message.get("ori_img")
            total_frames = message.get("total_frames")

            # print(f"[Frame index] from client 1 : {frame_index}")

            if total_frames != -1 :
                self.dict_data["[T]totalFr"] = total_frames
                self.total_frames = total_frames
                if self.frame_start == -1 :
                    self.frame_start = total_frames
            self.orig_img_size = message.get("orig_img_size")

            if frame_index == 0:
                self.start_receive_origin_image = time.time()

            # print(f"--- [Received Image] Frame Index: {frame_index} ---")
            self.image_buffer[frame_index] = frame

            # just use only queue
            self.image_buffer_queue.put(frame)
            self.handle_data()

            # if frame_index in self.bbox_buffer:
            #     self._process_pair(frame_index)
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _bbox_callback(self, ch, method, properties, body):
        try:
            message = pickle.loads(body)
            if message == 'STOP':
                print("[Tracker] STOP signal received from bbox queue.")
                self.bbox_stream_stopped = True
                return

            frame_index = message.get("frame_index")
            predictions = message.get("predictions")

            # print(f"[Frame index] from client 2 : {frame_index}")

            if frame_index == 0:
                self.start_receive_bounding_box = time.time()

            # print(f"--- [Received BBox] Frame Index: {frame_index} ---")
            # print(f"[Predictions] check type {type(predictions)}")
            self.bbox_buffer[frame_index] = predictions

            # handle by just only queue
            self.bbox_buffer_queue.put(predictions)
            self.handle_data()

            # if frame_index in self.image_buffer:
            #     self._process_pair(frame_index)

        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self._declare_queues()
        self.channel.basic_consume(queue=self.ori_img_queue, on_message_callback=self._image_callback, auto_ack=False)
        self.channel.basic_consume(queue=self.bbox_queue, on_message_callback=self._bbox_callback, auto_ack=False)

        print("[Tracker] Listening for confirmation... Press Ctrl+C to exit.")
        start_time = time.time()
        self.prev_frame = time.time()

        while not (self.image_stream_stopped and self.bbox_stream_stopped):
            if self.stop_event.is_set():
                break
            self.connection.process_data_events(time_limit=1)

        total_time = time.time() - start_time
        print(f"[Tracker][Time] total time: {total_time:.2f}s")
        self.dict_data["[T]totalTm"] = round(total_time , self.digits)

        print("\n[Tracker] All streams stopped. Loop finished.")

    def run(self):
        try:
            self.start_listening()
            self.task_display.join()
        except KeyboardInterrupt:
            print("\n[Tracker] Interrupted by user.")
            self.stop_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        # self.visual_time(self.lst_time , "Received")
        # self.visual_time(self.lst_time, "Imshow")
        self.dict_data["[T]TmRecv"] = round(time.time() - self.time_start_received , self.digits)
        self.dict_data["[T]FRPS"] = round(sum(self.fps_display) / len(self.fps_display) , self.digits)
        write_partial(self.dict_data)

        print(f"[Frame showed ] {self.frame_showed}")

        print("[Tracker] Cleaning up...")
        if self.connection and self.connection.is_open:
            self.connection.close()
            cv2.destroyAllWindows()
            print("[Tracker] Connection closed.")

    def _process_pair(self, frame_index):
        if self.frame_received == self.frame_start - 1 :
            print(f"[Frame received] : {self.frame_received}")
            print(f"[Frame start ] : {self.frame_start}")
            self.dict_data["[T]totalFr"] = self.total_frames

            display = False
            if display:
                # t_b = threading.Thread(target=self.display)
                # t_b.start()
                t_display = threading.Thread(target=self.display, daemon=True)
                t_display.start()
                self.display()
        else :
            self.frame_received += 1

    def display(self):
        while self.image_buffer_queue.qsize() != 0  and self.bbox_buffer_queue.qsize() != 0:
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
            self.task_display.start()
            self.check_display = True

        now = time.time()
        # if self.dict_data["[T]Fr/~1s"] == -1 and self.frame_received == 10:
        #     self.show_datetime()
        #     temp_time = round(now - self.time_start_received , self.digits)
        #     print(f"[Frame for 1st ] {temp_time}")
        #     self.dict_data["[T]Fr/~1s"] = temp_time
        # if self.dict_data["[T]Fr/~2s"] == -1 and self.frame_received == 20:
        #     self.show_datetime()
        #     temp_time = round(now - self.time_start_received, self.digits)
        #     print(f"[Frame for 2nd ] {temp_time}")
        #     self.dict_data["[T]Fr/~2s"] = temp_time
        # if self.dict_data["[T]Fr/~3s"] == -1 and self.frame_received == 30:
        #     self.show_datetime()
        #     temp_time = round(now - self.time_start_received, self.digits)
        #     print(f"[Frame for 3rd ] {temp_time}")
        #     self.dict_data["[T]Fr/~3s"] = temp_time
        #     fps_reallity = 20 // (self.dict_data["[T]Fr/~3s"] - self.dict_data["[T]Fr/~1s"])
        #     self.frame_start = int((self.fps - fps_reallity) * self.total_frames // self.fps)
        #     print(f"[Frame start] {self.frame_start}")
        if self.dict_data["[T]Fr/~1s"] == -1 and (now - self.time_start_received) >= 1:
            self.show_datetime()
            print(f"[Frame for 1st ] {self.frame_received}")
            self.dict_data["[T]Fr/~1s"] = self.frame_received
        if self.dict_data["[T]Fr/~2s"] == -1 and (now - self.time_start_received) >= 2:
            self.show_datetime()
            print(f"[Frame for 2nd ] {self.frame_received}")
            self.dict_data["[T]Fr/~2s"] = self.frame_received
        if self.dict_data["[T]Fr/~3s"] == -1 and (now - self.time_start_received) >= 3:
            self.show_datetime()
            print(f"[Frame for 3rd ] {self.frame_received}")
            self.dict_data["[T]Fr/~3s"] = self.frame_received
            fps_reallity = (self.dict_data["[T]Fr/~3s"] - self.dict_data["[T]Fr/~1s"])//2
            self.frame_start = int((self.fps - fps_reallity) * self.total_frames // self.fps)
            print(f"[Frame start] {self.frame_start}")

    def show_datetime(self):
        now = datetime.now()

        # Extract hour, minute, second, millisecond
        hour = now.hour
        minute = now.minute
        second = now.second
        millisecond = now.microsecond // 1000  # convert microseconds → milliseconds

        print(f"{hour}:{minute}:{second}.{millisecond}")