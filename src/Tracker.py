import yaml
import torch
import numpy as np
import os

from ultralytics.utils import ops
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from src.Utils import write_partial

from queue import Empty
from src.Model import BoundingBox
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process, shared_memory , Queue ,Manager
import time, pickle, cv2, pika
from multiprocessing import Event

import pika
import pickle
import time
import cv2

class Tracker:
    def __init__(self, config, img_queue, bbox_queue, shared_dict , dict_data):
        self.start_time_programming = time.time()
        self.rabbit_config = config.get("rabbit", {})
        self.image_buffer_queue = img_queue
        self.bbox_buffer_queue = bbox_queue
        self.shared_dict = shared_dict
        self.dict_data = dict_data

        self.dict_data["PointCut"] = config["server"]["cut-layer"]

        self.connection = None
        self.channel = None

        self.ori_img_queue = "ori_img_queue"
        self.bbox_queue = "bbox_queue"

        self.image_stream_stopped = False
        self.bbox_stream_stopped = False

        # frame
        self.total_frame = -1
        self.cnt_img = 0
        self.cnt_bbox = 0
        self.frame_start = -1

        # fps
        self.fps = 30

        # time
        self.time_start_receive = -1
        shared_dict["start_time_programming"] = self.start_time_programming
        self.dict_data["Time"] = datetime.now().strftime("%H:%M:%S")

        # digits
        self.digits = 4

    def _connect(self):
        """ Kết nối tới RabbitMQ """
        credentials = pika.PlainCredentials(
            self.rabbit_config.get("username"),
            self.rabbit_config.get("password")
        )
        params = pika.ConnectionParameters(
            host=self.rabbit_config.get("address"),
            virtual_host=self.rabbit_config.get("virtual-host"),
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        print("[Tracker] Connected to RabbitMQ.")

    def _declare_queues(self):
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)

    def _image_callback(self, ch, method, properties, body):
        try:
            message = pickle.loads(body)
            if isinstance(message, dict) and message.get("signal") == "STOP":
                print("[Tracker] STOP signal received from image queue.")
                self.image_stream_stopped = True
                self.image_buffer_queue.put(None)
                self.dict_data["[1]totalTm"] = round(message.get("total_time"), self.digits)
                self.dict_data["[1]outSze[T]"] = message.get("size_mess2tracker")
                self.dict_data["[1]outSze[2]"] = message.get("size_mess2cl2")
                return

            # get origin image
            frame = message.get("ori_img")
            if frame is not None:
                self.image_buffer_queue.put(frame)

            # get origin image size
            origin_image_size = message.get("orig_img_size")
            if origin_image_size is not None :
                self.shared_dict["origin_img_size"] = origin_image_size

            # get total frame
            if message.get("total_frames") is not None and self.total_frame == -1 :
                self.total_frame = message.get("total_frames")
                self.dict_data["[1]totalFr"] = message.get("total_frames")

            self.cnt_img += 1
            self.handle()
        except Exception as e:
            print("[Tracker][_image_callback] error:", e)
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _bbox_callback(self, ch, method, properties, body):
        try:
            message = pickle.loads(body)
            if isinstance(message, dict) and message.get("signal") == "STOP":
                print("[Tracker] STOP signal received from bbox queue.")
                self.bbox_stream_stopped = True
                self.bbox_buffer_queue.put(None)
                self.dict_data["[2]totalTm"] = round(message.get("total_time") , self.digits)
                self.dict_data["[2]outSize"] = message.get("size_mess2tracker")
                return

            preds = message.get("predictions")
            if preds is not None:
                self.bbox_buffer_queue.put(preds)
                # print("[Tracker][bbox] received successfully!")
            self.cnt_bbox += 1
            self.handle()
        except Exception as e:
            print("[Tracker][_bbox_callback] error:", e)
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def handle(self):
        frame_received = min(self.cnt_img , self.cnt_bbox)
        if frame_received == 1 :
            self.time_start_receive = time.time()
        else :
            now = time.time()
            period = 0
            if self.time_start_receive != -1 :
                period = now - self.time_start_receive
            if period >= 1 and self.frame_start == -1 :
                fps_real = round(frame_received / period , self.digits)
                self.dict_data["[T]FPSR"] = round(fps_real, self.digits)
                self.frame_start = ((self.fps - fps_real) * self.total_frame) // self.fps
                print("[Frame start] " , self.frame_start)
            if frame_received >= self.frame_start:
                self.shared_dict["start_signal"] = True

    def start_listening(self):
        self._declare_queues()
        self.channel.basic_consume(
            queue=self.ori_img_queue,
            on_message_callback=self._image_callback,
            auto_ack=False
        )
        self.channel.basic_consume(
            queue=self.bbox_queue,
            on_message_callback=self._bbox_callback,
            auto_ack=False
        )

        print("[Tracker] Listening for messages...")
        start_time = time.time()
        try:
            while not (self.image_stream_stopped and self.bbox_stream_stopped):
                self.connection.process_data_events(time_limit=1)
        except KeyboardInterrupt:
            print("[Tracker] Interrupted by user.")
        finally:
            total_time = time.time() - start_time
            print(f"[Tracker] Total running time: {total_time:.2f}s")
            self.cleanup()

    def run(self):
        try:
            self._connect()
            self.start_listening()
        except Exception as e:
            print("[Tracker][run] error:", e)
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            print("[Tracker] Cleaning up...")
            if self.connection and self.connection.is_open:
                self.connection.close()
        except Exception as e:
            print("[Tracker][cleanup] error:", e)
        finally:
            try:
                cv2.destroyAllWindows()
            except:
                pass
            print("[Tracker] Connection closed / windows destroyed.")


class Display:
    def __init__(self, img_queue, bbox_queue , share_dict , dict_data):
        self.img_queue = img_queue
        self.bbox_queue = bbox_queue
        self.shared_dict = share_dict
        self.dict_data = dict_data
        # fps
        self.fps = 30
        # frame
        self.cnt_frame = 0


    def run(self):
        print("[Display] Start displaying...")
        while True:
            frame = self.img_queue.get()
            if frame is None:  # sentinel stop
                break
            bbox = self.bbox_queue.get()
            if bbox is None:
                break
            origin_img_size = self.shared_dict["origin_img_size"]
            if origin_img_size is None :
                break

            if self.shared_dict.get("start_signal", False):
                keep_running = self.display(frame, bbox)
                if keep_running is False:
                    break

        print("[Display] Exit cleanly.")
        print("[Count frame]" , self.cnt_frame)
        total_time = time.time() - self.shared_dict["start_time_programming"]
        self.dict_data["[T]totalTM"] = round(total_time , 5)
        write_partial(self.dict_data)
        cv2.destroyAllWindows()

    def display(self , origin_frame_test , raw_prediction_tensor):
        try:
            origin_frame_shape = origin_frame_test.shape
            orig_imgs_list = [origin_frame_test]

            predictor = BoundingBox()
            results = predictor.postprocess(
                preds=raw_prediction_tensor,
                resized_shape=(640, 640),
                orig_shape=origin_frame_shape[:2],
                orig_imgs=orig_imgs_list
            )
            if results:
                final_result = results[0]
                annotated_image = final_result.plot()
                # ensure slicing within image bounds
                h_max = min(self.shared_dict["origin_img_size"][0], annotated_image.shape[0])
                w_max = min(self.shared_dict["origin_img_size"][1], annotated_image.shape[1])
                annotated_image = annotated_image[0:h_max, 0:w_max]

                cv2.imshow("Visual Detection Output", annotated_image)
                # use waitKey small; pressing q can be used to stop
                key = cv2.waitKey(int(1000 / max(1, self.fps))) & 0xFF
                if key == ord('q'):
                    print("[Tracker] 'q' pressed, stopping display.")
                self.cnt_frame += 1
        except Exception as e:
            print("[Tracker][display] processing error:", e)
