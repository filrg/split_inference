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

from queue import Queue, Empty
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
            "Time" : -1,
            "PointCut" : config["server"]["cut-layer"],
            "TmQoE" : -1 ,
            "[T]totalTM" : -1,
            "[T]FPSR" : -1,
            "[1]totalFr" : -1,
            "[1]totalTm" : -1,
            "[2]totalTm" : -1,
            "[1]outSze[T]" : -1,
            "[1]outSze[2]" : -1,
            "[2]outSize" : -1
        }
        self.digits = 5

        self.prev_imshow = time.time()
        # create thread but don't start immediately; make it daemon so it won't block exit
        self.task_display = threading.Thread(target=self.display, daemon=True)
        self.task_display_started = False
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
        self.cnt_img = 0
        self.cnt_bbox = 0

        # fps
        self.fps_mean = -1

        # time
        self.start_time =  -1
        self.time_start_receive = -1
        self.time_start_display = 0

    def _declare_queues(self):
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)

    def _image_callback(self, ch, method, properties, body):
        """ get image , frame and origin image size from client 1 .
        save image to queue ."""
        try:
            message = pickle.loads(body)
            # standardize STOP signal check
            if isinstance(message, dict) and message.get('signal') == 'STOP':
                print("[Tracker] STOP signal received from image queue.")
                self.image_stream_stopped = True
                # safe-get total_time if present
                if isinstance(message, dict):
                    try:
                        self.dict_data["[1]totalTm"] = round(message.get('total_time', -1), self.digits)
                        self.dict_data["[1]outSze[T]"] = message.get('size_mess2tracker', -1)
                        self.dict_data["[1]outSze[2]"] = message.get('size_mess2cl2', -1)
                    except Exception:
                        pass
                return

            frame = message.get("ori_img")
            total_frames = message.get("total_frames", -1)

            if total_frames != -1 :
                self.total_frames = total_frames
            self.orig_img_size = message.get("orig_img_size", self.orig_img_size)

            # just use only queue
            self.image_buffer_queue.put(frame)
            self.cnt_img += 1
            self.handle_data()

        except Exception as e:
            print("[Tracker][_image_callback] error:", e)
        finally:
            try:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                pass

    def _bbox_callback(self, ch, method, properties, body):
        """ get result and frame index from client 2 """
        try:
            message = pickle.loads(body)
            # expect dict with 'signal' or regular dict
            if isinstance(message, dict) and message.get('signal') == 'STOP':
                print("[Tracker] STOP signal received from bbox queue.")
                self.bbox_stream_stopped = True
                if isinstance(message, dict):
                    try:
                        self.dict_data["[2]totalTm"] = round(message.get('total_time', -1), self.digits)
                        self.dict_data["[2]outSize"] = message.get('size_mess2tracker', -1)
                    except Exception:
                        pass
                return

            predictions = message.get("predictions")
            # handle by just only queue
            self.bbox_buffer_queue.put(predictions)
            self.cnt_bbox += 1
            self.handle_data()
        except Exception as e:
            print("[Tracker][_bbox_callback] error:", e)
        finally:
            try:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                pass

    def start_listening(self):
        self._declare_queues()
        self.channel.basic_consume(queue=self.ori_img_queue, on_message_callback=self._image_callback, auto_ack=False)
        self.channel.basic_consume(queue=self.bbox_queue, on_message_callback=self._bbox_callback, auto_ack=False)

        print("[Tracker] Listening for confirmation... Press Ctrl+C to exit.")
        self.start_time = time.time()

        try:
            while not (self.image_stream_stopped and self.bbox_stream_stopped):
                if self.stop_event.is_set():
                    break
                # process incoming pika events; small timeout so we can check flags regularly
                self.connection.process_data_events(time_limit=1)
        except KeyboardInterrupt:
            print("\n[Tracker] Interrupted by user (start_listening).")
            self.stop_event.set()
        except Exception as e:
            print("[Tracker][start_listening] error:", e)
            self.stop_event.set()

        total_time = time.time() - self.start_time
        print(f"[Tracker][Time] total time: {total_time:.2f}s")
        print("\n[Tracker] All streams stopped. Loop finished.")

    def run(self):
        self.start_time = time.time()
        try:
            self.start_listening()
            # join only if the display thread was actually started
            if self.task_display_started:
                # join with timeout loop so main thread can still respond to stop_event
                while self.task_display.is_alive() and not self.stop_event.is_set():
                    self.task_display.join(timeout=0.5)
            else:
                # nothing to join; ensure stop_event set to allow cleanup
                self.stop_event.set()
        except KeyboardInterrupt:
            print("\n[Tracker] Interrupted by user.")
            self.stop_event.set()
        except Exception as e:
            print("[Tracker][run] error:", e)
            self.stop_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            self.data_for_csv()
            write_partial(self.dict_data)
            print(f"[Frame showed ] {self.frame_showed}")
            print("[Tracker] Cleaning up...")
            if self.connection and self.connection.is_open:
                self.connection.close()
        except Exception as e:
            print("[Tracker][cleanup] error:", e)
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("[Tracker] Connection closed / windows destroyed.")

    def display(self):
        print("Run display function")
        # Loop until everything signalled stopped and queues empty OR stop_event set
        while not self.stop_event.is_set():
            # break condition: both streams stopped and both queues empty and we've shown all frames
            if self.image_stream_stopped and self.bbox_stream_stopped and self.image_buffer_queue.empty() and self.bbox_buffer_queue.empty():
                break

            try:
                origin_frame_test = self.image_buffer_queue.get(timeout=1)  # wait up to 1s
                raw_prediction_tensor = self.bbox_buffer_queue.get(timeout=1)
            except Empty:
                # no data available currently, loop back and check stop flags
                continue
            except Exception as e:
                print("[Tracker][display] get error:", e)
                continue

            if self.frame_showed == 0 :
                self.time_start_display = time.time()

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
                    h_max = min(self.orig_img_size[0], annotated_image.shape[0])
                    w_max = min(self.orig_img_size[1], annotated_image.shape[1])
                    annotated_image = annotated_image[0:h_max, 0:w_max]

                    # get fps reality
                    now = time.time()
                    delta = now - self.prev_imshow if (now - self.prev_imshow) != 0 else 1e-6
                    fps_a = int(1 / delta)
                    self.prev_imshow = now
                    self.fps_display.append(fps_a)
                    cv2.putText(annotated_image, f"FPS: {fps_a}",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA)
                    cv2.imshow("Visual Detection Output", annotated_image)
                    # use waitKey small; pressing q can be used to stop
                    key = cv2.waitKey(int(1000 / max(1, self.fps))) & 0xFF
                    if key == ord('q'):
                        print("[Tracker] 'q' pressed, stopping display.")
                        self.stop_event.set()
                        break
                    self.frame_showed += 1

                if self.total_frames and self.frame_showed >= self.total_frames:
                    # compute fps_mean and stop display
                    elapsed = time.time() - self.time_start_display if self.time_start_display > 0 else 1e-6
                    self.fps_mean = round(self.total_frames / elapsed, self.digits)
                    break

            except Exception as e:
                print("[Tracker][display] processing error:", e)
                continue

    def visual_time(self , data , title):
        plt.plot(data, marker='o', linestyle='-', label="My Data")
        plt.title(title)
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def handle_data(self):
        # get number of frame for 1s , 2s and 3s
        # use queue sizes safely
        self.frame_received = min(self.cnt_img , self.cnt_bbox)
        if self.frame_received == 1:
            self.time_start_receive = time.time()

        if (self.frame_start != 0 and self.frame_received >= self.frame_start) and not self.check_display:
            print("Start show output at frame ", self.frame_received)
            self.check_display = True
            # start display thread only once
            if not self.task_display_started:
                try:
                    self.task_display.start()
                    self.task_display_started = True
                except RuntimeError as e:
                    print("[Tracker][handle_data] failed to start display thread:", e)

        now = time.time()
        if self.time_start_receive != -1 :
            if self.frame_for_1st == -1 and (now - self.time_start_receive) >= 1:
                print(f"[Frame for 1st ] {self.frame_received}")
                self.frame_for_1st = self.frame_received
            if self.frame_for_2nd == -1 and (now - self.time_start_receive) >= 2:
                print(f"[Frame for 2nd ] {self.frame_received}")
                self.frame_for_2nd = self.frame_received
            if self.frame_for_3rd == -1 and (now - self.time_start_receive) >= 3:
                print(f"[Frame for 3rd ] {self.frame_received}")
                self.frame_for_3rd = self.frame_received
                fps_reallity = (self.frame_for_3rd - self.frame_for_1st)//2 if self.frame_for_1st != -1 else 0
                # avoid division by zero
                try:
                    self.frame_start = int((self.fps - fps_reallity) * self.total_frames // self.fps) if self.fps else 0
                except Exception:
                    self.frame_start = 0
                print(f"[Frame start] {self.frame_start}")

    def get_datatime(self , millisecond = False):
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        second = now.second
        millisecond = now.microsecond // 1000  # convert microseconds → milliseconds

        if millisecond :
            return f"{hour}:{minute}:{second}.{millisecond}"
        else :
            return f"{hour}:{minute}:{second}"

    def data_for_csv(self):
        print("data for csv")
        self.dict_data["Time"] = self.get_datatime()
        self.dict_data["[T]totalTM"] = round(time.time() - self.start_time , self.digits) if self.start_time>0 else -1
        self.dict_data["[T]FPSR"] = self.fps_mean
        self.dict_data["[1]totalFr"] = self.total_frames
