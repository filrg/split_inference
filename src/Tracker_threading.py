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
from src.Utils import write_partial , dict_data

from queue import Queue, Empty
from src.Model import BoundingBox
from datetime import datetime

class Tracker:
    def __init__(self, config):
        self.time_start_process = time.time()
        rabbit_config = config.get("rabbit", {})
        self.batch_size = config["server"]["batch-frame"]
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

        self.fps = 28
        self.orig_img_size = (0 , 0)

        self.dict_data = dict_data
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
        self.num_frame_received = 0
        self.frame_start = -1
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
                        self.dict_data["[1]GPU_time"] = message.get('GPU_time')
                        self.dict_data["[1]peak_RAM"] = message.get('peak_RAM')
                        self.dict_data["[1]peak_VRAM"] = message.get('peak_VRAM')
                        print(gpu_time)
                    except Exception:
                        pass
                return

            frames = message.get("ori_img")
            self.image_buffer_queue.put(frames)
            # print(f"[Img][type]{type(frames)}")
            # print(f"Frame received : {len(frames)}")
            # print("Get images done !")

            total_frames = message.get("total_frames", -1)

            if total_frames != -1 :
                self.total_frames = total_frames
            self.orig_img_size = message.get("orig_img_size", self.orig_img_size)

            # just use only queue
            self.cnt_img += self.batch_size
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
                        self.dict_data["[2]GPU_time"] = message.get('GPU_time')
                        self.dict_data["[2]peak_RAM"] = message.get('peak_RAM')
                        self.dict_data["[2]peak_VRAM"] = message.get('peak_VRAM')
                    except Exception:
                        pass
                return

            predictions = message.get("predictions")

            # handle by just only queue
            self.bbox_buffer_queue.put(predictions)

            self.cnt_bbox += self.batch_size
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
            self.channel.queue_delete(queue= self.bbox_queue )
            self.channel.queue_delete(queue=self.ori_img_queue)
            if self.connection and self.connection.is_open:
                self.connection.close()
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("[Tracker] Connection closed / windows destroyed.")

    def display(self):
        print("   --- [DISPLAY] ---")
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
                origin_frame_shape = origin_frame_test[0].shape
                orig_imgs_list = origin_frame_test

                predictor = BoundingBox(overrides={"imgsz": 640})
                results = predictor.postprocess(
                    preds=raw_prediction_tensor,
                    img_shape=(640, 640),
                    orig_shape=origin_frame_shape[:2],
                    orig_imgs=orig_imgs_list
                )

                if results:
                    for idx , result in enumerate(results):
                        annotated_image = result.plot()
                        annotated_image = annotated_image[:self.orig_img_size[0] , : self.orig_img_size[1]]
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
        # use queue sizes safely
        self.frame_received = min(self.cnt_img , self.cnt_bbox)
        if self.frame_received == self.batch_size and self.time_start_receive == -1 :
            self.time_start_receive = time.time()
            print(f"[Time start receive ] {self.time_start_receive}")

        frame_profiler = 5 * self.batch_size
        if self.frame_received == frame_profiler and self.frame_start == -1 :
            current_time = time.time()
            period_time = current_time - self.time_start_receive
            fps_real = frame_profiler / period_time
            print(f"[FPS real ] {fps_real}")
            if int(fps_real) < int(self.fps):
                time_need = self.total_frames / fps_real
                time_target = self.total_frames / self.fps
                gap_time = time_need - time_target
                # print(f"[Gap time] {gap_time}")
                self.frame_start = int(gap_time * fps_real )
                # formula : frame_start : total * ( 1 - fps_r / fps  )
            else :
                self.frame_start = 6 * self.batch_size
            print(f"[Frame start] {self.frame_start}")

        elif (self.frame_start != -1 and self.frame_received >= self.frame_start) and not self.check_display:
            print("Start show output at frame ", self.frame_received)
            self.check_display = True
            # start display thread only once
            if not self.task_display_started:
                try:
                    self.task_display.start()
                    self.task_display_started = True
                except RuntimeError as e:
                    print("[Tracker][handle_data] failed to start display thread:", e)

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


