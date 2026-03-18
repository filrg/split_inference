import pika, pickle, yaml, time, os
import torch, cv2, threading
import numpy as np

from ultralytics.utils import ops
from queue import Queue, Empty
from src.tracker.tools import BoundingBox, Tools, write_partial, dict_data
from dataclasses import dataclass, field


# ---- Data Structures ----

@dataclass
class FPS:
    target: int = 25
    mean: int = 0


class Frame:
    total: int = 0
    showed: int = 0
    start: int = -1


# ---- Tracker Core ----

class Tracker:
    def __init__(self, config):

        # Split Inference Pipeline
        self.time_start_process = time.time()

        # ---- RabbitMQ Connection ----
        rabbit_config = config.get("rabbit", {})
        self.batch_size = config["server"]["batch-frame"]

        credentials = pika.PlainCredentials(
            rabbit_config.get("username"),
            rabbit_config.get("password")
        )

        params = pika.ConnectionParameters(
            host=rabbit_config.get("address"),
            virtual_host=rabbit_config.get("virtual-host"),
            credentials=credentials
        )

        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()

        print("[Tracker] Connected to RabbitMQ.")

        # ---- Queue Configuration ----
        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"

        self.bbox_buffer_queue = Queue()
        self.image_buffer_queue = Queue()

        self.bbox_buffer = {}
        self.image_buffer = {}

        # ---- Thread / State Control ----
        self.stop_event = threading.Event()
        self.image_stream_stopped = False
        self.bbox_stream_stopped = False

        # ---- FPS / Display ----
        self.fps = FPS()
        self.orig_img_size = (0, 0)

        self.dict_data = dict_data
        self.digits = 5

        self.prev_imshow = time.time()

        # display thread (lazy start)
        self.task_display = threading.Thread(target=self.display, daemon=True)
        self.task_display_started = False
        self.check_display = False

        # ---- Frame Tracking ----
        self.frame = Frame()
        self.cnt_img = 0
        self.cnt_bbox = 0

        # ---- Timing ----
        self.start_time = -1
        self.time_start_receive = -1
        self.time_start_display = 0

    # ---- Queue Setup ----

    def _declare_queues(self):
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)

    # ---- RabbitMQ Callbacks ----

    def _image_callback(self, ch, method, properties, body):
        """
        Receive original images from client 1
        Push to image buffer queue
        """

        try:
            message = pickle.loads(body)

            # STOP signal handling
            if isinstance(message, dict) and message.get('signal') == 'STOP':
                print("[Tracker] STOP signal received from image queue.")
                self.image_stream_stopped = True

                try:
                    self.dict_data["[1]totalTm"] = round(message.get('total_time', -1), self.digits)
                    self.dict_data["[1]outSze[T]"] = message.get('size_mess2tracker', -1)
                    self.dict_data["[1]outSze[2]"] = message.get('size_mess2cl2', -1)
                except Exception:
                    pass

                return

            # Split Inference Pipeline
            frames = message.get("ori_img")
            self.image_buffer_queue.put(frames)

            total_frames = message.get("total_frames", -1)
            if total_frames != -1:
                self.frame.total = total_frames

            self.orig_img_size = message.get("orig_img_size", self.orig_img_size)

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
        """
        Receive predictions from client 2
        """

        try:
            message = pickle.loads(body)

            # STOP signal handling
            if isinstance(message, dict) and message.get('signal') == 'STOP':
                print("[Tracker] STOP signal received from bbox queue.")
                self.bbox_stream_stopped = True

                try:
                    self.dict_data["[2]totalTm"] = round(message.get('total_time', -1), self.digits)
                    self.dict_data["[2]outSize"] = message.get('size_mess2tracker', -1)
                except Exception:
                    pass

                return

            # Split Inference Pipeline
            predictions = message.get("predictions")
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

    # ---- Listening Loop ----

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

        print("[Tracker] Listening for data...")
        self.start_time = time.time()

        try:
            while not (self.image_stream_stopped and self.bbox_stream_stopped):
                if self.stop_event.is_set():
                    break

                # process events with timeout to allow flag checks
                self.connection.process_data_events(time_limit=1)

        except KeyboardInterrupt:
            print("[Tracker] Interrupted by user.")
            self.stop_event.set()

        except Exception as e:
            print("[Tracker][start_listening] error:", e)
            self.stop_event.set()

        total_time = time.time() - self.start_time
        print(f"[Tracker][Time] total time: {total_time:.2f}s")
        print("[Tracker] All streams stopped.")

    # ---- Main Runner ----

    def run(self):
        self.start_time = time.time()

        try:
            self.start_listening()

            # wait for display thread if started
            if self.task_display_started:
                while self.task_display.is_alive() and not self.stop_event.is_set():
                    self.task_display.join(timeout=0.5)
            else:
                self.stop_event.set()

        except KeyboardInterrupt:
            print("[Tracker] Interrupted by user.")
            self.stop_event.set()

        except Exception as e:
            print("[Tracker][run] error:", e)
            self.stop_event.set()

        finally:
            self.cleanup()

    # ---- Cleanup ----

    def cleanup(self):
        try:
            self.data_for_csv()
            write_partial(self.dict_data)

            print(f"[Frame showed] {self.frame.showed}")
            print("[Tracker] Cleaning up...")

            self.channel.queue_delete(queue=self.bbox_queue)
            self.channel.queue_delete(queue=self.ori_img_queue)

            if self.connection and self.connection.is_open:
                self.connection.close()

        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            print("[Tracker] Connection closed.")

    # ---- Display Thread ----

    def display(self):
        print("[DISPLAY] started")

        while not self.stop_event.is_set():

            # exit condition
            if (
                self.image_stream_stopped and
                self.bbox_stream_stopped and
                self.image_buffer_queue.empty() and
                self.bbox_buffer_queue.empty()
            ):
                break

            try:
                origin_frame_test = self.image_buffer_queue.get(timeout=1)
                raw_prediction_tensor = self.bbox_buffer_queue.get(timeout=1)

            except Empty:
                continue

            except Exception as e:
                print("[Tracker][display] get error:", e)
                continue

            if self.frame.showed == 0:
                self.time_start_display = time.time()

            try:
                # Split Inference Pipeline
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
                    for result in results:

                        # render result
                        annotated_image = result.plot()
                        annotated_image = annotated_image[
                            :self.orig_img_size[0],
                            :self.orig_img_size[1]
                        ]

                        cv2.imshow("Visual Detection Output", annotated_image)

                        key = cv2.waitKey(int(1000 / max(1, self.fps.target))) & 0xFF
                        if key == ord('q'):
                            print("[Tracker] display stopped by user")
                            self.stop_event.set()
                            break

                        self.frame.showed += 1

                # FPS calculation
                if self.frame.total and self.frame.showed >= self.frame.total:
                    elapsed = time.time() - self.time_start_display if self.time_start_display > 0 else 1e-6
                    self.fps.mean = round(self.frame.total / elapsed, self.digits)
                    break

            except Exception as e:
                print("[Tracker][display] processing error:", e)

    # ---- Synchronization Logic ----

    def handle_data(self):

        # Split Inference Pipeline
        self.frame_received = min(self.cnt_img, self.cnt_bbox)

        if self.frame_received == self.batch_size and self.time_start_receive == -1:
            self.time_start_receive = time.time()
            print(f"[Time start receive] {self.time_start_receive}")

        # ---- FPS Profiling ----
        frame_profiler = 5 * self.batch_size

        if self.frame_received == frame_profiler and self.frame.start == -1:

            current_time = time.time()
            period_time = current_time - self.time_start_receive
            fps_real = frame_profiler / period_time

            print(f"[FPS real] {fps_real}")

            if int(fps_real) < int(self.fps.target):
                time_need = self.frame.total / fps_real
                time_target = self.frame.total / self.fps.target
                gap_time = time_need - time_target

                self.frame.start = int(gap_time * fps_real)
            else:
                self.frame.start = 6 * self.batch_size

            print(f"[Frame start] {self.frame.start}")

        # ---- Trigger Display Thread ----
        elif (
            self.frame.start != -1 and
            self.frame_received >= self.frame.start and
            not self.check_display
        ):
            print("Start display at frame", self.frame_received)

            self.check_display = True

            if not self.task_display_started:
                try:
                    self.task_display.start()
                    self.task_display_started = True
                except RuntimeError as e:
                    print("[Tracker][handle_data] thread error:", e)

    # ---- Logging ----

    def data_for_csv(self):
        print("data for csv")

        self.dict_data["Time"] = Tools().get_datatime()
        self.dict_data["[T]totalTM"] = round(time.time() - self.start_time, self.digits) if self.start_time > 0 else -1
        self.dict_data["[T]FPSR"] = self.fps.mean
        self.dict_data["[1]totalFr"] = self.frame.total