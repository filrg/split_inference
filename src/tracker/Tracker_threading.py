import pika, pickle, yaml, time, os
import torch, cv2, threading
import numpy as np

from ultralytics.utils import ops
from queue import Queue, Empty
from src.tracker.tools import BoundingBox, Tools, write_partial, dict_data
from dataclasses import dataclass, field

# from evaluation.mAP.visual import Visual
from src.tracker.visual import Visual_tracker

# ---- Data Structures ----

@dataclass
class FPS:
    target: int = 25
    mean: int = 0


class Frame:
    total: int = 0
    showed: int = 0
    start: int = -1

class Tracker:
    def __init__(self, config):

        self.time_start_process = time.time()

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

        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"

        self.bbox_buffer_queue = Queue()
        self.image_buffer_queue = Queue()

        self.bbox_buffer = {}
        self.image_buffer = {}

        self.stop_event = threading.Event()
        self.image_stream_stopped = False
        self.bbox_stream_stopped = False

        self.fps = FPS()
        self.tracker_visualizer = Visual_tracker(fps=25)
        self.orig_img_size = (0, 0)

        self.dict_data = dict_data
        self.digits = 5

        self.prev_imshow = time.time()

        self.task_display = threading.Thread(target=self.display, daemon=True)
        self.task_display_started = False
        self.check_display = False

        self.frame = Frame()
        self.cnt_img = 0
        self.cnt_bbox = 0

        self.start_time = -1
        self.time_start_receive = -1
        self.time_start_display = 0

    def _declare_queues(self):
        """
        Queue name setup
        """
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
            # print(f"get bbox from stage 2 {predictions}")

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
        """ Listening loop """
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

    def run(self):
        """ Main runner """
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

    def cleanup(self):
        """ Main runner """
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

    def display(self):
        pending_data_frame = None
        pending_data_bbox = None

        while not self.stop_event.is_set():

            if (
                    self.image_stream_stopped and
                    self.bbox_stream_stopped and
                    self.image_buffer_queue.empty() and
                    self.bbox_buffer_queue.empty()
            ):
                break

            if pending_data_frame is None:
                try:
                    pending_data_frame = self.image_buffer_queue.get(timeout=0.05)
                except Empty:
                    pass

            if pending_data_bbox is None:
                try:
                    pending_data_bbox = self.bbox_buffer_queue.get(timeout=0.05)
                except Empty:
                    pass

            if pending_data_frame is not None and pending_data_bbox is not None:
                if self.frame.showed == 0:
                    self.time_start_display = time.time()

                try:
                    if isinstance(pending_data_frame, list):
                        np_frames = np.array(pending_data_frame, dtype=np.uint8)
                    else:
                        np_frames = pending_data_frame

                    is_running = True

                    if np_frames.ndim == 3:
                        is_running = self.tracker_visualizer.run(np_frames, pending_data_bbox)
                        self.frame.showed += 1

                    elif np_frames.ndim == 4:
                        for i in range(np_frames.shape[0]):
                            single_frame = np_frames[i]
                            single_bbox = pending_data_bbox[i] if len(pending_data_bbox) > i else []

                            is_running = self.tracker_visualizer.run(single_frame, single_bbox)
                            self.frame.showed += 1

                            if not is_running:
                                break
                    else:
                        print(f"[Warning]  shape: {np_frames.shape}")

                    pending_data_frame = None
                    pending_data_bbox = None

                    if not is_running:
                        self.stop_event.set()
                        break

                except Exception as e:
                    print("[Tracker][display] run visualizer error:", e)
                    pending_data_frame = None
                    pending_data_bbox = None
            else:
                continue

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

    def data_for_csv(self):
        print("data for csv")

        self.dict_data["Time"] = Tools().get_datatime()
        self.dict_data["[T]totalTM"] = round(time.time() - self.start_time, self.digits) if self.start_time > 0 else -1
        self.dict_data["[T]FPSR"] = self.fps.mean
        self.dict_data["[1]totalFr"] = self.frame.total