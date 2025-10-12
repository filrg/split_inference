from src.Tracker_threading import Tracker
import yaml
import pika
import os
import multiprocessing as mp
from multiprocessing import Process, shared_memory , Queue ,Manager


# if __name__ == "__main__":
#     try:
#         with open('config.yaml', 'r') as file:
#             config = yaml.safe_load(file)
#
#         with Manager() as manager:
#             shared_dict = manager.dict()
#             img_queue = mp.Queue()
#             bbox_queue = mp.Queue()
#
#             shared_dict["start_signal"] = False
#             shared_dict["start_time_programming"] = -1
#             shared_dict["origin_img_size"] = (-1 , -1 )
#
#             dict_data = manager.dict({
#                 "Time": -1,
#                 "PointCut": -1,
#                 "[T]totalTM": -1,
#                 "[T]FPSR": -1,
#                 "[1]totalFr": -1,
#                 "[1]totalTm": -1,
#                 "[1]outSze[T]": -1,
#                 "[1]outSze[2]": -1,
#                 "[2]totalTm": -1,
#                 "[2]outSize": -1
#             })
#
#             tracker_app = Tracker(config, img_queue, bbox_queue, shared_dict , dict_data)
#             display_app = Display(img_queue, bbox_queue , shared_dict , dict_data)
#
#             p1 = Process(target=tracker_app.run)
#             p2 = Process(target=display_app.run)
#
#             print("[Main] Starting processes...")
#             p1.start()
#             p2.start()
#
#             p1.join()
#             p2.join()
#
#             print("[Main] All processes stopped cleanly.")
#
#     except FileNotFoundError:
#         print("Error: 'config.yaml' not found. Please run this script from the project root.")
#     except Exception as e:
#         print(f"Failed to start processes: {e}")

if __name__ == "__main__":
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        tracker_app = Tracker(config)
        tracker_app.run()
    except FileNotFoundError:
        print("Error: 'config.yaml' not found. Please run this script from the project root.")
    except Exception as e:
        print(f"Failed to start processes: {e}")




