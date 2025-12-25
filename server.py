import argparse
import sys
import signal, yaml

from src.Server import Server
from src.Utils import delete_old_queues
import src.Log
from src.partition.controller import Controller
from src.partition.handle_data import Data
from src.partition.dijkstra import Dijkstra
from src.Utils import get_layer_output , get_log , save_log

parser = argparse.ArgumentParser(description="Split learning framework with controller.")
args = parser.parse_args()

with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(address, username, password, virtual_host)
    sys.exit(0)

if __name__ == "__main__":
    if config["partition"]["auto"] :
        if config["partition"]["re-measure"] :
            data = Controller(config).run()
            layer_times = data["layer_times"]
            comm_times = data["comm_times"]
            cost = Data(layer_times, comm_times).run()
            print(layer_times[0])
            print(layer_times[1])
            print(comm_times)
            dijkstra_app = Dijkstra(cost, data["name_devices"])
            split_point = get_layer_output(dijkstra_app.run())
            # delete_old_queues(address, username, password, virtual_host)
            save_log(split_point)
        else:
            split_point = get_log()

        print(f"split point {split_point}")
        signal.signal(signal.SIGINT, signal_handler)
        # delete_old_queues(address, username, password, virtual_host)
        server = Server(config, split_point)
        # save_log(split_point)
    else :
        server = Server(config , -1)    # if 2nd para is -1 , get cut layer a , b, c or d
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
