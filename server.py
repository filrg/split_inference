import argparse
import sys
import signal
import time

from src.Server import Server
from src.Utils import delete_old_queues
import src.Log
import yaml
from src.controller import Controller
from src.handle_data import Data
from src.dijkstra import Dijkstra
from src.Utils import get_layer_output

parser = argparse.ArgumentParser(description="Split learning framework with controller.")
args = parser.parse_args()

with open('config.yaml') as file:
    config = yaml.safe_load(file)

address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(address, username, password, virtual_host)
    sys.exit(0)

def run_controller():
    app = Controller(config)
    data = app.run()
    layer_times = data["layer_times"]
    comm_times = data["comm_times"]
    cost = Data(layer_times, comm_times).run()
    dijkstra_app = Dijkstra(cost, data["name_devices"])
    print(get_layer_output(dijkstra_app.run()))
    return get_layer_output(dijkstra_app.run())

if __name__ == "__main__":
    app = Controller(config)
    data = app.run()
    layer_times = data["layer_times"]
    comm_times = data["comm_times"]
    cost = Data(layer_times, comm_times).run()
    dijkstra_app = Dijkstra(cost, data["name_devices"])
    print(get_layer_output(dijkstra_app.run()))
    split_point = get_layer_output(dijkstra_app.run())
    signal.signal(signal.SIGINT, signal_handler)
    delete_old_queues(address, username, password, virtual_host)
    server = Server(config , split_point)
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
