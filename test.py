import argparse
import yaml
from src.partition.consumers import MessageSender , MessageReceiver
from src.partition.controller import Controller
from src.partition.handle_data import Data
from src.partition.dijkstra import Dijkstra

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def start_running():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["sender", "receiver", "controller"], required=True)
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.role == "sender":
        app = MessageSender(config)
        app.run()
    elif args.role == "receiver":
        app = MessageReceiver(config)
        app.run()
    else:
        app = Controller(config)
        data = app.run()
        layer_times = data["layer_times"]
        comm_times = data["comm_times"]
        cost = Data(layer_times , comm_times ).run()
        dijkstra_app = Dijkstra(cost , data["name_devices"])
        dijkstra_app.run()

if __name__ == "__main__":
    start_running()