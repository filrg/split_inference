import pika
import uuid
import argparse
import yaml

import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler
from src.partition.consumers import MessageSender , MessageReceiver

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')

args = parser.parse_args()

with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


if __name__ == "__main__":
    if config["partition"]["auto"] and config["partition"]["re-measure"] :
        layer_id = args.layer_id
        if layer_id == 1:
            app = MessageSender(config)
        else :
            app = MessageReceiver(config)
        app.run()
        app.clean()

    client_id = uuid.uuid4()
    address = config["rabbit"]["address"]
    username = config["rabbit"]["username"]
    password = config["rabbit"]["password"]
    virtual_host = config["rabbit"]["virtual-host"]

    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "message": "Hello from Client!"}

    device = None

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using device: {torch.cuda.get_device_name(device)}")
        else:
            device = "cpu"
            print(f"Using device: CPU")
    else:
        device = args.device
        print(f"Using device: {device}")
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
    channel = connection.channel()

    scheduler = Scheduler(client_id, args.layer_id, channel, device , config["tracker"]["enable"])
    client = RpcClient(client_id, args.layer_id, address, username, password, virtual_host, scheduler.inference_func, scheduler.check_compress_func, device)
    client.send_to_server(data)
    client.wait_response()
