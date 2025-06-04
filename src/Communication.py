import torch
import time
from queue import Full
import pickle

def sender_thread(queue_out, next_queue, channel):
    channel.queue_declare(queue=next_queue, durable=False)
    """Send forward to the next layer"""
    while True:
        if not queue_out.empty():
            data = queue_out.get()
            if data != 'STOP':
                data["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in data["layers_output"]]
                message = pickle.dumps({
                    "action": "OUTPUT",
                    "data": data
                })
                channel.basic_publish(
                    exchange='',
                    routing_key=next_queue,
                    body=message,
                )
            else:
                message = pickle.dumps(data)
                channel.basic_publish(
                    exchange='',
                    routing_key=next_queue,
                    body=message,
                )
                break
        time.sleep(0.01)

def receiver_thread(queue_in, previous_queue, channel):
    channel.queue_declare(queue=previous_queue, durable=False)
    while True:
        method_frame, header_frame, body = channel.basic_get(queue=previous_queue, auto_ack=True)
        if method_frame and body:
            message = pickle.loads(body)
            while True:
                try:
                    queue_in.put(message, block=False)
                    break
                except Full:
                    time.sleep(0.5)
            if message == 'STOP':
                break
        else:
            continue

