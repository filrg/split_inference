import torch
import time
from queue import Full
import pickle
import pika

def sender_thread(queue_out, next_queue, channel):
    channel.queue_declare(queue=next_queue, durable=False)
    """Send forward to the next layer"""
    while True:
        print(queue_out.qsize())
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

def receiver_thread(queue_in, previous_queue):
    credentials = pika.PlainCredentials('dai', 'dai')
    connection = pika.BlockingConnection(pika.ConnectionParameters('192.168.101.91', 5672, f'/', credentials))
    channel = connection.channel()
    channel.queue_declare(queue=previous_queue, durable=False)

    def on_message(ch, method, properties, body):
        message = pickle.loads(body)

        while True:
            try:
                print(queue_in.qsize())
                queue_in.put(message, block=False)
                break
            except Full:
                time.sleep(0.05)

        if message == "STOP":
            print("[Receiver] STOP received. Stopping consumer.")
            ch.stop_consuming()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=previous_queue, on_message_callback=on_message)
    channel.start_consuming()

    channel.close()
    connection.close()
