import pickle
from tqdm import tqdm
import torch
import cv2
from src.Model import SplitDetectionPredictor
from src.Compress import Encoder,Decoder
from src.Utils import load_ground_truth, compute_map , format_size
import os
import copy
import time


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"

        self.size_mess_cl1_2_tracker = -1
        self.size_mess_cl1_2_cl2 = -1
        self.size_mess_cl2_2_tracker = -1 

    def send_next_layer(self, intermediate_queue, data, logger, compress,  signal = 'CONTINUE'):
        try :
            if signal != 'STOP':
                if compress["enable"]:
                    data["layers_output"] = [t.cpu().numpy() if isinstance(t, torch.Tensor) else None for t in
                                             data["layers_output"]]
                    logger.log_info(f'Start Encode.')
                    data["layers_output"], data["shape"] = Encoder(data_output=data["layers_output"],
                                                                   num_bits=compress["num_bit"])
                    logger.log_info(f'End Encode.')
                else:
                    data["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in
                                             data["layers_output"]]
                message = pickle.dumps({
                    "action": "OUTPUT",
                    "data": data
                })
                if self.size_mess_cl1_2_cl2 == - 1:
                    self.size_mess_cl1_2_cl2 = len(message)

                self.channel.basic_publish(
                    exchange='',
                    routing_key=intermediate_queue,
                    body=message,
                )
            else:
                message = pickle.dumps(data)
                self.channel.basic_publish(
                    exchange='',
                    routing_key=intermediate_queue,
                    body=message,
                )
        except Exception as e:
            logger.log_error(f"[send_next_layer]: Failed to send data to next layer. Error: {e}")
    def send_to_tracker(self, tracker_queue, predictions, frame_index, logger, signal='CONTINUE' ,
                        total_time = -1 ):
    # send bounding box to tracker from client 2 to tracker
        try:
            if signal != 'STOP':
                if not isinstance(predictions, (list, tuple)) or len(predictions) == 0 or not isinstance(predictions[0],
                                                                                                         torch.Tensor):
                    logger.log_warning(
                        f"Frame {frame_index}: Invalid prediction format received. Skipping send to tracker.")
                    return

                prediction_tensor = predictions[0]
                prediction_tensor_cpu = prediction_tensor.cpu()

                message_to_tracker = {
                    "predictions": prediction_tensor_cpu,
                    "frame_index": frame_index
                }
                if self.size_mess_cl2_2_tracker == -1 :
                    self.size_mess_cl2_2_tracker = len(message_to_tracker)

            else:
                message_to_tracker = {
                    'signal' : 'STOP' ,
                    'total_time' : total_time ,
                    'size_mess2tracker' : format_size(self.size_mess_cl2_2_tracker)
                }

            message_bytes = pickle.dumps(message_to_tracker)

            self.channel.basic_publish(
                exchange='',
                routing_key=tracker_queue,
                body=message_bytes
            )
        except Exception as e:
            logger.log_error(f"[send_to_tracker]: Failed to send data to tracker. Error: {e}")

    def send_ori_img(self, tracker_queue, frame_to_send, frame_index, orig_img_size, logger, total_frames=-1,
                     signal='CONTINUE' , total_time = -1 ):
    # send origin images from client 1 to tracker
        try:
            if signal != 'STOP':
                message = {
                    "ori_img": frame_to_send,
                    "frame_index": frame_index,
                    "orig_img_size": orig_img_size,
                    "total_frames": total_frames,
                }
            else:
                message = {
                    "signal" : 'STOP',
                    "total_time" : total_time,
                    "size_mess2tracker" : format_size(self.size_mess_cl1_2_tracker),
                    "size_mess2cl2" : format_size(self.size_mess_cl1_2_cl2)
                }


            message_bytes = pickle.dumps(message)
            if self.size_mess_cl1_2_tracker == -1 :
                self.size_mess_cl1_2_tracker = len(message_bytes)
            self.channel.basic_publish(
                exchange='',
                routing_key=tracker_queue,
                body=message_bytes
            )
        except Exception as e:
            logger.log_error(f"[send_ori_img]: Failed to send data to tracker. Error: {e}")

    def first_layer(self, model, data, save_layers, batch_frame, logger, compress):
        start_time = time.time()
        input_image = []
        lst_frame = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        frame_index = 1
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        model.eval()
        model.to(self.device)
        video_path = data
        cap = cv2.VideoCapture(video_path)

        total_frames = self.get_total_frames(video_path)

        if not cap.isOpened():
            logger.log_error(f"Not open video")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.log_info(f"FPS input: {fps}")

        path = None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            ret, frame = cap.read()
            # send origin frame
            if not ret or frame is None:
                y = 'STOP'
                total_time = time.time() - start_time
                self.send_next_layer(self.intermediate_queue, y, logger, compress , signal='STOP')
                self.send_ori_img(self.ori_img_queue, y, frame_index, (0, 0), logger, signal='STOP' ,
                                  total_time= total_time)
                break

            h, w, c = frame.shape
            orig_img_size = (h, w)
            # make border
            # size = max(h, w)
            # if h > w:
            #     border_size = h - w
            #     frame = cv2.copyMakeBorder(frame, 0, 0, 0, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # else:
            #     border_size = w - h
            #     frame = cv2.copyMakeBorder(frame, 0, border_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


            # self.send_ori_img(self.ori_img_queue, frame, frame_index, orig_img_size, logger, total_frames)
            lst_frame.append(frame)
            frame = cv2.resize(frame, (640, 640))
            frame = frame.astype('float32') / 255.0
            tensor = torch.from_numpy(frame).permute(2, 0, 1)  # shape: (3, 640, 640)
            input_image.append(tensor)

            if len(input_image) == batch_frame:
                # for i, t in enumerate(lst_frame):
                #     if isinstance(t, torch.Tensor):
                #         print(f"Frame {i}:")
                #         print(f"  type     : {type(t)}")
                #         print(f"  shape    : {t.shape}")
                #         print(f"  dtype    : {t.dtype}")
                #         print(f"  device   : {t.device}")
                #     else:
                #         print(f"Frame {i}: not a Tensor, type = {type(t)}")
                #
                # print(f"len(lst_frame) = {len(lst_frame)}")
                self.send_ori_img(self.ori_img_queue, lst_frame , frame_index, orig_img_size, logger, total_frames)
                input_image = torch.stack(input_image)
                logger.log_info(f'Start inference {batch_frame} frames.')
                input_image = input_image.to(self.device)
                # Prepare data
                predictor.setup_source(input_image)
                for predictor.batch in predictor.dataset:
                    path, input_image, _ = predictor.batch

                # Preprocess
                preprocess_image = predictor.preprocess(input_image)

                # Head predict
                y = model.forward_head(preprocess_image, save_layers)
                logger.log_info(f'End inference {batch_frame} frames.')

                y["img_shape"] = preprocess_image.shape[2:]
                y["orig_imgs_shape"] = input_image.shape[2:]
                y["orig_imgs"] = copy.copy(input_image)

                y["width"] = width
                y["height"] = height

                self.send_next_layer(self.intermediate_queue, y, logger, compress)
                logger.log_info('Send a message.')
                input_image = []
                lst_frame = []
                pbar.update(batch_frame)
                frame_index += 1
            else:
                continue
        print(f'size message: {self.size_mess_cl1_2_cl2} bytes.')
        logger.log_info(f'size message: {self.size_mess_cl1_2_cl2} bytes.')
        cap.release()
        pbar.close()
        logger.log_info(f"Finish Inference.")

    def last_layer(self, model, batch_frame, logger, compress):
        start_time = time.time()
        num_last = 1
        count = 0
        frame_index = 1

        model.eval()
        model.to(self.device)
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)


        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True)
            if method_frame and body:
                logger.log_info(f'Receive a message.')

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"]

                    if compress["enable"]:
                        logger.log_info(f'Start Decode.')
                        y["layers_output"] = Decoder(y["layers_output"], y["shape"])
                        logger.log_info(f'End Decode.')
                        y["layers_output"] = [torch.from_numpy(t) if t is not None else None for t in
                                              y["layers_output"]]

                    y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]

                    # Tail predict
                    logger.log_info(f'Start inference {batch_frame} frames.')
                    predictions = model.forward_tail(y)
                    self.send_to_tracker(self.bbox_queue, predictions, frame_index, logger)
                    frame_index += batch_frame

                    logger.log_info(f'End inference {batch_frame} frames.')

                    pbar.update(batch_frame)
                else:
                    total_time = time.time() - start_time
                    self.send_to_tracker(self.bbox_queue, 'STOP', frame_index, logger, 'STOP', total_time)
                    count += 1
                    if count == num_last:
                        break
                    continue
            else:
                continue
        pbar.close()
        logger.log_info(f"Finish Inference.")

    def middle_layer(self, model):
        pass

    def inference_func(self, model, data, num_layers, save_layers, batch_frame, logger, compress):
        if self.layer_id == 1:
            self.first_layer(model, data, save_layers, batch_frame, logger, compress)
        elif self.layer_id == num_layers:
            self.last_layer(model, batch_frame, logger, compress)
        else:
            self.middle_layer(model)

    def check_first_layer(self, model, data, save_layers, batch_frame, logger, compress, cal_map):
        input_image = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        image_dir = "frames/"
        label_dir = "labels/"

        model.eval()
        model.to(self.device)

        """Lấy dữ liệu"""
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        image_ids = [os.path.splitext(f)[0] for f in image_files]
        image_paths = [os.path.join(image_dir, f) for f in image_files]

        path = None
        size = None

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        for i in range(0, len(image_paths), batch_frame):
            batch_path = image_paths[i:i + batch_frame]
            batch_ids = image_ids[i:i + batch_frame]

            for img_path in batch_path:
                img = cv2.imread(img_path)
                if size is None:
                    h, w = img.shape[:2]
                    size = [h,w]
                if img is None:
                    print(f"Error: Can't read {img_path}.")
                    continue
                frame = cv2.resize(img, (640, 640))
                tensor = torch.from_numpy(frame).float().permute(2, 0, 1)  # shape: (3, 640, 640)
                tensor /= 255.0
                input_image.append(tensor)
            input_image = torch.stack(input_image)
            input_image = input_image.to(self.device)

            # Prepare data
            predictor.setup_source(input_image)
            for predictor.batch in predictor.dataset:
                path, input_image, _ = predictor.batch

            # Preprocess
            preprocess_image = predictor.preprocess(input_image)

            # Head predict
            y = model.forward_head(preprocess_image, save_layers)
            y["batch_ids"] = batch_ids
            y["img"] = preprocess_image
            y["orig_imgs"] = input_image
            y["path"] = path
            y["size"] = size
            logger.log_info(f'Complete {batch_frame} frame.')
            self.send_next_layer(self.intermediate_queue, y, logger, compress)
            input_image = []
            pbar.update(batch_frame)

        y = 'STOP'
        self.send_next_layer(self.intermediate_queue, y, logger, compress, 'STOP')

        print(f'size message: {self.size_mess_cl1_2_cl2} bytes.')
        logger.log_info(f'size message: {self.size_mess_cl1_2_cl2} bytes.')
        pbar.close()
        logger.log_info(f"Finish Inference.")

    def check_last_layer(self, model, batch_frame, logger, compress, cal_map):
        image_dir = "frames/"
        label_dir = "labels/"
        label_output_dir = "labels/"
        os.makedirs(label_output_dir, exist_ok=True)

        frame_id = 0
        create_label = cal_map["create_label"]

        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})
        all_preds = []

        model.eval()
        model.to(self.device)
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True)
            if method_frame and body:

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"]
                    batch_ids = y["batch_ids"]

                    if compress["enable"]:
                        y["layers_output"] = Decoder(y["layers_output"], y["shape"])
                        y["layers_output"] = [torch.from_numpy(t) if t is not None else None for t in y["layers_output"]]

                    y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]
                    size = y["size"]
                    # Tail predict
                    predictions = model.forward_tail(y)

                    results = predictor.postprocess(predictions, y["img"], y["orig_imgs"], y["path"])
                    for img_id, res in zip(batch_ids, results):
                        for box in res.boxes.data.cpu().numpy():
                            x1, y1, x2, y2, conf, cls = box
                            all_preds.append(
                                [img_id, int(cls), float(x1), float(y1), float(x2), float(y2), float(conf)])

                        if create_label:
                            frame_name = f"frame_{frame_id:05}"
                            label_path = os.path.join(label_output_dir, frame_name + ".txt")

                            boxes = res.boxes.xyxy.cpu().numpy()
                            scores = res.boxes.conf.cpu().numpy()
                            classes = res.boxes.cls.cpu().numpy().astype(int)

                            with open(label_path, "w") as f:
                                for box, cls, conf in zip(boxes, classes, scores):
                                    if conf < 0.1:
                                        continue
                                    x1, y1, x2, y2 = box
                                    xc = (x1 + x2) / 2 / size[1]
                                    yc = (y1 + y2) / 2 / size[0]
                                    bw = (x2 - x1) / size[1]
                                    bh = (y2 - y1) / size[0]
                                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                            frame_id += 1

                    logger.log_info(f'Complete {batch_frame} frames.')

                    pbar.update(batch_frame)
                else:
                    break
            else:
                continue
        pbar.close()

        if create_label is False:
            list_map = []
            all_gts = load_ground_truth(label_dir, image_dir)
            for i in range(10):
                threshold = 0.5 + i * 0.05
                map_score = compute_map(all_preds, all_gts, iou_threshold=threshold)
                list_map.append(map_score)
                print(f"mAP@{threshold:.2f}: {map_score:.4f}")
                logger.log_info(f"mAP@{threshold:.2f}: {map_score:.4f}")
            if list_map:
                average = sum(list_map) / len(list_map)
            else:
                average = 0
            print(f"mAP@0.5:0.95: {average:.4f}")
            logger.log_info(f"mAP@0.5:0.95: {average:.4f}")
        logger.log_info(f"Finish Inference.")

    def check_compress_func(self, model, data, num_layers, save_layers, batch_frame, logger, compress, cal_map):
        if self.layer_id == 1:
            self.check_first_layer(model, data, save_layers, batch_frame, logger, compress, cal_map)
        elif self.layer_id == num_layers:
            self.check_last_layer(model, batch_frame, logger, compress, cal_map)
        else:
            self.middle_layer(model)

    def get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

