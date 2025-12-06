import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics.nn.tasks import DetectionModel
import yaml

class Scheduler_split :
    def __init__(self):
        pass
    def first_layers(self):
        cfg = yaml.safe_load(open('..\cfg\yolo11n.yaml', 'r', encoding='utf-8'))
        model = DetectionModel(cfg)

        state_dict_head = torch.load('head.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict_head, strict=False)  # Bỏ qua các key bị thiếu của part2
        model.eval()

        print("Load head model done ! ")
