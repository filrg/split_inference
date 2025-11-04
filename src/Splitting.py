import torch
from ultralytics import YOLO
from collections import OrderedDict

model = YOLO("../cfg/yolo11n.pt").model
# pick your cut layer ( only one output ) 
split_index = 4
print(f"\n Splitting model at layer index = {split_index}")

full_state_dict = model.state_dict()
head_state_dict = OrderedDict()
tail_state_dict = OrderedDict()

print("   Processing state_dict keys...")
for key, value in full_state_dict.items():
    if not key.startswith('model.'):
        continue

    try:
        layer_index = int(key.split('.')[1])

        if layer_index < split_index:
            head_state_dict[key] = value
        else:
            tail_state_dict[key] = value

    except (ValueError, IndexError):
        tail_state_dict[key] = value

print(f"   Head 1 has {len(head_state_dict)} keys.")
print(f"   Tail 2 has {len(tail_state_dict)} keys.")

torch.save(head_state_dict, "head.pt")
torch.save(tail_state_dict, "tail.pt")

print("\nState dictionaries saved with original keys:")
print(f" - head.pt (layers 0 → {split_index - 1})")
print(f" - tail.pt (layers {split_index} → end)")