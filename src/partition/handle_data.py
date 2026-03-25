import sys , json , torch

from ultralytics import YOLO
from src.Compress import Encoder


class Data:
 def __init__(self , layer_times , comm_times , count_devices, verbose= False):
  self.layer_times_1 = layer_times[0]
  self.layer_times_2 = layer_times[1]

  self.comm_times = comm_times
  self.cost_1 = 0
  self.cost_2 = sum(self.layer_times_2)
  self.layer_times_1.insert(0, -1)
  self.layer_times_2.insert(0, -1)
  self.comm_times.insert(0, -1)
  if verbose :
   print(count_devices)
   print("Client 1 " , self.layer_times_1)
   print("Client 2 " ,self.layer_times_2)
   print(self.comm_times)

  #
  self.capacity = len(self.layer_times_1)
  self.cost = [[-1 for _ in range((self.capacity) * 2)] for _ in range((self.capacity) * 2)]
  self.num_points = len(self.layer_times_1) - 1


 def get_test_bed_cost(self):
  for i in range(1, self.capacity - 1):
   # option 1
   # self.cost[i][i + 1] = self.layer_times_1[i + 1]
   # self.cost[i + self.num_points][i + self.num_points + 1] = self.layer_times_2[i + 1]
   # self.cost[i][i + self.num_points + 1] = self.comm_times[i] + self.layer_times_2[i + 1]
   # option 2
   self.cost_1 += self.layer_times_1[i]
   self.cost_2 -= self.layer_times_2[i]
   self.cost[i][i + 1] = 0
   self.cost[i + self.num_points][i + self.num_points + 1] = 0
   self.cost[i][i + self.num_points + 1] = max(self.cost_1 + self.comm_times[i-1] , self.cost_2)
   # print(f"cost { self.cost[i][i + self.num_points + 1] } \t sum cost a {self.cost_1} \t sum cost b {self.cost_2}")

 def run(self):
  self.get_test_bed_cost()
  return self.cost

class EstimateSize:
 def __init__(self):
  pass
 def get_size(self , x, unit="MB"):
  """
  Return memory size of:
  - torch.Tensor
  - tuple / list (nested)
  - bytes / bytearray
  - int / float / bool / str  -> 0 byte (metadata)
  """
  # None
  if x is None:
   bytes_ = 0

  # Tensor
  elif torch.is_tensor(x):
   bytes_ = x.numel() * x.element_size()

  # Serialized data
  elif isinstance(x, (bytes, bytearray)):
   bytes_ = len(x)

  # Metadata (ignore)
  elif isinstance(x, (int, float, bool, str)):
   bytes_ = 0

  # Tuple / List (nested)
  elif isinstance(x, (tuple, list)):
   bytes_ = 0
   for t in x:
    bytes_ += get_size(t, unit="B")

  else:
   # Fallback: try __sizeof__ (very defensive)
   try:
    bytes_ = x.__sizeof__()
   except Exception:
    raise TypeError(f"Unsupported type: {type(x)}")

  # Unit convert
  if unit == "B":
   return bytes_
  if unit == "KB":
   return bytes_ / 1024
  if unit == "MB":
   return bytes_ / (1024 ** 2)

  raise ValueError("unit must be 'B', 'KB', or 'MB'")


 def save_json_simple(self ,data, path):
  with open(path, "w", encoding="utf-8") as f:
   json.dump(data, f, indent=2)

 def run(self):
  yolo = YOLO("yolo11n.pt")
  model = yolo.model
  layers = model.model
  big_data = []

  for batch_size in range(1 , 31):
      x = torch.randn(batch_size, 3, 640, 640)

      y = {}   # lưu output các layer

      with torch.no_grad():
          for i, layer in enumerate(layers):

              if layer.f != -1:
                  if isinstance(layer.f, int):
                      x = y[layer.f]
                  else:  # list
                      x = [
                          x if j == -1 else y[j]
                          for j in layer.f
                      ]
              # ------------------------------------

              x = layer(x)
              y[i] = x

              # print(f"Layer {i:02d} | {layer.__class__.__name__}")

      orin_size = []
      for i in range(len(y)):
          orin_size.append(get_size(y[i]))

      # print(orin_size)
      encoder_size = []

      for i in range(len(y) - 1):
          encoder_size.append(get_size(Encoder(y[i] , num_bits=8)))
      data = {
          "batchsize": batch_size,
          "non-compress": orin_size,
          "compress": encoder_size
      }

      big_data.append(data)

  path = 'res/size_output_layers.json'
  save_json_simple(data=big_data, path=path)

