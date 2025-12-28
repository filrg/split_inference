import sys , json

from ultralytics import YOLO
import torch
from src.Compress import Encoder

chubby_size = 10000000
#
# comm_times = [chubby_size, 484.752, 94.563, 41.322, 51.551, 28.33, 24.968, 16.532,
#  20.588, 17.426, 52.015, 67.003, 24.234, 100.574, chubby_size,
#  47.96, 12.344, 32.771, 20.26, 6.187, 21.604, 13.5, 2.423]
#
# layer_times_2 = [13150.81, 15468.55, 35690.06, 17578.67, 22310.97, 14488.44, 22642.61,
#  6065.51, 18103.24, 19662.81, 16138.71, 5363.45, 676.75, 16020.68,
#  1698.42, 1770.54, 33257.24, 7237.94, 1548.84, 29074.08, 3864.21,
#  246.86, 44671.88, 106679.49]
#
# layer_times_3 = [16026.25, 14163.68, 44144.43, 18671.9, 41584.95, 21272.84, 61831.02,
#  10811.47, 42549.29, 21310.4, 22975.56, 2589.66, 683.63, 19536.7,
#  2296.24, 3873.87, 26174.95, 4480.21, 482.07, 15115.18, 4723.94,
#  152.32, 59023.33, 116022.15]

class Data:
 def __init__(self , layer_times , comm_times , count_devices, verbose= False):
  if count_devices[0] == '1':
   # print('case 1')
   self.layer_times_1 = layer_times[0]
   self.layer_times_2 = layer_times[1]
  else :
   # print('case 2')
   self.layer_times_1 = layer_times[1]
   self.layer_times_2 = layer_times[0]
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
   self.cost[i][i + self.num_points + 1] = self.comm_times[i - 1] + max(self.cost_1, self.cost_2)

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


