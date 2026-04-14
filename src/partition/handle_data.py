import sys , json , torch

from ultralytics import YOLO
from src.Compress import Encoder


class Data:
    def __init__(self, layer_times, comm_times, count_devices, verbose=False):
        self.stage_1 = layer_times[0]
        self.stage_2 = layer_times[1]
        self.depth = len(self.stage_2)

        self.comm_times = comm_times  # no append(0)
        self.comm_times.append(0)

        self.vertex_cost = []
        self.link_cost = []

        self.cost = {}

        # parameters
        self.e = 0.2    # bigger , lefter
        self.c = 0.1

        print("Running on handle Data")

    def get_cost(self):

        # ---- RESET ----
        self.vertex_cost = []

        N = 2 * self.depth + 2
        self.link_cost = [[-1 for _ in range(N)] for _ in range(N)]

        # ---- VERTEX ----
        self.vertex_cost.append(0)  # input

        # stage 1 (device)
        for i in range(self.depth):
            v_cost = self.stage_1[i] * ( 1 + i / self.depth + self.e * i )   # add parameters
            self.vertex_cost.append(v_cost)

        # stage 2 (server)
        for i in range(self.depth):
            v_cost = self.stage_2[i] * ( 2 - i / self.depth + self.c * ( self.depth - i))
            self.vertex_cost.append(v_cost)

        self.vertex_cost.append(0)  # output

        # ---- EDGES ----

        # device chain
        for i in range(1, self.depth + 1):
            self.link_cost[i - 1][i] = 0

        # server chain
        for i in range(self.depth + 1, 2 * self.depth + 1):
            self.link_cost[i][i + 1] = 0

        # split edges
        for i in range(1, self.depth + 1):
            self.link_cost[i][i + self.depth + 1] = self.comm_times[i - 1]

        # final edge
        self.link_cost[2 * self.depth][2 * self.depth + 1] = 0

        self.cost["vertex"] = self.vertex_cost
        self.cost["link"] = self.link_cost

        print(f"vertex cost = \n{self.vertex_cost}")
        print(f"comm time  = \n{self.comm_times}")

    def find_best_split_point(self):
        min_cost = 1e8
        split_point = 0
        cum_stage_1 = 0
        cum_stage_2 = 0

        for cost in self.stage_2:
            cum_stage_2 += cost

        for i in range(len(self.comm_times)):
            cost = max(cum_stage_1 + self.comm_times[i] , cum_stage_2)
            cum_stage_1 += self.stage_1[i]
            cum_stage_2 -= self.stage_2[i]
            if cost < min_cost:
                min_cost = cost
                split_point = i + 2

        print(f"best cost {split_point}")

    def run(self):
        self.get_cost()
        self.find_best_split_point()
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

     for batch_size in range(1, 31):
         x = torch.randn(batch_size, 3, 640, 640)

         y = {}  # lưu output các layer

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
             encoder_size.append(get_size(Encoder(y[i], num_bits=8)))
         data = {
             "batchsize": batch_size,
             "non-compress": orin_size,
             "compress": encoder_size
         }

         big_data.append(data)

     path = 'res/size_output_layers.json'
     save_json_simple(data=big_data, path=path)
