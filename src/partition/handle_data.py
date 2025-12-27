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
   self.cost[i][i + 1] = 0
   self.cost[i + self.num_points][i + self.num_points + 1] = 0
   self.cost[i][i + self.num_points + 1] = self.comm_times[i - 1] + max(self.cost_1, self.cost_2)

 def run(self):
  self.get_test_bed_cost()
  return self.cost
