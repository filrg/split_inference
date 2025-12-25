from src.partition.time_layers import LayerProfiler
import yaml
from src.Utils import get_output_sizes


with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)

# profiler = LayerProfiler(config, mode="time")
# time_list = profiler.run()
#
# print(time_list)

profiler = LayerProfiler(config, mode="shape" , unit="MB")
shape_list = profiler.run()

print(shape_list)
