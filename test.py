from src.partition.time_layers import LayerProfiler
import yaml


with open('cfg/config.yaml') as file:
    config = yaml.safe_load(file)
profile = LayerProfiler(config)
profile.run(verbose=True)