# split_inference

## Configuration
Application configuration is in the `config.yaml` file:
```yaml
name: YOLO
server:
  cut-layer: a #or b, c
  clients:
    - 1
    - 1
  model: yolov8n
  batch-frame: 1
rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

data: video.mp4
log-path: .
control-count: 10
debug-mode: False
```
This configuration is use for server.

## How to Run
Alter your configuration, you need to run the server to listen and control the request from clients.
### Server
```commandline
python server.py
```
### Client
Now, when server is ready, run clients simultaneously with total number of client that you defined.

**Layer 1**

```commandline
python client.py --layer_id 1 
```
Where:
- `--layer_id` is the ID index of client's layer, start from 1.

If you want to use a specific device configuration for the training process, declare it with the `--device` argument when running the command line:
```commandline
python client.py --layer_id 1 --device cpu
```

## Result
Results include inference time, operating time, utilization. It locates in `result.log`.  
```text
2025-04-16 23:51:35,944 - my_logger - INFO - Start Inference
2025-04-16 23:51:35,982 - my_logger - INFO - FPS input: 30.0
2025-04-16 23:54:12,896 - my_logger - INFO - End Inference.
2025-04-16 23:54:12,899 - my_logger - INFO - All time: 156.95556831359863s
2025-04-16 23:54:12,900 - my_logger - INFO - Inference time: 152.65051984786987s
2025-04-16 23:54:12,900 - my_logger - INFO - Utilization: 97.26 %

```