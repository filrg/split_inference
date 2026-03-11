# Split Inference

This project implements **Split Inference for YOLOv11** to enable real-time object detection on low-power **edge devices (Jetson Nano)** by dividing the neural network across multiple machines.

Instead of transmitting full video frames, the edge device executes the first part of the model (**head**) and sends only **intermediate feature maps** to another device that runs the remaining layers (**tail**). A **server** manages the workflow, while a **tracker** visualizes the results on a laptop display.

---

# Table of Contents

* [System Overview](#system-overview)
* [System Architecture](#system-architecture)
* [Inference Data Flow](#inference-data-flow)
* [Pipeline](#pipeline)
* [Project Structure](#project-structure)
* [Installation](#installation)
  * [Clone Repository](#1-clone-the-repository)
  * [Install Dependencies](#2-install-dependencies)
  * [Start RabbitMQ](#3-start-rabbitmq)
* [Configuration](#configuration)
* [Running the System](#running-the-system)
* [Outputs](#outputs)
* [Automatic Partitioning](#automatic-partitioning)
* [Tested Hardware](#tested-hardware)
* [Application Scenarios](#application-scenarios)
* [License](#license)

---

# System Overview

<p align="center">
  <img src="imgs/overview.png" width="850">
</p>

In traditional edge AI pipelines, raw video frames are transmitted to a centralized server for processing. This creates high network bandwidth usage and latency.

**Split inference** solves this by dividing the neural network into two parts:

1. **Head (Edge Device)** – processes the early layers of the model.
2. **Tail (Server / Cloud)** – processes the remaining layers.

Only **intermediate feature maps** are transmitted instead of full images, reducing bandwidth and improving scalability.

---

# System Architecture

The system consists of four main components.

## Stage 1 – Edge Device (Head)

Devices located at the edge such as **traffic cameras or embedded devices (Jetson Nano)**.

Responsibilities:

* Capture video frames
* Run the first layers of YOLOv11
* Compress intermediate feature maps using quantization
* Send feature maps to the network

---

## Stage 2 – Tail Device (Tail)

Devices located in the **cloud or high-performance servers**.

Responsibilities:

* Receive feature maps from edge devices
* Run the remaining layers of the neural network
* Produce final detection results

---

## Server – Controller

Central coordination service responsible for:

* Registering clients
* Selecting model cut-layers
* Managing inference workflow
* Coordinating communication using **RabbitMQ**

---

## Tracker – Viewer

Displays inference results on a laptop or desktop.

Features:

* bounding box visualization
* detection confidence
* FPS monitoring

This component is useful because **Jetson Nano typically runs without a display**.

---

# Inference Data Flow

<p align="center">
  <img src="imgs/START.png" width="700">
</p>

The system workflow:

1. Edge device captures video frames.
2. The head model processes early layers.
3. Intermediate **feature maps** are transmitted through the network.
4. Tail device completes the inference.
5. Detection results and origin images are sent to the tracker.

---

# Pipeline

<p align="center">
  <img src="imgs/SI_pipeline.drawio.png" width="900">
</p>

Pipeline steps:

1. Clients register with the server.
2. Server collects device information.
3. Clustering or profiling may be performed.
4. Layer execution time is measured.
5. The optimal **partition point** is computed.
6. The model is split and inference begins.

---

# Project Structure

```
split_inference/
│
├── client.py          # Edge or tail inference node
├── server.py          # Central controller
├── tracker.py         # Visualization interface
├── config.yaml        # System configuration
├── requirements.txt   # Python dependencies
│
├── imgs/              # Images used in README
│   ├── overview.png
│   ├── START.png
│   └── SI_pipeline.drawio.png
│
├── src/               # Core framework modules
└── output.csv         # Performance results
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/filrg/split_inference
cd split_inference
```

---

## 2. Install dependencies

Python **3.8 or higher** is required.

```bash
pip install -r requirements.txt
```

---

## 3. Start RabbitMQ

RabbitMQ is used for communication between distributed components.

RabbitMQ admin interface:

```
http://localhost:15672
```

Default credentials:

```
username: guest
password: guest
```

---

# Configuration

Edit **config.yaml** before running the system.

Example configuration:

```yaml
name: YOLO

server:
  cut-layer: a
  clients:
    - 1
    - 1
  model: yolo11n
  batch-frame: 5

model: yolo11n

rabbit:
  address: 127.0.0.1
  username: guest
  password: guest
  virtual-host: /
  queue_device_1: "device1"
  queue_device_2: "device2"

data: videos/video.mp4
log-path: res
control-count: 10
debug-mode: False
```

Feature map compression:

```yaml
compress:
  enable: True
  num_bit: 8
```

---

# Running the System

## Step 1 – Start Server

```bash
python server.py
```

---

## Step 2 – Start Clients

Edge device:

```bash
python client.py --layer_id 1
```

Optional CPU mode:

```bash
python client.py --layer_id 1 --device cpu
```

Tail device:

```bash
python client.py --layer_id 2
```

---

## Step 3 – Start Tracker

```bash
python tracker.py
```

The tracker window displays:

* original frames
* bounding boxes
* detection confidence
* FPS

Make sure:

```
tracker.enable: True
```

in `config.yaml`.

---

# Outputs

## Result File

```
output.csv
```

Contains performance statistics such as:

* FPS
* total inference time
* communication delay
* device utilization

---

## Sample Logs

```
Start Inference
FPS input: 30
End Inference
All time: 156.95 s
Inference time: 152.65 s
Utilization: 97.26 %
```

---

# Automatic Partitioning

Enable automatic optimization:

```yaml
partition:
  auto: True
  re-measure: True
```

The server will:

1. Benchmark layer execution times
2. Measure network transmission cost
3. Build a cost graph
4. Use **Dijkstra optimization** to determine the optimal split layer

---

# Tested Hardware

| Device           | Role                   |
| ---------------- | ---------------------- |
| Jetson Nano      | Edge Client (Head)     |
| Jetson Nano      | Tail Client            |
| Laptop / Desktop | Tracker                |
| DAI              | Server               |
| LAN Network      | RabbitMQ communication |

---

# Application Scenarios

* Smart traffic monitoring
* Edge surveillance AI
* Distributed deep learning research
* Bandwidth reduction experiments

---

# License

See [LICENSE](./LICENSE)
 