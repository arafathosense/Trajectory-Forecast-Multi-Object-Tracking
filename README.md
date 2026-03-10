# Trajectory Forecast Multi Object Tracking

![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.0%2B-blue?logo=ultralytics\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue?logo=python\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Research](https://img.shields.io/badge/Research-Computer%20Vision-purple)

Trajectory Forecast is a lightweight, modular extension built on top of Ultralytics YOLO that enables real-time multi-object tracking with future motion prediction. It combines detection, tracking, motion history modeling, and velocity-based forecasting into a unified pipeline that can be used both as a command-line tool and as a Python library. The system is designed for practical computer vision applications such as traffic analytics, surveillance systems, robotics pipelines, and edge AI deployments. Unlike heavy deep-learning forecasting architectures, this framework uses a **lightweight motion model optimized for stability and speed**, making it suitable for:

* Autonomous systems
* Intelligent surveillance
* Traffic analytics
* Robotics perception pipelines
* Edge AI deployments

# Demo

[https://github.com/user-attachments/assets/9a1267c2-4ba4-49f6-9802-e80fed5e682f](https://github.com/user-attachments/assets/9a1267c2-4ba4-49f6-9802-e80fed5e682f)

The demo shows **real-time object tracking with projected future trajectories**, where predicted points indicate the expected motion path.


# Key Features

### Real-Time Performance

Designed for **low-latency environments** such as robotics and surveillance.

### Multi-Object Tracking

Supports advanced trackers including:

* ByteTrack
* BoT-SORT

### Motion Forecasting

Predicts **future object trajectories** using velocity-based projection.

### Modular Design

The framework is structured as reusable modules that can be integrated into larger vision systems.

### CLI + Python API

Use as:

* Command-line application
* Python library

# System Architecture

```
Video Input
     │
     ▼
YOLO Object Detection
     │
     ▼
Multi-Object Tracker
(ByteTrack / BoT-SORT)
     │
     ▼
Trajectory History Buffer
     │
     ▼
Velocity Estimation
     │
     ▼
Trajectory Forecast Module
     │
     ▼
Future Motion Visualization
```

# Installation

Clone the repository:

```bash
git clone https://github.com/arafathosense/trajectory-forecast.git
cd trajectory-forecast
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended environment:

```
Python ≥ 3.10
CUDA-enabled GPU (optional)
```

# Configuration

Create a configuration file named `config.yaml`.

```yaml
conf: 0.5
tracker: "bytetrack.yaml"
classes: [2, 3, 5]
history: 40
min_points: 8
forecast_steps: 35
vel_window: 10
ema_alpha: 0.7
forecast_color: [255, 0, 0]
```

### Parameter Description

| Parameter      | Description                                  |
| -------------- | -------------------------------------------- |
| conf           | Detection confidence threshold               |
| tracker        | Tracker configuration file                   |
| history        | Number of frames used for trajectory history |
| forecast_steps | Number of future trajectory points           |
| vel_window     | Frames used for velocity estimation          |
| ema_alpha      | Smoothing coefficient                        |


## CLI Inference

```bash
trajectory-forecast \
--model yolo26n.pt \
--source "https://tinyurl.com/bddswzba" \
--config "path/to/config.yaml"
```

Supported input sources:

* Webcam streams
* Video files
* Network streams
* Image sequences

# Python API

The framework can be embedded into Python pipelines.

```python
from tf import run_inference
from tf.config import ForecastConfig

config = ForecastConfig(
    conf=0.5,
    forecast_steps=50,
    ema_alpha=0.7,
    classes=[0, 2, 5, 6, 7]
)

run_inference(
    model="yolo26n.pt",
    source="video.mp4",
    config=config
)
```


# Forecasting Methodology

The trajectory forecasting module is based on **lightweight motion modeling**.

Pipeline components include:

### 1. Object Center Extraction

Bounding box centers are extracted from tracked objects.

### 2. Exponential Moving Average

Noise reduction using trajectory smoothing.

### 3. Sliding Window Velocity Estimation

Velocity is estimated using median displacement over recent frames.

### 4. Linear Motion Projection

Future positions are predicted using velocity extrapolation.

### 5. Stationary Filtering

Objects with near-zero movement are filtered to prevent unstable predictions.

This design provides **stable real-time trajectory prediction** without requiring computationally expensive deep neural networks.


# Example Applications

Trajectory prediction is useful in many AI domains:

| Application        | Example                       |
| ------------------ | ----------------------------- |
| Autonomous Driving | Vehicle path prediction       |
| Traffic Monitoring | Flow analysis                 |
| Robotics           | Motion planning               |
| Security Systems   | Suspicious movement detection |
| Smart Cities       | Crowd analytics               |


# Performance

Example benchmark (RTX GPU):

| Model       | FPS      | Forecast Steps |
| ----------- | -------- | -------------- |
| YOLO-Nano   | ~120 FPS | 30             |
| YOLO-Small  | ~85 FPS  | 35             |
| YOLO-Medium | ~55 FPS  | 40             |

Actual performance depends on **GPU, video resolution, and tracker configuration**.

# Acknowledgments

This project builds upon the open-source computer vision ecosystem developed by:

* Ultralytics for YOLO detection models
* Open-source multi-object tracking research community

# Contributing

Contributions are welcome.

You can contribute by:

* Improving trajectory prediction algorithms
* Adding new trackers
* Improving performance optimization
* Enhancing documentation

Please open an issue or submit a pull request.

# License

Released under the **MIT License**.

## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Image Computing, Image Processing, Machine Learning, Deep Learning, Computer Vision**


