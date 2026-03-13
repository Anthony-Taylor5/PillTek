# PillTek: Pill Monitoring System

# PillTek — Vision-Based Medication Monitoring System

PillTek is a capstone project built by a five-person team at CSUN. It passively monitors medication adherence by detecting when a user interacts with a pill bottle using a combination of computer vision, BLE proximity sensing, and an ESP32-S3 camera module.

The system turns on automatically when the user is nearby (detected via a BLE beacon), streams live video from the ESP32 to a Python backend, runs YOLO object detection to identify pill bottles and hazards, uses MediaPipe to detect hands, and flags any bottle interaction in real time.

---

## How It Works

1. A **KBeacon BLE device** sits near the medication area. The ESP32 scans for it continuously.
2. When the beacon's RSSI crosses a threshold (user is close), the ESP32 activates its camera and notifies the Python backend via HTTP POST.
3. The Python backend receives the trigger, starts the inference pipeline, and begins pulling the MJPEG stream from the ESP32.
4. **YOLO** detects pill bottles (A, B, C, D, F) and hazards. **MediaPipe** detects hands independently on a separate thread.
5. When a hand bounding box overlaps a bottle bounding box, an interaction event is logged.
6. When the user walks away, the beacon goes out of range, and the camera shuts back off.

---

## Hardware

| Component | Details |
|---|---|
| ESP32-S3-CAM | Seeed XIAO ESP32-S3 Sense |
| BLE Beacon | KBeacon, MAC `dd:34:02:0a:2d:f1` |
| Host machine | Any Python 3.10+ machine on the same WiFi network |

---

## Project Structure

```
PillTek/
│
├── working_server/                 # ESP32 firmware (Arduino sketch)
│   └── cam_and_server_code.ino/
│       ├── cam_and_server_code.ino.ino   # Main sketch
│       ├── board_config.h                # Camera model pin config
│       ├── app_httpd.cpp                 # HTTP stream server
│       └── camera_pins.h                 # Pin definitions
│
├── dataset/                        # Training images (not tracked in full)
│   ├── A images/
│   ├── B images/
│   ├── C images/
│   ├── combined images/
│   ├── hazard images/
│   ├── hand images/
│   ├── same bottle images/
│   ├── all_images_original/
│   └── all_images_augment/
│
├── v2_with_background/             # Training dataset v2 (bottles + background)
├── v4_with_combined/               # v4 (+ combined scenes)
├── v6_with_hazards/                # v6 (+ hazard class)
├── v8_with_hands/                  # v8 (+ hand images)
├── v10_with_same_bottles/          # v10 (+ same-bottle disambiguation)
│
├── runs/                           # YOLO training outputs (weights, metrics)
│
├── test_with_hand_recognition.py   # Main inference script (YOLO + MediaPipe)
├── train_two_datasets_yolo26n.py   # Model training script
├── beacon_trigger.py               # Flask server — receives ESP32 events
├── detect_beacon.py                # Standalone BLE beacon scanner (PC-side)
├── infer_dataset_images_to_csv.py  # Batch inference logger → CSV
├── latency_diagnostic.py           # Per-stage pipeline timing tool
├── test_inference.py               # Simple YOLO-only inference on live stream
├── detect_people.py                # Person detection using YOLOv5 (early prototype)
├── motion_detection.py             # Frame-differencing motion detection (early prototype)
├── grab_frames.py                  # Capture test frames from ESP32 stream
├── take_pics.py                    # Interactive image capture tool for dataset collection
├── create_augments.py              # Offline image augmentation for training data
│
├── yolo26n.pt                      # Base YOLO model weights
├── hand_landmarker.task            # MediaPipe hand landmarker model file
├── inference_log.csv               # Sample output from the CSV logger
├── latency_report.csv              # Per-frame timing data from diagnostic tool
└── latency_summary.txt             # Aggregated latency stats and bottleneck analysis
```

---

## File-by-File Breakdown

### `working_server/` — ESP32 Firmware

The Arduino sketch that runs on the ESP32-S3. It handles three things at once: BLE scanning, WiFi camera streaming, and HTTP event notifications to the Python backend.

- Scans for the KBeacon by MAC address. When RSSI rises above `-70 dBm` (beacon is close), it sets a `beaconDetectedNear` flag. When RSSI drops back below the threshold (beacon is far), it sets `beaconDetectedFar`.
- Flags are checked in the main loop (not inside BLE callbacks) to avoid blocking the BLE stack, which would degrade scan timing.
- On a near event: activates the camera and sends `POST /trigger {"event": "beacon_near"}` to the Python server.
- On a far event: stops the camera and sends `POST /trigger {"event": "beacon_far"}`.
- The MJPEG stream is served at `http://<ESP32_IP>:81/stream`.
- **Important settings**: PSRAM must be enabled in Arduino IDE. Camera grab mode must be `CAMERA_GRAB_LATEST` for low latency. JPEG quality is set to `12` to keep frame size small.

---

### `test_with_hand_recognition.py` — Main Inference Script

This is the core of the system. It pulls the live stream from the ESP32 and runs YOLO + MediaPipe on every frame.

- **`ThreadedVideoCapture`**: reads frames in a background thread into a queue of size 1, always keeping the latest frame and dropping old ones. Includes auto-reconnect logic if the stream drops.
- **`PillWorker`**: a persistent background thread that runs YOLO inference. The main loop submits a frame and immediately moves on — it never waits for YOLO to finish.
- **`HandWorker`**: same idea but for MediaPipe hand detection. Runs on its own thread independently from YOLO.
- The main loop draws whatever results the workers have already computed, without stalling. This is why the display stays smooth even when inference is slower than the stream.
- **Interaction detection**: if any hand landmark falls inside (or within 10px padding of) a bottle bounding box across a majority of the last 16 frames, it's flagged as an interaction.
- **Bounding box smoothing**: bottle boxes are averaged over the last 5 frames to reduce jitter.
- **Color coding**: Gray = background class, Green = Bottle A, Blue = Bottle B, Yellow = Bottle C, Purple = Hazard, Red = hand (no overlap), Orange = hand interacting with a bottle.

**Usage:**
```bash
# Default (home network stream)
python test_with_hand_recognition.py --weights runs/detect/runs/train_v10/weights/best.pt

# Hotspot
python test_with_hand_recognition.py --weights runs/detect/runs/train_v10/weights/best.pt --source http://192.168.0.211:81/stream

# Webcam
python test_with_hand_recognition.py --weights runs/detect/runs/train_v10/weights/best.pt --source 0
```

---

### `beacon_trigger.py` — Flask Event Receiver

A small Flask server that listens on port `5000` for POST requests from the ESP32.

- `POST /trigger {"event": "beacon_near"}` → starts `test_with_hand_recognition.py` as a subprocess.
- `POST /trigger {"event": "beacon_far"}` → terminates that subprocess.
- Only one subprocess runs at a time. If a near event comes in while the script is already running, it's ignored.

**Usage:**
```bash
python beacon_trigger.py
# Runs on 0.0.0.0:5000
```

---

### `train_two_datasets_yolo26n.py` — Model Training

Trains the custom YOLO model through up to five dataset versions, each building on the last.

- Starts from `yolo26n.pt` (the base pretrained weights).
- Each version adds more data: v2 adds background, v4 adds combined scenes, v6 adds hazards, v8 adds hand images, v10 adds same-bottle disambiguation.
- `--dataset both` runs all five stages sequentially, fine-tuning from the previous stage's best weights. This is how the final `best.pt` was produced.
- Validates the dataset folder layout before training (`data.yaml` + `train/`, `valid/`, `test/` must exist).

**Usage:**
```bash
# Train on one dataset
python train_two_datasets_yolo26n.py --dataset v10 --epochs 50 --device cpu

# Train all stages sequentially
python train_two_datasets_yolo26n.py --dataset both --epochs 50 --device cpu
```

---

### `infer_dataset_images_to_csv.py` — Batch Inference Logger

Runs the trained model over a folder of images and writes results to a CSV file. Useful for evaluating model performance across the full dataset.

Output columns: `image_name`, `bottles_detected`, `hand_present`, `interaction`, `hazard_flag`, `raw_detections` (full JSON), `timestamp`.

**Usage:**
```bash
python infer_dataset_images_to_csv.py --model runs/detect/runs/train_v10/weights/best.pt --input dataset/
python infer_dataset_images_to_csv.py --model best.pt --input dataset/ --output results.csv --conf 0.5
```

---

### `latency_diagnostic.py` — Pipeline Timing Tool

Instruments every stage of the pipeline — capture, queue wait, YOLO inference, MediaPipe inference — and outputs per-frame timing to a CSV and a human-readable summary report.

Key metrics it tracks: queue depth (tells you if inference is falling behind), OpenCV read time (network latency to ESP32), YOLO time, MediaPipe time, and estimated display lag.

**Usage:**
```bash
# Stream only (no models)
python latency_diagnostic.py --no-models

# Full pipeline
python latency_diagnostic.py --yolo runs/detect/runs/train_v10/weights/best.pt --frames 200
```

---

### `detect_beacon.py` — PC-Side BLE Scanner

Standalone script to verify the KBeacon is visible and check its RSSI from the host machine. Useful for debugging proximity thresholds without involving the ESP32.

```bash
python detect_beacon.py
```

---

### `take_pics.py` — Dataset Image Capture

Shows a live feed from the ESP32 stream. Press a key to save a labeled frame to the correct dataset folder:

| Key | Saved to | Label |
|---|---|---|
| `a` | `A images/` | Bottle A |
| `b` | `B images/` | Bottle B |
| `c` | `C images/` | Bottle C |
| `d` | `combined images/` | Mixed scene |
| `e` | `hazard images/` | Hazard |
| `f` | `hand images/` | Hand |
| `x` | `same bottle images/` | Same-bottle scene |

---

### `create_augments.py` — Image Augmentation

Takes the original 300 captured images and generates 2 augmented copies of each, tripling the dataset size. Augmentations include random rotation, flips, crop, color jitter, Gaussian blur, and hue/saturation shifts.

Output goes to `all_images_augment/`. The class label is parsed from the filename (e.g., `pill~bottle-A_timestamp.jpg` → class `A`).

---

### `grab_frames.py` — Quick Frame Capture

Connects to the ESP32 stream and saves 5 test frames as `images/testframe0.jpg` through `testframe4.jpg`. Used to verify the stream is working and check image quality before running inference.

---

### `test_inference.py` — YOLO-Only Inference

Simpler version of the main inference script — YOLO only, no MediaPipe. Good for quickly testing a new set of weights without the hand detection overhead.

---

### `detect_people.py` — Person Detection (Prototype)

Early prototype using a pretrained `yolov5s.pt` to detect people in frame, as an alternative proximity trigger before BLE was integrated. Not used in the final pipeline.

---

### `motion_detection.py` — Motion Detection (Prototype)

Early prototype using frame differencing to detect motion in a defined ROI. Predates the YOLO pipeline. Not used in the final system.

---

## Setup

### Requirements

```bash
pip install ultralytics opencv-python mediapipe flask bleak pillow torchvision
```

MediaPipe's `hand_landmarker.task` model file is downloaded automatically on first run if it's not already present.

### ESP32 Setup

1. Open `working_server/cam_and_server_code.ino/cam_and_server_code.ino.ino` in Arduino IDE.
2. Set your WiFi credentials in the sketch.
3. In Arduino IDE: **Tools > Board > XIAO_ESP32S3**, **Tools > PSRAM > OPI PSRAM** (required).
4. Upload to the board. The stream will be available at `http://<ESP32_IP>:81/stream`.

### Running the System

```bash
# Terminal 1 — Start the Flask trigger server
python beacon_trigger.py

# The ESP32 will auto-start inference when the beacon is detected nearby.
# Or run inference manually:
python test_with_hand_recognition.py --weights runs/detect/runs/train_v10/weights/best.pt
```

---

## Model Classes

The trained model detects six classes:

| Class | Description |
|---|---|
| `-` (class 0) | Background / no object |
| Bottle A | Pill bottle type A |
| Bottle B | Pill bottle type B |
| Bottle C | Pill bottle type C |
| Bottle D | Combined scene bottle |
| Bottle F | Additional bottle variant |
| hazard | Hazard / dangerous combination |

---

## Known Issues & Notes

- The ESP32 stream URL defaults to `http://192.168.0.211:81/stream` across most scripts. Update this if your network assigns a different IP.
- WiFi instability (especially on a mobile hotspot) is the primary remaining source of latency spikes. A stable router significantly improves performance.
- `CAMERA_GRAB_LATEST` mode on the ESP32 is critical for low latency. `CAMERA_GRAB_WHEN_EMPTY` causes multi-second stalls.
- PSRAM must be enabled in Arduino IDE or the camera will not function.
- NimBLE library v1.4.2 is the tested stable version. v2.3.6 had compatibility issues.
