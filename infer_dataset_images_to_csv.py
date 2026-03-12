"""
YOLO Inference Logger
=====================
Runs a pretrained YOLOv26 model over a folder of images (or a single image)
and logs results to a CSV file.

Logged columns:
  image_name       – filename of the processed image
  bottles_detected – semicolon-separated list of detected bottle classes
  hand_present     – yes / no
  interaction      – hand->class if overlap detected, else "none"
  hazard_flag      – yes / no  (True when the "hazard" class is detected)
  raw_detections   – full JSON dump of every detection (class, confidence, bbox)
  timestamp        – ISO-8601 datetime of when the image was processed

Dataset folder structure expected:
  C:/Users/Anthony/Documents/CS490/code/dataset/
      A images/   -> images labelled like: pill~bottle-A_20260120_155305
      B images/
      C images/
      ...
      F images/
      X images/   (or similar subfolder name containing Bottle X images)
      hand images/
      hazard images/
      Background pics/
      combined images/
      all_images_augment/
      all_images_original/
      same bottle images/

Usage
-----
  # Process the full dataset folder
  python yolo_inference_logger.py --model path/to/best.pt \
      --input "C:/Users/Anthony/Documents/CS490/code/dataset"

  # Process a single subfolder
  python yolo_inference_logger.py --model path/to/best.pt \
      --input "C:/Users/Anthony/Documents/CS490/code/dataset/A images"

  # Custom output file and confidence threshold
  python yolo_inference_logger.py --model best.pt \
      --input "C:/Users/Anthony/Documents/CS490/code/dataset" \
      --output results.csv --conf 0.4

Requirements
------------
  pip install ultralytics opencv-python
  (MediaPipe is used for hand detection if installed; falls back to YOLO-only otherwise)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

# ── optional MediaPipe hand detection ────────────────────────────────────────
import urllib.request
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False



HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"


def ensure_hand_model():
    """Download the hand landmarker model file if it is not already present."""
    if not Path(HAND_MODEL_PATH).exists():
        print(f"[INFO] Downloading hand landmarker model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("[INFO] Download complete.")


def build_hand_landmarker():
    """Create and return a MediaPipe HandLandmarker configured for static images."""
    ensure_hand_model()
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics is required. Install it with: pip install ultralytics")



# ─────────────────────────────────────────────────────────────────────────────
# Class name helpers
# ─────────────────────────────────────────────────────────────────────────────

# Exact class names as they appear in your YOLO model.
# Your model has five bottle classes: A, B, C, D, and F (no E, no X).
BOTTLE_CLASSES = {"Bottle A", "Bottle B", "Bottle C", "Bottle D", "Bottle F"}

HAZARD_CLASS = "hazard"


def is_bottle(class_name: str) -> bool:
    """Return True if the class is one of the five pill-bottle variants."""
    return class_name.strip() in BOTTLE_CLASSES


def is_hazard(class_name: str) -> bool:
    return class_name.strip().lower() == HAZARD_CLASS.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box overlap helper
# ─────────────────────────────────────────────────────────────────────────────

def boxes_overlap(box_a, box_b) -> bool:
    """
    Both boxes are (x1, y1, x2, y2) in pixel coordinates.
    Returns True when they overlap.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe hand detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hands_mediapipe(image_bgr, landmarker) -> list:
    """
    Returns a list of bounding boxes [(x1,y1,x2,y2), ...] for each hand found.
    Coordinates are in pixels.
    """
    if not _MP_AVAILABLE or landmarker is None:
        return []

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results   = landmarker.detect(mp_image)

    hand_boxes = []
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            hand_boxes.append((x1, y1, x2, y2))

    return hand_boxes


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path: Path, model, conf_threshold: float, landmarker) -> dict:
    """
    Run YOLO (and optionally MediaPipe) on one image.
    Returns a dict matching the CSV columns.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  [WARN] Could not read image: {image_path.name}")
        return None

    # ── YOLO inference ────────────────────────────────────────────────────────
    results = model(image_bgr, conf=conf_threshold, verbose=False)[0]

    bottle_detections = []   # list of {"class": str, "conf": float, "box": [x1,y1,x2,y2]}
    raw_detections    = []

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        cls_name   = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

        det = {"class": cls_name, "conf": round(confidence, 3), "box": [x1, y1, x2, y2]}
        raw_detections.append(det)

        if is_bottle(cls_name) or is_hazard(cls_name):
            bottle_detections.append(det)

    # ── Hand detection ────────────────────────────────────────────────────────
    # Prefer MediaPipe; fall back to YOLO's own hand class if present.
    hand_boxes   = detect_hands_mediapipe(image_bgr, landmarker)
    hand_present = len(hand_boxes) > 0

    # ── Interaction detection ─────────────────────────────────────────────────
    interaction_labels = []
    if hand_present:
        for hbox in hand_boxes:
            for det in bottle_detections:
                if boxes_overlap(hbox, det["box"]):
                    interaction_labels.append(f"hand->{det['class']}")

    interaction = ";".join(sorted(set(interaction_labels))) if interaction_labels else "none"

    # ── Summarise bottle names ────────────────────────────────────────────────
    detected_names = [d["class"] for d in bottle_detections]
    bottles_str    = ";".join(detected_names) if detected_names else "none"
    hazard_flag    = any(is_hazard(n) for n in detected_names)

    return {
        "image_name"      : image_path.name,
        "bottles_detected": bottles_str,
        "hand_present"    : "yes" if hand_present else "no",
        "interaction"     : interaction,
        "hazard_flag"     : "yes" if hazard_flag else "no",
        "raw_detections"  : json.dumps(raw_detections),
        "timestamp"       : datetime.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image discovery
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def gather_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "image_name",
    "bottles_detected",
    "hand_present",
    "interaction",
    "hazard_flag",
    "raw_detections",
    "timestamp",
]

def write_csv_header(csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_row(csv_path: Path, row: dict):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLOv26 model over images and log results to CSV."
    )
    parser.add_argument(
        "--model", default= "runs/detect/runs/train_v10/weights/best.pt",
        help="Path to your YOLOv26 weights file (e.g. best.pt)"
    )
    parser.add_argument(
        "--input", default=r"C:\Users\Anthony\Documents\CS490\code\dataset", 
        help=(
            "Path to an image file or a folder of images. "
        )
    )
    parser.add_argument(
        "--output", default="inference_log.csv",
        help="Output CSV filename (default: inference_log.csv)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="YOLO confidence threshold (default: 0.5)"
    )
    return parser.parse_args()


def main():
    args  = parse_args()
    model = YOLO(args.model)
    landmarker = build_hand_landmarker() if _MP_AVAILABLE else None

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input path does not exist: {input_path}")

    csv_path = Path(args.output)
    write_csv_header(csv_path)

    images = gather_images(input_path)
    if not images:
        sys.exit(f"No images found at: {input_path}")

    print(f"Model        : {args.model}")
    print(f"Input        : {input_path}  ({len(images)} image{'s' if len(images) != 1 else ''})")
    print(f"Output       : {csv_path}")
    print(f"Conf         : {args.conf}")
    print(f"Bottle classes : Bottle A, B, C, D, F")
    print(f"MediaPipe    : {'enabled' if _MP_AVAILABLE else 'not installed – using YOLO hand class only'}")
    print("-" * 60)

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}", end=" ... ", flush=True)
        row = process_image(img_path, model, args.conf, landmarker)
        if row:
            append_csv_row(csv_path, row)
            bottles = row["bottles_detected"]
            hand    = row["hand_present"]
            inter   = row["interaction"]
            hazard  = "WARNING HAZARD" if row["hazard_flag"] == "yes" else ""
            print(f"bottles={bottles}  hand={hand}  interaction={inter}  {hazard}")
        else:
            print("SKIPPED")
    if landmarker is not None:
        landmarker.close()
    print("-" * 60)
    print(f"Done. Results saved to: {csv_path}")
    
if __name__ == "__main__":
    main()