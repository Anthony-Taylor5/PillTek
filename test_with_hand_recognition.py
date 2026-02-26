from __future__ import annotations
import time
import urllib.request
import cv2
import argparse
import mediapipe as mp
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import threading

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

"""
Usage command:
python test_with_hand_recognition.py --weights runs/detect/runs/train_v82/weights/best.pt

Optional args:
  --source 0                               # webcam
  --source http://192.168.0.211:81/stream  # ESP32 stream (default)
  --conf 0.70
  --device cpu

Color coding:
  '-'      (class 0)  ->  Gray
  Bottle A (class 1)  ->  Green
  Bottle B (class 2)  ->  Blue
  Bottle C (class 3)  ->  Yellow
  Hazard   (class 4)  ->  Purple
  Hand (no overlap)   ->  Red
  Hand + bottle       ->  Orange
"""

# ─── Color definitions (BGR) ──────────────────────────────────────────────────
BOTTLE_COLORS = {
    0: (128, 128, 128),   # Gray   - '-'
    1: (0,   255,   0),   # Green  - Bottle A
    2: (255,   0,   0),   # Blue   - Bottle B
    3: (0,   255, 255),   # Yellow - Bottle C
    4: (255,   0, 128),   # Purple - Hazard
}
HAND_COLOR         = (0,   0, 255)   # Red
HAND_OVERLAP_COLOR = (0, 165, 255)   # Orange

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"


def ensure_hand_model():
    if not Path(HAND_MODEL_PATH).exists():
        print(f"[INFO] Downloading hand landmarker model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("[INFO] Download complete.")


# ─── Capture ──────────────────────────────────────────────────────────────────

def create_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url, cv2.CAP_ANY)
    return cap


class ThreadedVideoCapture:
    def __init__(self, src):
        self.src        = src
        self.cap        = create_capture(src)
        if not self.cap.isOpened():
            print("Cannot connect to stream!")
            exit()
        self.ret        = False
        self.frame      = None
        self.stopped    = False
        self.fail_count = 0
        self.last_ok    = time.time()
        self.thread     = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        time.sleep(1.5)

    def reopen(self):
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(0.5)
        self.cap = create_capture(self.src)
        if self.cap.isOpened():
            self.fail_count = 0
            self.last_ok    = time.time()
            return True
        return False

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self.reopen()
                time.sleep(0.2)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.ret        = True
                self.frame      = frame
                self.fail_count = 0
                self.last_ok    = time.time()
            else:
                self.fail_count += 1
                if self.fail_count >= 30 or (time.time() - self.last_ok) > 2.0:
                    self.reopen()
                    self.fail_count = 0
                time.sleep(0.01)

    def read(self):
        return self.ret, (None if self.frame is None else self.frame.copy())

    def release(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()


# ─── Decoupled worker threads ─────────────────────────────────────────────────
# Each worker runs independently at its own speed.
# The main loop never waits — it just draws whatever results are available.

class PillWorker:
    def __init__(self, model: YOLO, device: str, imgsz: int, conf: float):
        self.model   = model
        self.device  = device
        self.imgsz   = imgsz
        self.conf    = conf
        self._in_frame  = None
        self._result    = None
        self._in_lock   = threading.Lock()
        self._out_lock  = threading.Lock()
        self._new_frame = threading.Event()
        self._stopped   = False
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray):
        with self._in_lock:
            self._in_frame = frame
        self._new_frame.set()

    def get_result(self):
        with self._out_lock:
            return self._result

    def _run(self):
        while not self._stopped:
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()
            with self._in_lock:
                frame = self._in_frame
            if frame is None:
                continue
            try:
                results = self.model.predict(
                    source=frame, device=self.device,
                    imgsz=self.imgsz, conf=self.conf, verbose=False
                )
                with self._out_lock:
                    self._result = results
            except Exception as e:
                print(f"[WARN] Pill detection error: {e}")

    def stop(self):
        self._stopped = True


class HandWorker:
    def __init__(self, landmarker):
        self.landmarker = landmarker
        self._in_frame  = None
        self._in_ts     = 0
        self._result    = None
        self._in_lock   = threading.Lock()
        self._out_lock  = threading.Lock()
        self._new_frame = threading.Event()
        self._stopped   = False
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, timestamp_ms: int):
        with self._in_lock:
            self._in_frame = frame
            self._in_ts    = timestamp_ms
        self._new_frame.set()

    def get_result(self):
        with self._out_lock:
            return self._result

    def _run(self):
        while not self._stopped:
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()
            with self._in_lock:
                frame = self._in_frame
                ts    = self._in_ts
            if frame is None:
                continue
            try:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results  = self.landmarker.detect_for_video(mp_image, ts)
                with self._out_lock:
                    self._result = results
            except Exception as e:
                print(f"[WARN] Hand detection error: {e}")

    def stop(self):
        self._stopped = True


# ─── Drawing ──────────────────────────────────────────────────────────────────

def get_bottle_boxes(pill_results):
    boxes = []
    if pill_results is None:
        return boxes
    for box in pill_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        boxes.append((x1, y1, x2, y2, cls, conf))
    return boxes


def hand_overlaps_bottle(hand_landmarks, bottle_boxes, frame_shape) -> bool:
    h, w = frame_shape[:2]
    lm   = hand_landmarks[9]
    hx, hy = int(lm.x * w), int(lm.y * h)
    for (x1, y1, x2, y2, _, _) in bottle_boxes:
        if x1 <= hx <= x2 and y1 <= hy <= y2:
            return True
    return False


def draw_pill_boxes(frame, bottle_boxes, class_names):
    for (x1, y1, x2, y2, cls, conf) in bottle_boxes:
        color = BOTTLE_COLORS.get(cls, (255, 255, 255))
        name  = class_names[cls] if class_names and cls < len(class_names) else f"class {cls}"
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hands(frame, hand_results, bottle_boxes):
    if hand_results is None or not hand_results.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_landmarks in hand_results.hand_landmarks:
        overlapping = hand_overlaps_bottle(hand_landmarks, bottle_boxes, frame.shape)
        color       = HAND_OVERLAP_COLOR if overlapping else HAND_COLOR
        points      = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], color, 2)
        for (px, py) in points:
            cv2.circle(frame, (px, py), 4, color, -1)
        xs  = [p[0] for p in points]
        ys  = [p[1] for p in points]
        pad = 10
        hx1, hy1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        hx2, hy2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
        status = "Interacting" if overlapping else "Hand"
        cv2.putText(frame, status, (hx1, hy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_legend(frame, class_names):
    entries = []
    for cls_id, color in BOTTLE_COLORS.items():
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Class {cls_id}"
        entries.append((name, color))
    entries.append(("Hand",          HAND_COLOR))
    entries.append(("Hand + Bottle", HAND_OVERLAP_COLOR))
    x, y, line_h, box_w, box_h = 10, 10, 22, 16, 14
    for label, color in entries:
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, -1)
        cv2.putText(frame, label, (x + box_w + 6, y + box_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h


# ─── Inference ────────────────────────────────────────────────────────────────

def run_infer(weights: Path, device: str, imgsz: int, conf: float, source: str) -> None:
    ensure_hand_model()

    pill_model  = YOLO(str(weights))
    class_names = [pill_model.names[i] for i in sorted(pill_model.names.keys())]

    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    # Start persistent workers — they run independently, never block the display loop
    pill_worker = PillWorker(pill_model, device, imgsz, conf)
    hand_worker = HandWorker(landmarker)

    print(f"[INFO] Inference using: {weights}")
    print(f"[INFO] Source: {source}")
    print(f"[INFO] Classes: {class_names}")
    print("[INFO] Press 'q' to quit.")

    window_name = "Pill + Hand Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    src      = int(source) if source.isdigit() else source
    cap      = ThreadedVideoCapture(src)
    start_ms = int(time.time() * 1000)
    no_frame_warned = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if not no_frame_warned:
                print("[WARN] Waiting for stream frames...")
                no_frame_warned = True
            time.sleep(0.05)
            continue
        no_frame_warned = False

        timestamp_ms = int(time.time() * 1000) - start_ms

        # Submit frames to workers — fire and forget, no waiting
        pill_worker.submit(frame.copy())
        hand_worker.submit(frame.copy(), timestamp_ms)

        # Draw latest available results immediately
        pill_snap    = pill_worker.get_result()
        hand_snap    = hand_worker.get_result()
        bottle_boxes = get_bottle_boxes(pill_snap)

        draw_pill_boxes(frame, bottle_boxes, class_names)
        draw_hands(frame, hand_snap, bottle_boxes)
        draw_legend(frame, class_names)

        cv2.imshow(window_name, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    pill_worker.stop()
    hand_worker.stop()
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


# ─── Main ─────────────────────────────────────────────────────────────────────

DATASETS = {
    "v2": "v2_with_background",
    "v4": "v4_with_combined",
    "v6": "v6_with_hazards",
    "v8": "v8_with_hands"
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO + MediaPipe hand detection inference.")
    parser.add_argument("--weights", default=None, help="Path to YOLO weights (.pt)")
    parser.add_argument("--source",  default="http://192.168.0.211:81/stream",
                        help="Video source: ESP32 stream URL or webcam index (0, 1, 2...)")
    parser.add_argument("--conf",    type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--imgsz",   type=int,   default=640,  help="Image size")
    parser.add_argument("--device",  default="cpu",            help="Device: cpu or GPU id (0, 1...)")
    args = parser.parse_args()

    if not args.weights:
        print("[ERROR] --weights is required")
        return 2

    cwd = Path.cwd()
    w   = Path(args.weights)
    if not w.is_absolute():
        w = cwd / w
    if not w.exists():
        print(f"[ERROR] Weights not found: {w}")
        return 2

    run_infer(weights=w, device=args.device, imgsz=args.imgsz, conf=args.conf, source=args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())