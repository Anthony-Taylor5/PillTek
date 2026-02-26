from __future__ import annotations
import sys
import time

import cv2


import argparse
from pathlib import Path

from ultralytics import YOLO
import threading

"""
Usage command:
python test_inference.py --weights runs/detect/runs/train_v82/weights/best.pt



"""

# Robust capture creator: tries FFMPEG first, then ANY
def create_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url, cv2.CAP_ANY)
    return cap

# Threaded video capture with reconnection logic for smoother FPS
class ThreadedVideoCapture:
    def __init__(self, src):
        self.src = src
        self.cap = create_capture(src)
        if not self.cap.isOpened():
            print("Cannot connect to stream!")
            exit()
        self.ret = False
        self.frame = None
        self.stopped = False
        self.fail_count = 0
        self.last_ok = time.time()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        # warm-up buffer to avoid early None frames
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
            self.last_ok = time.time()
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
                self.ret = True
                self.frame = frame
                self.fail_count = 0
                self.last_ok = time.time()
            else:
                self.fail_count += 1
                # If no frame for a while, attempt reconnect
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

DATASETS = {
    "v2": "v2_with_background",
    "v4": "v4_with_combined",
    "v6": "v6_with_hazards",
    "v8": "v8_with_hands"
}


def assert_dataset_layout(ds_dir: Path) -> Path:
    """
    Validate that the dataset folder contains data.yaml and train/valid/test folders.
    Returns the path to data.yaml if valid.
    """
    data_yaml = ds_dir / "data.yaml"
    missing = []

    if not data_yaml.exists():
        missing.append("data.yaml")
    for sub in ("train", "valid", "test"):
        if not (ds_dir / sub).exists():
            missing.append(sub + "/")

    if missing:
        raise FileNotFoundError(
            f"Dataset folder '{ds_dir}' is missing: {', '.join(missing)}\n"
            f"Expected structure:\n"
            f"  {ds_dir}/data.yaml\n"
            f"  {ds_dir}/train/\n"
            f"  {ds_dir}/valid/\n"
            f"  {ds_dir}/test/\n"
        )

    return data_yaml


def run_infer(weights: Path, device: str, imgsz: int, conf: float, source: str) -> None:
    """
    Run inference on a video source (URL or webcam index).
    
    Args:
        weights: Path to YOLO model weights
        device: Device to run on (cpu or GPU id)
        imgsz: Image size for inference
        conf: Confidence threshold
        source: Video source - can be URL (http://...) or webcam index (0, 1, 2, etc.)
    """
    model = YOLO(str(weights))
    print(f"[INFO] Inference using: {weights}")
    print(f"[INFO] Source: {source}")
    print("[INFO] Press 'q' in the window to quit.")

    window_name = "YOLO Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Determine if source is a webcam index or URL/file path
    if source.isdigit():
        src = int(source)
        print(f"[INFO] Using webcam at index {src}")
    else:
        src = source
        print(f"[INFO] Using stream: {source}")

    cap = ThreadedVideoCapture(src)
    no_frame_warned = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Don't exit immediately—ESP32 streams can hiccup. Keep trying.
            if not no_frame_warned:
                print("[WARN] Waiting for stream frames...")
                no_frame_warned = True
            time.sleep(0.05)
            continue
        no_frame_warned = False

        results = model.predict(
            source=frame,
            device=device,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )

        annotated = results[0].plot()
        cv2.imshow(window_name, annotated)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO26n on v2/v4 datasets and run webcam inference.")
    parser.add_argument(
        "--dataset",
        choices=["v2", "v4", "v6", "v8", "both"],
        default="v8",
        help="Which dataset to train on (v2, v4, v6, or both sequentially)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device: cpu or GPU id like 0 or 0,1")
    parser.add_argument("--weights", default=None, help="Weights for inference (required if --mode infer)")
    parser.add_argument("--conf", type=float, default=0.60, help="Confidence threshold for webcam inference")
    parser.add_argument("--source", default="http://192.168.0.211:81/stream", 
                        help="Video source: ESP32 stream URL or webcam index (0, 1, 2, etc.)")
    args = parser.parse_args()

    cwd = Path.cwd()

    
    if not args.weights:
        print("[ERROR] --weights is required when --mode infer")
        return 2
    w = Path(args.weights)
    if not w.is_absolute():
        w = cwd / w
    if not w.exists():
        print(f"[ERROR] Weights not found: {w}")
        return 2
    run_infer(weights=w, device=args.device, imgsz=args.imgsz, conf=args.conf, source=args.source)

       

    return 0


if __name__ == "__main__":
    raise SystemExit(main())