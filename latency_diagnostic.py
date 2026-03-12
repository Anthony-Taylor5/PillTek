"""
Pipeline Latency Diagnostic Tool
=================================
Instruments every stage of the ESP32 → Backend → Display pipeline
to pinpoint exactly where lag is accumulating.

Usage:
    python latency_diagnostic.py --stream http://192.168.0.211:81/stream
    python latency_diagnostic.py --stream http://192.168.0.211:81/stream --yolo path/to/best.pt
    
    # stream-only test
    Hotspot:
    python latency_diagnostic.py --no-models 
    Home:
    python latency_diagnostic.py --stream http://192.168.0.31:81/stream --no-models 
    
Output:
    - Live per-frame timing table in terminal
    - latency_report.csv  (raw data for every frame)
    - latency_summary.txt (aggregated stats per stage)
"""

import cv2
import time
import queue
import threading
import argparse
import csv
import os
from collections import deque
from datetime import datetime

# Optional model imports — gracefully skip if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found — YOLO stage will be skipped")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARN] mediapipe not found — MediaPipe stage will be skipped")


# ──────────────────────────────────────────────
# Timing utilities
# ──────────────────────────────────────────────

class FrameTimer:
    """Tracks timestamps for all stages of a single frame."""
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.timestamps = {}

    def mark(self, stage: str):
        self.timestamps[stage] = time.perf_counter()

    def delta(self, start: str, end: str) -> float:
        """Returns ms between two stages."""
        if start in self.timestamps and end in self.timestamps:
            return (self.timestamps[end] - self.timestamps[start]) * 1000
        return -1.0

    def total(self) -> float:
        """Total ms from first to last stage."""
        if len(self.timestamps) < 2:
            return -1.0
        vals = list(self.timestamps.values())
        return (vals[-1] - vals[0]) * 1000


class StageStats:
    """Rolling stats for a pipeline stage."""
    def __init__(self, name: str, window: int = 60):
        self.name = name
        self.samples = deque(maxlen=window)

    def add(self, ms: float):
        if ms >= 0:
            self.samples.append(ms)

    @property
    def avg(self): return sum(self.samples) / len(self.samples) if self.samples else 0
    @property
    def max(self): return max(self.samples) if self.samples else 0
    @property
    def min(self): return min(self.samples) if self.samples else 0
    @property
    def p95(self):
        if not self.samples: return 0
        s = sorted(self.samples)
        return s[int(len(s) * 0.95)]


# ──────────────────────────────────────────────
# Stage 1: Stream capture (threaded)
# ──────────────────────────────────────────────

class InstrumentedCapture:
    """
    Captures frames from ESP32 stream.
    Measures:
      - capture_interval: time between successive frames arriving (1000/FPS)
      - queue_depth:       how many frames are sitting unprocessed
      - queue_wait:        how long the consumer waits to get a frame
    """
    def __init__(self, src: str, maxsize: int = 30):
        self.src = src
        self.frame_counter = 0
        self.q = queue.Queue(maxsize=maxsize)  # large queue so we can observe buildup
        self.running = False
        self.last_capture_time = None
        self.capture_intervals = deque(maxlen=100)
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.src}")
        threading.Thread(target=self._reader, daemon=True, name="CaptureThread").start()
        print(f"[CAPTURE] Stream opened: {self.src}")
        # Warm up
        time.sleep(1.5)

    def _reader(self):
        while self.running:
            t_before_read = time.perf_counter()
            ok, frame = self.cap.read()
            t_after_read = time.perf_counter()

            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._lock:
                fid = self.frame_counter
                self.frame_counter += 1

                if self.last_capture_time is not None:
                    interval_ms = (t_before_read - self.last_capture_time) * 1000
                    self.capture_intervals.append(interval_ms)
                self.last_capture_time = t_before_read

            timer = FrameTimer(fid)
            timer.mark("t_captured")          # frame arrived from ESP32
            timer.timestamps["_read_ms"] = (t_after_read - t_before_read) * 1000

            try:
                self.q.put_nowait((frame, timer))
            except queue.Full:
                pass  # intentionally drop — we'll measure this separately

    def read(self) -> tuple:
        """Blocking read. Returns (frame, timer) or (None, None) on timeout."""
        try:
            t_wait_start = time.perf_counter()
            frame, timer = self.q.get(timeout=5.0)
            timer.timestamps["_queue_wait_ms"] = (time.perf_counter() - t_wait_start) * 1000
            timer.mark("t_dequeued")
            return frame, timer
        except queue.Empty:
            return None, None

    @property
    def queue_depth(self): return self.q.qsize()

    @property
    def avg_capture_interval(self):
        if not self.capture_intervals: return 0
        return sum(self.capture_intervals) / len(self.capture_intervals)

    def stop(self):
        self.running = False
        if hasattr(self, "cap"):
            self.cap.release()


# ──────────────────────────────────────────────
# Stage 2 & 3: YOLO + MediaPipe inference
# ──────────────────────────────────────────────

def run_yolo(model, frame, timer: FrameTimer):
    timer.mark("t_yolo_start")
    results = model(frame, verbose=False)
    timer.mark("t_yolo_end")
    return results

def run_mediapipe(hands, frame, timer: FrameTimer):
    timer.mark("t_mp_start")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    timer.mark("t_mp_end")
    return result


# ──────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────

FIELDNAMES = [
    "frame_id", "wall_time",
    "queue_depth_at_dequeue",
    "queue_wait_ms",
    "opencv_read_ms",
    "capture_to_dequeue_ms",
    "yolo_ms", "mediapipe_ms",
    "total_pipeline_ms",
    "estimated_display_lag_ms",
]

def timer_to_row(timer: FrameTimer, queue_depth: int, capture_interval_ms: float) -> dict:
    yolo_ms   = timer.delta("t_yolo_start", "t_yolo_end")
    mp_ms     = timer.delta("t_mp_start", "t_mp_end")
    total_ms  = timer.total()

    # Estimated display lag = age of frame when it finally appears on screen
    # = time from capture to end of all processing
    stages = list(timer.timestamps.keys())
    last_stage = stages[-1] if stages else None
    display_lag = timer.delta("t_captured", last_stage) if last_stage else -1

    return {
        "frame_id":                 timer.frame_id,
        "wall_time":                datetime.now().isoformat(timespec="milliseconds"),
        "queue_depth_at_dequeue":   queue_depth,
        "queue_wait_ms":            round(timer.timestamps.get("_queue_wait_ms", -1), 2),
        "opencv_read_ms":           round(timer.timestamps.get("_read_ms", -1), 2),
        "capture_to_dequeue_ms":    round(timer.delta("t_captured", "t_dequeued"), 2),
        "yolo_ms":                  round(yolo_ms, 2),
        "mediapipe_ms":             round(mp_ms, 2),
        "total_pipeline_ms":        round(total_ms, 2),
        "estimated_display_lag_ms": round(display_lag, 2),
    }


# ──────────────────────────────────────────────
# Live terminal display
# ──────────────────────────────────────────────

COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[96m",
    "dim":    "\033[2m",
}

def colorize(val: float, warn: float, bad: float, unit="ms") -> str:
    s = f"{val:.1f}{unit}"
    if val < 0:      return f"{COLORS['dim']}  N/A{COLORS['reset']}"
    if val >= bad:   return f"{COLORS['red']}{s}{COLORS['reset']}"
    if val >= warn:  return f"{COLORS['yellow']}{s}{COLORS['reset']}"
    return f"{COLORS['green']}{s}{COLORS['reset']}"

def print_header():
    print(f"\n{COLORS['bold']}{'─'*90}{COLORS['reset']}")
    print(f"{COLORS['bold']}{'FID':>6} │ {'Q':>3} │ {'Q-Wait':>8} │ {'OCV-Read':>9} │ {'YOLO':>8} │ {'MediaPipe':>10} │ {'Total':>8} │ {'Display Lag':>12}{COLORS['reset']}")
    print(f"{'─'*90}")

def print_row(row: dict, frame_num: int):
    if frame_num % 30 == 0:
        print_header()

    q     = row["queue_depth_at_dequeue"]
    q_col = COLORS["red"] if q > 5 else (COLORS["yellow"] if q > 2 else COLORS["green"])

    print(
        f"{row['frame_id']:>6} │ "
        f"{q_col}{q:>3}{COLORS['reset']} │ "
        f"{colorize(row['queue_wait_ms'],        50,  200):>16} │ "
        f"{colorize(row['opencv_read_ms'],       80,  200):>17} │ "
        f"{colorize(row['yolo_ms'],              80,  200):>16} │ "
        f"{colorize(row['mediapipe_ms'],         40,  100):>18} │ "
        f"{colorize(row['total_pipeline_ms'],   150,  500):>16} │ "
        f"{colorize(row['estimated_display_lag_ms'], 300, 1000):>20}"
    )


# ──────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────

def write_summary(stats: dict, output_path: str, args):
    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE LATENCY DIAGNOSTIC SUMMARY")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Stream:    {args.stream}")
    lines.append("=" * 60)

    bottleneck = max(stats.values(), key=lambda s: s.avg)

    for name, s in stats.items():
        if not s.samples:
            continue
        marker = " ◄ LIKELY BOTTLENECK" if s is bottleneck and s.avg > 50 else ""
        lines.append(f"\n[{s.name}]{marker}")
        lines.append(f"  avg:  {s.avg:.1f}ms")
        lines.append(f"  min:  {s.min:.1f}ms")
        lines.append(f"  max:  {s.max:.1f}ms")
        lines.append(f"  p95:  {s.p95:.1f}ms")

    lines.append("\n" + "=" * 60)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 60)
    lines.append("""
Queue Depth > 2 consistently:
  → Inference is slower than frame arrival rate.
  → Fix: process every 2nd frame, use smaller YOLO model,
         or limit queue size to 1 to always use latest frame.

Queue Wait > 200ms:
  → Consumer is blocked waiting — either inference is slow
    or stream has gaps/reconnections.

OpenCV Read > 150ms:
  → Network latency to ESP32, or stream is stalling.
  → Fix: check WiFi signal, reduce JPEG quality on ESP32,
         try FRAMESIZE_QQVGA.

YOLO > 150ms:
  → Model is too heavy for real-time use.
  → Fix: switch to yolov8n, use half=True (GPU), or skip frames.

MediaPipe > 80ms:
  → Hands model is slow on this hardware.
  → Fix: set max_num_hands=1, lower model_complexity=0.

Display Lag > 500ms:
  → Combined pipeline is too slow. Likely queue buildup.
  → Fix: maxsize=1 queue + always discard old frames.
""")

    report = "\n".join(lines)
    print("\n" + report)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\n[INFO] Summary saved → {output_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline latency diagnostic")
    parser.add_argument("--stream",    default="http://192.168.0.211:81/stream", help="ESP32 stream URL")
    parser.add_argument("--yolo",      default=None, help="Path to YOLO weights (.pt)")
    parser.add_argument("--no-models", default=None, action="store_true", help="Skip YOLO and MediaPipe, stream only")
    parser.add_argument("--frames",    type=int, default=200, help="Number of frames to sample (default 200)")
    parser.add_argument("--out-csv",   default=None, help="Output CSV path")
    parser.add_argument("--out-txt",   default=None, help="Output summary path")
    args = parser.parse_args()

    run_models = not args.no_models

    # Load models
    yolo_model = None
    hands_tracker = None

    if run_models and YOLO_AVAILABLE and args.yolo:
        print(f"[INFO] Loading YOLO from {args.yolo}")
        yolo_model = YOLO(args.yolo)
        # Warm up
        dummy = __import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8)
        yolo_model(dummy, verbose=False)
        print("[INFO] YOLO warmed up")

    if run_models and MP_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands_tracker = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[INFO] MediaPipe Hands loaded")

    # Stage stats
    stats = {
        "queue_wait":    StageStats("Queue Wait (consumer blocked)"),
        "opencv_read":   StageStats("OpenCV Read (network fetch time)"),
        "yolo":          StageStats("YOLO Inference"),
        "mediapipe":     StageStats("MediaPipe Inference"),
        "total":         StageStats("Total Pipeline"),
        "display_lag":   StageStats("Estimated Display Lag"),
    }

    # Start capture
    capture = InstrumentedCapture(args.stream, maxsize=30)
    capture.start()

    # CSV
    csv_rows = []

    print(f"\n{COLORS['bold']}Starting diagnostic — sampling {args.frames} frames...{COLORS['reset']}")
    print(f"Stream: {args.stream}")
    print(f"Models: YOLO={'yes' if yolo_model else 'no'}, MediaPipe={'yes' if hands_tracker else 'no'}")
    print_header()

    frame_num = 0
    try:
        while frame_num < args.frames:
            frame, timer = capture.read()
            if frame is None:
                print("[WARN] Timeout waiting for frame — is the ESP32 streaming?")
                break

            q_depth = capture.queue_depth

            # Run models if available
            if yolo_model:
                run_yolo(yolo_model, frame, timer)
            if hands_tracker:
                run_mediapipe(hands_tracker, frame, timer)

            timer.mark("t_display_ready")

            # Build row
            row = timer_to_row(timer, q_depth, capture.avg_capture_interval)
            csv_rows.append(row)

            # Update stats
            stats["queue_wait"].add(row["queue_wait_ms"])
            stats["opencv_read"].add(row["opencv_read_ms"])
            stats["yolo"].add(row["yolo_ms"])
            stats["mediapipe"].add(row["mediapipe_ms"])
            stats["total"].add(row["total_pipeline_ms"])
            stats["display_lag"].add(row["estimated_display_lag_ms"])

            print_row(row, frame_num)
            frame_num += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        capture.stop()
        if hands_tracker:
            hands_tracker.close()

    if not csv_rows:
        print("[ERROR] No frames captured. Check stream URL and ESP32 connection.")
        return

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[INFO] Raw data saved → {args.out_csv}")

    # Write summary
    write_summary(stats, args.out_txt, args)

    # Quick capture FPS report
    if capture.capture_intervals:
        avg_interval = sum(capture.capture_intervals) / len(capture.capture_intervals)
        print(f"\n[STREAM] Avg frame interval from ESP32: {avg_interval:.1f}ms ({1000/avg_interval:.1f} FPS)")
        if stats["total"].avg > avg_interval:
            print(f"{COLORS['red']}[!] Pipeline ({stats['total'].avg:.0f}ms avg) is SLOWER than frame rate ({avg_interval:.0f}ms) — queue will grow over time{COLORS['reset']}")
        else:
            print(f"{COLORS['green']}[✓] Pipeline is keeping up with frame rate{COLORS['reset']}")


if __name__ == "__main__":
    main()