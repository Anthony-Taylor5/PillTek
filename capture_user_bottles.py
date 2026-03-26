"""
capture_user_bottles.py

A guided image capture tool for collecting custom pill bottle training data,
followed by automatic fine-tuning of your YOLO model on the new images.

Usage:
    python capture_user_bottles.py --weights runs/detect/runs/train_v10/weights/best.pt

    #hotspot
    python capture_user_bottles.py --weights runs/detect/runs/train_v10/weights/best.pt --source http://192.168.0.211:81/stream 
   
    #home
    python capture_user_bottles.py --weights runs/detect/runs/train_v10/weights/best.pt --source http://192.168.0.31:81/stream


    python capture_user_bottles.py --weights runs/detect/runs/train_v10/weights/best.pt --epochs 30 --device gpu


Controls (shown on-screen):
    [1] Select SHORT bottle box
    [2] Select TALL  bottle box
    [3] Select WIDE  bottle box
    SPACE  Capture a photo
    Q      Quit early (skips training)

Output:
    /user_bottles/
        images/train/    <- captured images split 80% train
        images/val/      <- 20% validation
        labels/train/    <- YOLO annotation .txt files
        labels/val/
        data.yaml        <- dataset config pointing to above folders

    /runs/user_tuned_X/  <- incrementing run folder with fine-tuned weights

Requirements:
    pip install -U ultralytics opencv-python
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install -U ultralytics")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_CAPTURES   = 24
DEGREES_PER_SHOT = 15          # 24 × 15 = 360 degrees

# Bounding box definitions: (width, height) as fraction of the shorter display dim
BOX_PRESETS = {
    "short": {"label": "SHORT",  "w_frac": 0.22, "h_frac": 0.28, "color": (0, 220, 120)},
    "tall":  {"label": "TALL",   "w_frac": 0.18, "h_frac": 0.52, "color": (80, 160, 255)},
    "wide":  {"label": "WIDE",   "w_frac": 0.38, "h_frac": 0.22, "color": (255, 180, 40)},
}

# UI colours (BGR)
C_BG_PANEL  = (18,  18,  22)
C_WHITE     = (240, 240, 240)
C_MUTED     = (120, 120, 130)
C_ACCENT    = (60,  200, 140)
C_WARN      = (60,  120, 255)
C_SHUTTER   = (245, 245, 245)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Video capture (reused from test_with_hand_recognition.py)
# ---------------------------------------------------------------------------

import queue
import threading


def _create_cap(src):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(src, cv2.CAP_ANY)
    return cap


class ThreadedCapture:
    def __init__(self, src):
        self.src        = src
        self.cap        = _create_cap(src)
        self.stopped    = False
        self.frame      = None
        self.ret        = False
        self.fail_count = 0
        self.last_ok    = time.time()
        self._lock      = threading.Lock()
        self._thread    = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        time.sleep(1.5)

    def _reopen(self):
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(0.5)
        self.cap = _create_cap(self.src)
        if self.cap.isOpened():
            self.fail_count = 0
            self.last_ok    = time.time()

    def _update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._reopen()
                time.sleep(0.2)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret   = True
                    self.frame = frame
                self.fail_count = 0
                self.last_ok    = time.time()
            else:
                self.fail_count += 1
                if self.fail_count >= 30 or (time.time() - self.last_ok) > 2.0:
                    self._reopen()
                    self.fail_count = 0
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return self.ret, (None if self.frame is None else self.frame.copy())

    def release(self):
        self.stopped = True
        self._thread.join(timeout=1.5)
        try:
            self.cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def prepare_dataset_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "images_train": base / "images" / "train",
        "images_val":   base / "images" / "val",
        "labels_train": base / "labels" / "train",
        "labels_val":   base / "labels"  / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_data_yaml(base: Path, class_name: str = "user_bottle") -> Path:
    yaml_path = base / "data.yaml"
    content = (
        f"path: {base.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def yolo_label(cx_frac: float, cy_frac: float, w_frac: float, h_frac: float, cls: int = 0) -> str:
    return f"{cls} {cx_frac:.6f} {cy_frac:.6f} {w_frac:.6f} {h_frac:.6f}\n"


def save_capture(
    frame: np.ndarray,
    box_cx: int, box_cy: int, box_w: int, box_h: int,
    dirs: dict[str, Path],
    index: int,
    val_fraction: float = 0.20,
) -> None:
    """Save image and its YOLO label into train or val split."""
    is_val   = (index % round(1 / val_fraction)) == 0
    split    = "val" if is_val else "train"
    stem     = f"bottle_{index:03d}"
    img_path = dirs[f"images_{split}"] / f"{stem}.jpg"
    lbl_path = dirs[f"labels_{split}"]  / f"{stem}.txt"

    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    fh, fw = frame.shape[:2]
    cx_frac = box_cx / fw
    cy_frac = box_cy / fh
    w_frac  = box_w  / fw
    h_frac  = box_h  / fh
    lbl_path.write_text(yolo_label(cx_frac, cy_frac, w_frac, h_frac))


# ---------------------------------------------------------------------------
# Next available run folder
# ---------------------------------------------------------------------------

def next_run_dir(project: Path, prefix: str = "user_tuned_") -> Path:
    existing = sorted(project.glob(f"{prefix}*"))
    index    = len(existing) + 1
    d        = project / f"{prefix}{index}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=12):
    cv2.line(img,  (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img,  (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img,  (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img,  (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0,   0, 90,  color, thickness)


def draw_pill_button(canvas, x, y, w, h, label, key_hint, color, selected, hover=False):
    alpha   = 0.35 if selected else (0.18 if hover else 0.10)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    border_t = 2 if not selected else 3
    draw_rounded_rect(canvas, x, y, x + w, y + h, color, border_t, r=10)

    # Key hint badge
    badge_r = 14
    bx, by  = x + badge_r + 4, y + badge_r + 4
    cv2.circle(canvas, (bx, by), badge_r, color, -1)
    cv2.putText(canvas, key_hint, (bx - 5, by + 6), FONT_BOLD, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    cv2.putText(canvas, label, (x + 38, y + h // 2 + 6),
                FONT_BOLD, 0.6, C_WHITE if selected else color, 1, cv2.LINE_AA)

    if selected:
        cv2.putText(canvas, "SELECTED", (x + w - 90, y + h // 2 + 6),
                    FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_capture_button(canvas, cx, y, w, h, captures_done, active):
    x    = cx - w // 2
    color = C_ACCENT if active else C_MUTED
    alpha = 0.25 if active else 0.08
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    draw_rounded_rect(canvas, x, y, x + w, y + h, color, 2, r=12)

    label = f"[ SPACE ]  CAPTURE  ({captures_done}/{TOTAL_CAPTURES})"
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.65, 1)
    cv2.putText(canvas, label, (cx - tw // 2, y + h // 2 + th // 2),
                FONT_BOLD, 0.65, color, 1, cv2.LINE_AA)


def draw_progress_dots(canvas, x, y, done, total, color):
    dot_r   = 5
    spacing = 16
    for i in range(total):
        cx_dot = x + i * spacing
        filled = i < done
        cv2.circle(canvas, (cx_dot, y), dot_r,
                   color if filled else C_MUTED, -1 if filled else 1)


def draw_rotation_arc(canvas, cx, cy, radius, done, total, color):
    """Draw a circular arc showing how far around 360 degrees we are."""
    start_angle = -90
    sweep       = int(360 * done / total)
    # Background ring
    cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 0, 360, C_MUTED, 2)
    if sweep > 0:
        cv2.ellipse(canvas, (cx, cy), (radius, radius), start_angle, 0, sweep, color, 3)

    # Marker dot at current position
    angle_rad = math.radians(start_angle + sweep)
    mx = int(cx + radius * math.cos(angle_rad))
    my = int(cy + radius * math.sin(angle_rad))
    cv2.circle(canvas, (mx, my), 6, color, -1)

    text = f"{done * DEGREES_PER_SHOT}°"
    (tw, _), _ = cv2.getTextSize(text, FONT_BOLD, 0.55, 1)
    cv2.putText(canvas, text, (cx - tw // 2, cy + 7), FONT_BOLD, 0.55, color, 1, cv2.LINE_AA)


def draw_bounding_box_overlay(canvas, box_x1, box_y1, box_x2, box_y2, color, captures_done):
    """Draw the guide box with corner brackets and dim outside the box."""
    h, w = canvas.shape[:2]

    # Subtle dim outside the guide box
    dim = np.zeros_like(canvas, dtype=np.uint8)
    dim[:, :] = (0, 0, 0)
    mask = np.ones((h, w), dtype=np.uint8) * 80
    mask[box_y1:box_y2, box_x1:box_x2] = 0
    dim_layer = canvas.copy()
    cv2.addWeighted(dim, 0, canvas, 1, 0, dim_layer)
    overlay = canvas.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.45).astype(np.uint8)
    canvas[:] = overlay

    # Corner bracket length
    blen = min((box_x2 - box_x1), (box_y2 - box_y1)) // 5
    thick = 3
    corners = [
        (box_x1, box_y1, 1,  1),
        (box_x2, box_y1, -1, 1),
        (box_x1, box_y2, 1, -1),
        (box_x2, box_y2, -1, -1),
    ]
    for (cx, cy, dx, dy) in corners:
        cv2.line(canvas, (cx, cy), (cx + dx * blen, cy),          color, thick)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * blen),           color, thick)

    # Crosshair at center
    ccx = (box_x1 + box_x2) // 2
    ccy = (box_y1 + box_y2) // 2
    ch  = 10
    cv2.line(canvas, (ccx - ch, ccy), (ccx + ch, ccy), color, 1)
    cv2.line(canvas, (ccx, ccy - ch), (ccx, ccy + ch), color, 1)


def draw_instruction(canvas, captures_done, selected_box):
    h, w = canvas.shape[:2]
    if selected_box is None:
        msg  = "Select a bottle size above, then press SPACE to capture."
        color = C_WARN
    elif captures_done == 0:
        msg  = "Place your bottle inside the guide box and press SPACE."
        color = C_ACCENT
    elif captures_done < TOTAL_CAPTURES:
        deg  = captures_done * DEGREES_PER_SHOT
        msg  = f"Rotate bottle {DEGREES_PER_SHOT} degrees clockwise (now at {deg}°) and press SPACE."
        color = C_ACCENT
    else:
        msg  = "All captures done! Training will begin shortly..."
        color = (80, 220, 180)

    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.60, 1)
    px, py = (w - tw) // 2, h - 30
    # subtle shadow
    cv2.putText(canvas, msg, (px + 1, py + 1), FONT, 0.60, (0, 0, 0),     2, cv2.LINE_AA)
    cv2.putText(canvas, msg, (px,     py),      FONT, 0.60, color,          1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main capture session
# ---------------------------------------------------------------------------

def run_capture(source, dataset_base: Path, class_name: str = "test_user_bottle_1") -> tuple[dict[str, Path], Path, dict]:
    """
    Opens the stream, runs the interactive capture UI, and returns
    (dirs, data_yaml_path, box_info_of_last_selection).
    """
    src = int(source) if str(source).isdigit() else source
    cap = ThreadedCapture(src)

    dirs      = prepare_dataset_dirs(dataset_base)
    data_yaml = write_data_yaml(dataset_base, class_name=class_name)

    selected_key: str | None = None
    captures_done            = 0
    shutter_frames           = 0     # countdown for white-flash effect
    SHUTTER_DURATION         = 6     # frames to show flash

    WIN = "Pill Bottle Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Mouse hover tracking for button highlight (optional polish)
    mouse_pos = [0, 0]
    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
    cv2.setMouseCallback(WIN, on_mouse)

    last_frame  = None
    no_frame_warned = False
    last_box    = {}  # will hold {x1,y1,x2,y2,w,h,cx,cy} of selected box

    while captures_done < TOTAL_CAPTURES:
        ok, frame = cap.read()
        if not ok or frame is None:
            if not no_frame_warned:
                print("[WARN] Waiting for frames...")
                no_frame_warned = True
            time.sleep(0.05)
            continue
        no_frame_warned = False
        last_frame = frame

        fh, fw = frame.shape[:2]

        # Build side panel dimensions
        PANEL_W   = 260
        canvas_w  = fw + PANEL_W
        canvas    = np.zeros((fh, canvas_w, 3), dtype=np.uint8)
        canvas[:, :fw] = frame

        # Fill panel background
        cv2.rectangle(canvas, (fw, 0), (canvas_w, fh), C_BG_PANEL, -1)

        # Compute box geometry for the selected preset
        if selected_key and selected_key in BOX_PRESETS:
            p      = BOX_PRESETS[selected_key]
            box_w  = int(fw * p["w_frac"])
            box_h  = int(fh * p["h_frac"])
            box_cx = fw // 2
            box_cy = fh // 2
            box_x1 = box_cx - box_w // 2
            box_y1 = box_cy - box_h // 2
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h
            box_color = p["color"]
            last_box = dict(x1=box_x1, y1=box_y1, x2=box_x2, y2=box_y2,
                            w=box_w, h=box_h, cx=box_cx, cy=box_cy)
        else:
            box_color = C_MUTED
            last_box = {}

        # Draw bounding guide box over live feed
        if last_box:
            draw_bounding_box_overlay(
                canvas, last_box["x1"], last_box["y1"],
                last_box["x2"], last_box["y2"], box_color, captures_done
            )

        # ---- Side panel UI ------------------------------------------------
        px = fw + 16
        py = 22

        # Title
        cv2.putText(canvas, "BOTTLE", (px, py + 16), FONT_BOLD, 0.75, C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "CAPTURE", (px, py + 40), FONT_BOLD, 0.75, C_WHITE, 1, cv2.LINE_AA)
        py += 62
        cv2.line(canvas, (px, py), (canvas_w - 16, py), C_MUTED, 1)
        py += 16

        # Size selector buttons
        btn_w = PANEL_W - 32
        btn_h = 44
        btn_gap = 10
        button_keys = ["short", "tall", "wide"]
        key_hints   = ["1", "2", "3"]
        button_rects = {}

        for i, (bk, kh) in enumerate(zip(button_keys, key_hints)):
            bx = px
            by = py + i * (btn_h + btn_gap)
            sel = (selected_key == bk)
            mx, my = mouse_pos
            hover = (bx <= mx <= bx + btn_w) and (by <= my <= by + btn_h)
            draw_pill_button(canvas, bx, by, btn_w, btn_h,
                             BOX_PRESETS[bk]["label"], kh,
                             BOX_PRESETS[bk]["color"], sel, hover)
            button_rects[bk] = (bx, by, bx + btn_w, by + btn_h)

        py += 3 * (btn_h + btn_gap) + 8
        cv2.line(canvas, (px, py), (canvas_w - 16, py), C_MUTED, 1)
        py += 18

        # Progress ring
        ring_cx = fw + PANEL_W // 2
        ring_cy = py + 52
        ring_r  = 46
        ring_color = box_color if selected_key else C_MUTED
        draw_rotation_arc(canvas, ring_cx, ring_cy, ring_r, captures_done, TOTAL_CAPTURES, ring_color)
        py = ring_cy + ring_r + 18

        # Progress label
        prog_text = f"{captures_done} / {TOTAL_CAPTURES} photos"
        (tw, _), _ = cv2.getTextSize(prog_text, FONT, 0.52, 1)
        cv2.putText(canvas, prog_text,
                    (ring_cx - tw // 2, py), FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)
        py += 18

        # Dot row
        dots_total_w = TOTAL_CAPTURES * 16
        dots_x0      = ring_cx - dots_total_w // 2
        draw_progress_dots(canvas, dots_x0, py, captures_done, TOTAL_CAPTURES, ring_color)
        py += 22

        cv2.line(canvas, (px, py), (canvas_w - 16, py), C_MUTED, 1)
        py += 14

        # Capture button
        can_capture = (selected_key is not None) and (captures_done < TOTAL_CAPTURES)
        cap_btn_w   = PANEL_W - 24
        cap_btn_h   = 46
        draw_capture_button(canvas, ring_cx, py, cap_btn_w, cap_btn_h, captures_done, can_capture)
        py += cap_btn_h + 14

        # Quit hint
        cv2.putText(canvas, "[ Q ] quit early", (px + 12, py + 14),
                    FONT, 0.42, C_MUTED, 1, cv2.LINE_AA)

        # Instruction bar across bottom of the feed area
        draw_instruction(canvas, captures_done, selected_key)

        # Shutter flash effect
        if shutter_frames > 0:
            alpha_f = shutter_frames / SHUTTER_DURATION
            white   = np.full_like(canvas, 245, dtype=np.uint8)
            cv2.addWeighted(white, alpha_f, canvas, 1 - alpha_f, 0, canvas)
            shutter_frames -= 1

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF

        # Key handling
        if key == ord("q") or key == ord("Q"):
            print("[INFO] Quit early. Captured images will still be saved.")
            break

        if key == ord("1"):
            selected_key = "short"
        elif key == ord("2"):
            selected_key = "tall"
        elif key == ord("3"):
            selected_key = "wide"
        elif key == ord(" ") and can_capture and last_frame is not None:
            # Trigger shutter
            shutter_frames = SHUTTER_DURATION

            # Crop to the guide box from the raw frame (no overlays)
            raw   = last_frame
            rfh, rfw = raw.shape[:2]

            # Recompute box on raw frame coordinates
            p      = BOX_PRESETS[selected_key]
            bw_raw = int(rfw * p["w_frac"])
            bh_raw = int(rfh * p["h_frac"])
            bx1    = rfw // 2 - bw_raw // 2
            by1    = rfh // 2 - bh_raw // 2
            bx2    = bx1 + bw_raw
            by2    = by1 + bh_raw

            # Clamp to frame
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(rfw, bx2); by2 = min(rfh, by2)


            save_capture(
                frame=raw,           # save full frame with label centred on the box
                box_cx=rfw // 2, box_cy=rfh // 2,
                box_w=bw_raw, box_h=bh_raw,
                dirs=dirs,
                index=captures_done,
            )

            captures_done += 1
            deg = (captures_done) * DEGREES_PER_SHOT
            print(f"[CAPTURE] {captures_done}/{TOTAL_CAPTURES} saved. "
                  f"Next: rotate {DEGREES_PER_SHOT}° clockwise ({deg}° total).")

        # Mouse click: size buttons
        if key == 255:  # no key, check mouse click via flag trick -- use flag param instead
            pass

    cv2.destroyAllWindows()
    cap.release()

    return dirs, data_yaml, last_box


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(
    base_weights: str,
    data_yaml: Path,
    run_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Path:
    resolved_device = "0" if device == "gpu" else "cpu"
    print(f"\n[TRAIN] Starting fine-tune on user_bottles dataset.")
    print(f"[TRAIN] Base weights : {base_weights}")
    print(f"[TRAIN] Output dir   : {run_dir}")
    print(f"[TRAIN] Device       : {device} -> {resolved_device}")

    model   = YOLO(base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else run_dir
    best_pt  = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training completed but best.pt not found at: {best_pt}")

    print(f"[TRAIN] Done. Best weights saved to: {best_pt}")
    return best_pt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture custom pill bottle images and fine-tune your YOLO model."
    )
    parser.add_argument(
        "--class-name",
        default="test_user_bottle_1",
        help="Class name for this bottle (e.g. john_tylenol_500mg)",
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/runs/train_v10/weights/best.pt",
        help="Base model weights to fine-tune from",
    )
    parser.add_argument(
        "--source",
        default="http://192.168.0.31:81/stream",
        help="Video source: ESP32 stream URL or webcam index (0, 1, 2...)",
    )
    parser.add_argument("--epochs",    type=int, default=30,      help="Fine-tune epochs (default: 30)")
    parser.add_argument("--imgsz",     type=int, default=640,     help="Image size (default: 640)")
    parser.add_argument("--batch",     type=int, default=8,       help="Batch size (default: 8)")
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="cpu",
        help="Device: gpu or cpu (default: cpu)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="user_bottles",
        help="Root folder for captured dataset (default: user_bottles/)",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root folder for training runs (default: runs/)",
    )
    args = parser.parse_args()

    cwd         = Path.cwd()
    dataset_dir = cwd / args.dataset_dir / args.class_name
    runs_dir    = cwd / args.runs_dir

    # Validate base weights
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = cwd / weights_path
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        return 2

    print("=" * 60)
    print("  Pill Bottle Capture Tool")
    print("=" * 60)
    print(f"  Base model : {weights_path}")
    print(f"  Stream     : {args.source}")
    print(f"  Dataset    : {dataset_dir}")
    print(f"  Photos     : {TOTAL_CAPTURES} (every {DEGREES_PER_SHOT}°)")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    [1] Short bottle box")
    print("    [2] Tall  bottle box")
    print("    [3] Wide  bottle box")
    print("    [SPACE] Capture photo")
    print("    [Q] Quit")
    print()

    # Run capture session
    dirs, data_yaml, _ = run_capture(source=args.source, dataset_base=dataset_dir, class_name=args.class_name)

    # Count how many images were actually captured
    total_saved = sum(1 for _ in (dataset_dir / "images").rglob("*.jpg"))
    if total_saved == 0:
        print("[WARN] No images were captured. Skipping training.")
        return 0

    print(f"\n[INFO] Captured {total_saved} images total.")
    print(f"[INFO] Dataset written to: {dataset_dir}")
    print(f"[INFO] data.yaml: {data_yaml}")

    # Find the next available user_tuned_X run folder
    run_dir = next_run_dir(runs_dir, prefix="user_tuned_")
    print(f"[INFO] Training output: {run_dir}")

    best_pt = run_training(
        base_weights=str(weights_path),
        data_yaml=data_yaml,
        run_dir=run_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print()
    print("=" * 60)
    print("  Done!")
    print(f"  Fine-tuned model: {best_pt}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())