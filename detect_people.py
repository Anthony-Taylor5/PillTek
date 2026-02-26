import cv2
from ultralytics import YOLO
import time
import threading

# Threaded video capture for smoother FPS
class ThreadedVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print("Cannot connect to stream!")
            exit()
        self.ret = False
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# Initialize model with robust stream handling and warm-up
print("Loading model...")
model = YOLO('yolov5s.pt')
print("Model loaded!")

STREAM_URL = "http://192.168.0.211:81/stream" #'http://172.20.10.4:81/stream' 

# --- Robust capture creator: tries FFMPEG first, then ANY ---
def create_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url, cv2.CAP_ANY)
    return cap

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

cap = ThreadedVideoCapture(STREAM_URL)

frame_count = 0
last_detection_time = 0
fps_time = time.time()
fps_counter = 0
fps_display = 0
person_detected = False

# Optional ROI mask ratio to ignore top region (e.g., lights)
TOP_MASK_RATIO = 0.0 #.15  # 15% of frame height

# Dynamic detection interval (seconds)
TARGET_HZ = 4.0  # run detection ~4 times per second
detect_interval = 1.0 / TARGET_HZ

no_frame_warned = False

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        # Don't exit immediately—ESP32 streams hiccup. Keep trying.
        if not no_frame_warned:
            print("Waiting for stream frames...")
            no_frame_warned = True
        time.sleep(0.05)
        continue
    no_frame_warned = False

    display_frame = frame.copy()
    h, w = frame.shape[:2]

    # Apply ROI mask to ignore lights in top portion
    if 0.0 < TOP_MASK_RATIO <= 0.5:
        mask_height = int(h * TOP_MASK_RATIO)
        frame[:mask_height, :] = 0

    now = time.time()
    # Run inference at fixed time interval instead of every N frames
    if now - last_detection_time >= detect_interval:
        last_detection_time = now

        results = model(frame, classes=[0], imgsz=512, conf=0.15, iou=0.6, verbose=False)

        if len(results[0].boxes) > 0:
            person_detected = True
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            person_detected = False

    # Calculate and display FPS
    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        fps_display = fps_counter / (time.time() - fps_time)
        fps_counter = 0
        fps_time = time.time()

    status_text = "PERSON DETECTED" if person_detected else "No Person"
    status_color = (0, 255, 0) if person_detected else (0, 0, 255)
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Person Detection (Optimized)', display_frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped")