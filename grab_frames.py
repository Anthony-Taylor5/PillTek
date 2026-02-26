import cv2
import time
import os

#Replace with your ESP32 stream URL
STREAM_URL = "http://192.168.0.211:81/stream"        #"http://172.20.10.4:81/stream"  # Your hotspot IP

#Connect to stream
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Error: Cannot connect to stream")
    exit()

print("Connected to ESP32 stream!")

#Capture a few test frames
# Ensure images directory exists
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

for i in range(5):
    ret, frame = cap.read()
    if ret:
        # Save frame
        out_path = os.path.join(IMAGES_DIR, f"testframe{i}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved frame {i} -> {out_path}")
        time.sleep(2)
    else:
        print(f"Failed to capture frame {i}")

cap.release()
print("Test complete! Check your directory for testframe*.jpg files")


