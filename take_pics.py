

### **Step 4: Start Collecting Pill Bottle Images**

# collect_pill_bottle_images.py
import cv2
import os
import time
from datetime import datetime

STREAM_URL = "http://192.168.0.211:81/stream" #"http://172.20.10.4:81/stream"
cap = cv2.VideoCapture(STREAM_URL)

# print("Press 's' to save an image, 'q' to quit")
print("Press a for A, Press b for B, Press c for C, Press d for combined, Press e for Hazard, print f for hand")
A_IMAGES_DIR = "A images"
B_IMAGES_DIR = "B images"
C_IMAGES_DIR = "C images"
COMBINED_IMAGES_DIR = "combined images"
HAZARD_IMAGES_DIR = "hazard images"
HAND_IMAGES_DIR = "hand images"

os.makedirs(A_IMAGES_DIR, exist_ok=True)
os.makedirs(B_IMAGES_DIR, exist_ok=True)
os.makedirs(C_IMAGES_DIR, exist_ok=True)
os.makedirs(COMBINED_IMAGES_DIR, exist_ok=True)
os.makedirs(HAZARD_IMAGES_DIR, exist_ok=True)
os.makedirs(HAND_IMAGES_DIR, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show live feed
    cv2.imshow("Press a for A, Press b for B, Press c for C, Press d for combined, Press e for Hazard, press f for hand", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
   
    if key == ord('a'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pill~bottle-A_{timestamp}.jpg"
        out_path = os.path.join(A_IMAGES_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Saved: {filename}")

    elif key == ord('b'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pill~bottle-B_{timestamp}.jpg"
        out_path = os.path.join(B_IMAGES_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Saved: {filename}")
    
    elif key == ord('c'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pill~bottle-C_{timestamp}.jpg"
        out_path = os.path.join(C_IMAGES_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Saved: {filename}")
    
    elif key == ord('d'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pill~bottle-D_{timestamp}.jpg"
        out_path = os.path.join(COMBINED_IMAGES_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Saved: {filename}")
    
    elif key == ord('e'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pill~bottle-E_{timestamp}.jpg"
        out_path = os.path.join(HAZARD_IMAGES_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Saved: {filename}")
    
    elif key == ord('f'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pill~bottle-F_{timestamp}.jpg"
        out_path = os.path.join(HAND_IMAGES_DIR, filename)
        cv2.imwrite(out_path, frame)
        print(f"Saved: {filename}")


cap.release()
cv2.destroyAllWindows()


# **What to capture:**
# - Pill bottle in different positions
# - Different lighting conditions
# - With and without the bottle in frame
# - Different angles
# - Target: 50-100 images minimum
