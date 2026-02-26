
# motion_detection.py
import cv2
import numpy as np

STREAM_URL = "http://192.168.0.211:81/stream"                    #"http://172.20.10.4:81/stream"
cap = cv2.VideoCapture(STREAM_URL)

# Read first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Define ROI (Region of Interest) - adjust coordinates for your medication area
# Format: [y1:y2, x1:x2]
roi_y1, roi_y2 = 100, 400  # Adjust these
roi_x1, roi_x2 = 150, 500  # Adjust these

print("Running motion detection. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (240, 240))

    
    # Extract ROI
    roi_prev = prev_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_curr = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Calculate difference
    frame_diff = cv2.absdiff(roi_prev, roi_curr)
    
    # Threshold
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Count motion pixels
    motion_pixels = np.sum(thresh > 0)
    motion_percent = (motion_pixels / thresh.size) * 100
    
    # Draw ROI rectangle on frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Display motion status
    if motion_percent > 2:  # Adjust threshold as needed
        cv2.putText(frame, f"MOTION: {motion_percent:.1f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Motion detected: {motion_percent:.1f}%")
    else:
        cv2.putText(frame, "No Motion", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Motion Detection', frame)
    
    # Update previous frame
    prev_gray = gray.copy()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
## Recommended Order of Tasks

**Week 1-2: Backend Setup & Person Detection**
- [ ] Set up Python environment
- [ ] Test stream capture
- [ ] Integrate pretrained YOLO
- [ ] Verify person detection works

**Week 3-4: Data Collection**
- [ ] Collect 50-100 pill bottle images
- [ ] Set up medication area with consistent background
- [ ] Capture various scenarios

**Week 5-6: Motion Detection**
- [ ] Implement frame differencing
- [ ] Tune ROI and thresholds
- [ ] Test with actual hand/arm movements

**Later (Spring 2026): Custom Model Training**
- [ ] Label collected images
- [ ] Train custom YOLOv5 for pill bottles
- [ ] Integrate trained model

## Quick Troubleshooting Tips

**If stream connection fails:**
- Check ESP32 IP address: `http://192.168.x.x` in browser
- Verify you're on same hotspot network
- Try increasing timeout: `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`

**If YOLO is slow:**
- Use `yolov5n` (nano) instead of `yolov5s` for faster inference
- Process fewer frames (every 60th instead of 30th)

Which step would you like to tackle first? I can provide more detailed code for whichever component you want to work on next!
message.txt
7 KB
'''