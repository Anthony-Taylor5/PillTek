import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("http://172.20.10.4:81/stream")

# Load pretrained model
model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Motion Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()