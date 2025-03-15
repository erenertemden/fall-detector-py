import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("Untitled.mp4")

if not cap.isOpened():
    print("❌ Error opening video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if results.names[cls] != "person" or conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width, height = x2 - x1, y2 - y1

        # Simple aspect ratio check for fall detection
        aspect_ratio = width / height
        if aspect_ratio > 1.2: # adjust this threshold based on your video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            cv2.putText(frame, "FALL", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
