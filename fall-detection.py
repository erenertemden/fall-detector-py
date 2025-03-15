import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("fall_video.mp4")

if not cap.isOpened():
    print("❌ Could not opened.")
    exit()

previous_positions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    current_positions = {}
    for idx, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if results.names[cls] == "person" and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            current_positions[idx] = (center_x, center_y, y2 - y1)

            # fall control
            if idx in previous_positions:
                prev_center_x, prev_center_y, prev_height = previous_positions[idx]

                vertical_speed = center_y - prev_center_y
                height_change = abs((y2 - y1) - prev_height)

                # fall detection levels
                if vertical_speed > 20 and height_change > 15:
                    print("🚨 FALL DETECTED!")
                    cv2.putText(frame, "FALL DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    previous_positions = current_positions.copy()

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
