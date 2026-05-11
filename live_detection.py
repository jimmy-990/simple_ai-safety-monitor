import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

danger_zone = (300, 80, 630, 470)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    alert = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name == "teddy bear":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2

            dz_x1, dz_y1, dz_x2, dz_y2 = danger_zone

            if dz_x1 <= person_center_x <= dz_x2 and dz_y1 <= person_center_y <= dz_y2:
                alert = True

   
    cv2.rectangle(
        frame,
        (danger_zone[0], danger_zone[1]),
        (danger_zone[2], danger_zone[3]),
        (0, 0, 255),
        3
    )

    if alert:
        cv2.putText(
            frame,
            "ALERT: Restricted Area Access",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("AI Safety Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()