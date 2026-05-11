import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

image_path = "bus_output.jpg"
frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError("Could not read output.jpg. Make sure output.jpg exists in the same folder.")

results = model(frame)

danger_zone = (600, 250, 1050, 700)
alert = False

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    if class_name == "person":
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        dz_x1, dz_y1, dz_x2, dz_y2 = danger_zone

        if dz_x1 <= person_center_x <= dz_x2 and dz_y1 <= person_center_y <= dz_y2:
            alert = True
            cv2.putText(
                frame,
                "ALERT: Person in restricted area",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

cv2.rectangle(
    frame,
    (danger_zone[0], danger_zone[1]),
    (danger_zone[2], danger_zone[3]),
    (0, 0, 255),
    3
)

cv2.imwrite("restricted_area_output.jpg", frame)
cv2.imshow("Restricted Area Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Alert:", alert)
print("Saved as restricted_area_output.jpg")