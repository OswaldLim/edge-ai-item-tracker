import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    results = model(
        frame,
        conf=0.5,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    # Read each detected object
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0],
        )

        class_name = model.names[class_id]

        print(
            f"{class_name}: "
            f"{confidence:.2f} "
            f"({x1}, {y1}, {x2}, {y2})"
        )

    annotated_frame = result.plot()

    cv2.imshow(
        "Spatial AI - YOLO11n",
        annotated_frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()