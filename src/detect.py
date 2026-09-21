import cv2
import sqlite3
from datetime import datetime
from ultralytics import YOLO


# -------------------------
# Configuration
# -------------------------

MODEL_PATH = "yolo26n.pt"
CAMERA_INDEX = 1
DETECTION_INTERVAL = 5
CONFIDENCE_THRESHOLD = 0.5
DATABASE_PATH = "detections.db"

MAX_ENTRIES_PER_ITEM = 5


# -------------------------
# SQLite setup
# -------------------------

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL
)
""")

conn.commit()


# -------------------------
# YOLO setup
# -------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

frame_count = 0
annotated_frame = None


# -------------------------
# Main loop
# -------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    frame_count += 1

    # Run YOLO only every 5 frames
    if frame_count % DETECTION_INTERVAL == 0:

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        timestamp = datetime.now().isoformat(
            sep=" ",
            timespec="milliseconds",
        )

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
                f"[{timestamp}] "
                f"{class_name}: "
                f"{confidence:.2f} "
                f"({x1}, {y1}, {x2}, {y2})"
            )

            # Save detection to SQLite
            cursor.execute(
                """
                INSERT INTO detections (
                    timestamp,
                    frame_number,
                    class_id,
                    class_name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    frame_count,
                    class_id,
                    class_name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            )

            # Keep only the latest 5 detections
            # for this particular class
            cursor.execute(
                """
                DELETE FROM detections
                WHERE class_name = ?
                AND id NOT IN (
                    SELECT id
                    FROM detections
                    WHERE class_name = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (
                    class_name,
                    class_name,
                    MAX_ENTRIES_PER_ITEM,
                ),
            )

        conn.commit()

        # Generate annotated frame
        annotated_frame = result.plot()

    # Show latest YOLO result between detection frames
    if annotated_frame is not None:
        cv2.imshow(
            "Spatial AI - YOLO26n",
            annotated_frame,
        )
    else:
        cv2.imshow(
            "Spatial AI - YOLO26n",
            frame,
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()
conn.close()