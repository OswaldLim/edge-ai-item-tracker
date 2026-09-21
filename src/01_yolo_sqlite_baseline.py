import cv2
import sqlite3
import uuid
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "yolo26n.pt"

CAMERA_INDEX = 1

# Run detection every N frames.
DETECTION_INTERVAL = 5

CONFIDENCE_THRESHOLD = 0.5

# Use a new DB so your old detections.db remains untouched.
DATABASE_PATH = "spatial_memory.db"

# For now, keep None.
# After training your 5-class model, you can optionally use:
#
# TARGET_CLASSES = {
#     "wallet",
#     "keys",
#     "bottle",
#     "remote",
#     "backpack",
# }
#
TARGET_CLASSES = None


# ============================================================
# Helper functions
# ============================================================

def current_timestamp():
    return datetime.now().isoformat(
        sep=" ",
        timespec="milliseconds",
    )


def setup_database(database_path):
    conn = sqlite3.connect(database_path)

    # Enable FK checking.
    conn.execute("PRAGMA foreign_keys = ON")

    # Better behaviour for a DB being written continuously.
    conn.execute("PRAGMA journal_mode = WAL")

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Session table
    #
    # One row = one program run.
    # This becomes useful later when testing cross-session SLAM.
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT
        )
        """
    )

    # --------------------------------------------------------
    # Objects table
    #
    # Eventually:
    #
    # wallet_001
    # wallet_002
    # bottle_001
    #
    # No objects are inserted yet because instance recognition
    # has not been implemented.
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS objects (
            instance_id TEXT PRIMARY KEY,

            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,

            created_at TEXT NOT NULL,

            thumbnail_path TEXT
        )
        """
    )

    # --------------------------------------------------------
    # Observations table
    #
    # One row = one observation of an object.
    #
    # Some columns are intentionally NULL for now.
    #
    # They will be filled later when depth, SLAM and instance
    # recognition are added.
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id TEXT NOT NULL,

            timestamp TEXT NOT NULL,
            frame_number INTEGER NOT NULL,

            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,

            confidence REAL NOT NULL,

            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL,

            frame_width INTEGER NOT NULL,
            frame_height INTEGER NOT NULL,

            -- Future instance recognition
            instance_id TEXT,

            -- Future depth
            depth_m REAL,

            -- Future 3D position relative to camera
            camera_x REAL,
            camera_y REAL,
            camera_z REAL,

            -- Future persistent SLAM/map coordinates
            map_x REAL,
            map_y REAL,
            map_z REAL,

            FOREIGN KEY(session_id)
                REFERENCES sessions(session_id),

            FOREIGN KEY(instance_id)
                REFERENCES objects(instance_id)
        )
        """
    )

    # Helpful indexes once the database gets larger.

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_observations_class_name
        ON observations(class_name)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_observations_instance_id
        ON observations(instance_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_observations_timestamp
        ON observations(timestamp)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_observations_session
        ON observations(session_id)
        """
    )

    conn.commit()

    return conn, cursor


def save_observation(
    cursor,
    session_id,
    timestamp,
    frame_number,
    class_id,
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height,
):
    """
    Save the information currently available.

    instance_id, depth and SLAM coordinates remain NULL
    until those components are implemented.
    """

    cursor.execute(
        """
        INSERT INTO observations (
            session_id,
            timestamp,
            frame_number,

            class_id,
            class_name,
            confidence,

            x1,
            y1,
            x2,
            y2,

            frame_width,
            frame_height,

            instance_id,
            depth_m,

            camera_x,
            camera_y,
            camera_z,

            map_x,
            map_y,
            map_z
        )
        VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            NULL, NULL,
            NULL, NULL, NULL,
            NULL, NULL, NULL
        )
        """,
        (
            session_id,
            timestamp,
            frame_number,

            class_id,
            class_name,
            confidence,

            x1,
            y1,
            x2,
            y2,

            frame_width,
            frame_height,
        ),
    )


# ============================================================
# Database setup
# ============================================================

conn, cursor = setup_database(DATABASE_PATH)

session_id = uuid.uuid4().hex

session_start = current_timestamp()

cursor.execute(
    """
    INSERT INTO sessions (
        session_id,
        started_at
    )
    VALUES (?, ?)
    """,
    (
        session_id,
        session_start,
    ),
)

conn.commit()

print("=" * 60)
print("Spatial Memory Prototype")
print(f"Session ID: {session_id}")
print(f"Started:    {session_start}")
print("=" * 60)


# ============================================================
# YOLO setup
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# Camera setup
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Could not open camera index {CAMERA_INDEX}")
    conn.close()
    raise SystemExit


# ============================================================
# Main loop
# ============================================================

frame_count = 0

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera")
            break

        frame_count += 1

        frame_height, frame_width = frame.shape[:2]

        # By default show the CURRENT camera frame.
        #
        # This fixes the issue in the original code where
        # annotated_frame caused an old frame to remain visible
        # between YOLO inference frames.
        display_frame = frame.copy()

        # ----------------------------------------------------
        # Run YOLO only every N frames
        # ----------------------------------------------------

        if frame_count % DETECTION_INTERVAL == 0:

            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            timestamp = current_timestamp()

            # ------------------------------------------------
            # Process detected objects
            # ------------------------------------------------

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0],
                    )

                    class_name = model.names[
                        class_id
                    ]

                    # Optional filter once your custom
                    # five-class model is ready.
                    if (
                        TARGET_CLASSES is not None
                        and
                        class_name not in TARGET_CLASSES
                    ):
                        continue

                    print(
                        f"[{timestamp}] "
                        f"{class_name} "
                        f"conf={confidence:.2f} "
                        f"bbox=({x1}, {y1}, {x2}, {y2})"
                    )

                    # ----------------------------------------
                    # Save observation
                    # ----------------------------------------

                    save_observation(
                        cursor=cursor,

                        session_id=session_id,

                        timestamp=timestamp,
                        frame_number=frame_count,

                        class_id=class_id,
                        class_name=class_name,

                        confidence=confidence,

                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,

                        frame_width=frame_width,
                        frame_height=frame_height,
                    )

            conn.commit()

            # Only this frame receives YOLO annotations.
            display_frame = result.plot()

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Spatial AI - YOLO26n",
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# Cleanup
# ============================================================

finally:

    session_end = current_timestamp()

    cursor.execute(
        """
        UPDATE sessions
        SET ended_at = ?
        WHERE session_id = ?
        """,
        (
            session_end,
            session_id,
        ),
    )

    conn.commit()

    cap.release()

    cv2.destroyAllWindows()

    conn.close()

    print("=" * 60)
    print(f"Session ended: {session_end}")
    print("=" * 60)