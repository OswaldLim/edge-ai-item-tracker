import cv2
import os


# -------------------------
# Configuration
# -------------------------

CAMERA_INDEX = 1

OUTPUT_DIR = "calibration_images"

# Number of INNER corners on your checkerboard
# Example:
# 10 squares x 7 squares = 9 x 6 inner corners
CHECKERBOARD_SIZE = (9, 6)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# -------------------------
# Setup
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not cap.isOpened():
    raise RuntimeError(f"Unable to open camera {CAMERA_INDEX}")


image_count = len(
    [
        file
        for file in os.listdir(OUTPUT_DIR)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)


print("Camera Calibration Image Capture")
print("--------------------------------")
print("SPACE : Save image")
print("Q     : Quit")
print()
print("Try to capture 15-30 images from different angles.")
print()


# -------------------------
# Capture loop
# -------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_SIZE,
        None,
    )

    if found:
        cv2.drawChessboardCorners(
            display_frame,
            CHECKERBOARD_SIZE,
            corners,
            found,
        )

        status_text = "Checkerboard detected - press SPACE"
    else:
        status_text = "Move checkerboard into view"

    cv2.putText(
        display_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if found else (0, 0, 255),
        2,
    )

    cv2.putText(
        display_frame,
        f"Images: {image_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Calibration Capture", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:  # SPACE
        if not found:
            print("Checkerboard not detected. Image not saved.")
            continue

        filename = os.path.join(
            OUTPUT_DIR,
            f"calibration_{image_count:03d}.jpg",
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

        image_count += 1


cap.release()
cv2.destroyAllWindows()

print(f"\nCaptured {image_count} calibration images.")