import cv2
import numpy as np

CAMERA_INDEX = 1
CALIBRATION_FILE = "output/calibration.npz"

data = np.load(CALIBRATION_FILE)

camera_matrix = data["camera_matrix"]
dist_coeffs = data["distortion_coefficients"]

calib_width = int(data["image_width"])
calib_height = int(data["image_height"])

print("Calibration resolution:")
print(calib_width, calib_height)

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_height)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    print_once = False

    # Only use calibration directly if resolutions match
    if width != calib_width or height != calib_height:
        print(
            f"WARNING: Camera is {width}x{height}, "
            f"but calibration is {calib_width}x{calib_height}"
        )

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (calib_width, calib_height),
        1,  # alpha=1 keeps all pixels
        (calib_width, calib_height),
    )

    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
    )

    original_small = cv2.resize(
        frame,
        (width // 2, height // 2)
    )

    undistorted_small = cv2.resize(
        undistorted,
        (width // 2, height // 2)
    )

    comparison = np.hstack(
        [original_small, undistorted_small]
    )

    cv2.putText(
        comparison,
        "Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        comparison,
        "Undistorted",
        (width // 2 + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Calibration Test", comparison)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()