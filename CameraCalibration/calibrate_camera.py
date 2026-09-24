import cv2
import glob
import os
import numpy as np


# -------------------------
# Configuration
# -------------------------

CHECKERBOARD_SIZE = (9, 6)

# Physical size of each checkerboard square.
#
# Use metres if you want measurements in metres.
# Example:
# 25 mm = 0.025 m
#
# For just intrinsic calibration, the exact unit does not matter,
# but it becomes important if you later estimate real-world distances.
SQUARE_SIZE = 0.025

IMAGE_DIR = "calibration_images"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "calibration.npz",
)


# -------------------------
# Checkerboard coordinates
# -------------------------

object_point_template = np.zeros(
    (
        CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1],
        3,
    ),
    dtype=np.float32,
)

object_point_template[:, :2] = (
    np.mgrid[
        0 : CHECKERBOARD_SIZE[0],
        0 : CHECKERBOARD_SIZE[1],
    ]
    .T.reshape(-1, 2)
)

object_point_template *= SQUARE_SIZE


# -------------------------
# Calibration data
# -------------------------

object_points = []
image_points = []

images = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.*",
        )
    )
)

if not images:
    raise RuntimeError(
        f"No calibration images found in '{IMAGE_DIR}'."
    )


image_size = None

valid_images = []


# More precise corner localisation
criteria = (
    cv2.TERM_CRITERIA_EPS
    + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


# -------------------------
# Detect checkerboards
# -------------------------

for image_path in images:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to read: {image_path}")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_SIZE,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
        ),
    )

    if not found:
        print(
            f"[SKIPPED] Checkerboard not found: "
            f"{os.path.basename(image_path)}"
        )
        continue

    refined_corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )

    object_points.append(
        object_point_template.copy()
    )

    image_points.append(
        refined_corners
    )

    valid_images.append(
        image_path
    )

    print(
        f"[OK] {os.path.basename(image_path)}"
    )


# -------------------------
# Validate dataset
# -------------------------

if len(object_points) < 10:
    raise RuntimeError(
        "Not enough valid calibration images. "
        f"Only {len(object_points)} detected. "
        "Capture at least 15-20."
    )


# -------------------------
# Run camera calibration
# -------------------------

rms_error, camera_matrix, distortion_coefficients, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
)


# -------------------------
# Calculate reprojection error
# -------------------------

total_error = 0.0
per_image_errors = []

for i in range(len(object_points)):
    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients,
    )

    # Convert both into simple Nx2 arrays
    detected = image_points[i].reshape(-1, 2).astype(np.float64)
    projected = projected_points.reshape(-1, 2).astype(np.float64)

    # Pixel distance for every checkerboard corner
    errors = np.linalg.norm(
        detected - projected,
        axis=1,
    )

    mean_error = np.mean(errors)

    per_image_errors.append(mean_error)
    total_error += mean_error


mean_reprojection_error = (
    total_error / len(object_points)
)


# -------------------------
# Save calibration
# -------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    image_width=image_size[0],
    image_height=image_size[1],
    checkerboard_width=CHECKERBOARD_SIZE[0],
    checkerboard_height=CHECKERBOARD_SIZE[1],
    square_size=SQUARE_SIZE,
    rms_error=rms_error,
    mean_reprojection_error=mean_reprojection_error,
)


# -------------------------
# Output results
# -------------------------

print()
print("=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)

print(
    f"Valid images: {len(valid_images)} / {len(images)}"
)

print(
    f"Image size: {image_size}"
)

print()
print("Camera matrix:")
print(camera_matrix)

print()
print("Distortion coefficients:")
print(distortion_coefficients)

print()
print(
    f"RMS calibration error: {rms_error:.4f}"
)

print(
    "Mean reprojection error: "
    f"{mean_reprojection_error:.4f} pixels"
)

print()
print(
    f"Per-image reprojection errors: {per_image_errors}"
)

print()
print(
    f"Calibration saved to: {OUTPUT_FILE}"
)