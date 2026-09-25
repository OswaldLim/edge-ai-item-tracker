import numpy as np


NPZ_FILE = "output/calibration.npz"
OUTPUT_FILE = "MyCamera.yaml"
FPS = 30.0


data = np.load(NPZ_FILE)

K = data["camera_matrix"]
D = data["distortion_coefficients"].flatten()

width = int(data["image_width"])
height = int(data["image_height"])

fx = float(K[0, 0])
fy = float(K[1, 1])
cx = float(K[0, 2])
cy = float(K[1, 2])

k1 = float(D[0])
k2 = float(D[1])
p1 = float(D[2])
p2 = float(D[3])
k3 = float(D[4]) if len(D) > 4 else 0.0


yaml = f"""%YAML:1.0

File.version: "1.0"

Camera.type: "PinHole"

# Camera calibration
Camera1.fx: {fx:.10f}
Camera1.fy: {fy:.10f}
Camera1.cx: {cx:.10f}
Camera1.cy: {cy:.10f}

# Distortion
Camera1.k1: {k1:.10f}
Camera1.k2: {k2:.10f}
Camera1.p1: {p1:.10f}
Camera1.p2: {p2:.10f}
Camera1.k3: {k3:.10f}

# Image properties
Camera.width: {width}
Camera.height: {height}
Camera.fps: {FPS}

# OpenCV images are BGR
Camera.RGB: 0

# ORB extractor
ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

# Viewer
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


with open(OUTPUT_FILE, "w") as f:
    f.write(yaml)

print(f"Saved ORB-SLAM3 configuration to {OUTPUT_FILE}")
print()
print(yaml)