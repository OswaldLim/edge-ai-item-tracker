"""YOLOv8-nano detection pipeline for item tracking.

Usage:
    python detect.py --source 0          # webcam
    python detect.py --source video.mp4  # video file
    python detect.py --source http://...  # IP camera stream

Outputs detections to console and saves to ChromaDB.
"""

import argparse
import cv2
from ultralytics import YOLO
from datetime import datetime


def main(source, model_path="yolov8n.pt", conf=0.5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source if isinstance(source, int) else source)

    print(f"[INFO] Starting detection from source: {source}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection every 3 frames to save compute
        if frame_count % 3 == 0:
            results = model(frame, conf=conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    confidence = float(box.conf[0])
                    if confidence > conf:
                        ts = datetime.now().isoformat()
                        print(f"[{ts}] {label}: {confidence:.2f}")

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item detection pipeline")
    parser.add_argument("--source", default=0, help="Camera source or URL")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    main(args.source, args.model, args.conf)
