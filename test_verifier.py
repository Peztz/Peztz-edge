import json
import time
from pathlib import Path

from ultralytics import YOLO


WORK_DIR = Path("/home/admin/peztz-ai")
MODEL_PATH = WORK_DIR / "yolov8n_ncnn_model"
IMAGE_NAMES = ("DOG1.jpg", "tapo_dog_frame.jpg", "rtsp_frame.jpg")
TARGET_CLASSES = {0: "person", 16: "dog"}


def main() -> None:
    model = YOLO(str(MODEL_PATH), task="detect")

    for image_name in IMAGE_NAMES:
        image_path = WORK_DIR / image_name
        if not image_path.is_file():
            print(json.dumps({"image": image_name, "error": "missing"}))
            continue

        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.10,
            classes=list(TARGET_CLASSES),
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - started
        height, width = result.orig_shape

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            detections.append(
                {
                    "class": TARGET_CLASSES[class_id],
                    "confidence": round(confidence, 3),
                    "bbox_normalized": [
                        round(x1 / width, 4),
                        round(y1 / height, 4),
                        round(x2 / width, 4),
                        round(y2 / height, 4),
                    ],
                }
            )

        print(
            json.dumps(
                {
                    "image": image_name,
                    "seconds": round(elapsed, 3),
                    "detections": detections,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
