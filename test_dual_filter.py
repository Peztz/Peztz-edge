import json
from pathlib import Path

from ultralytics import YOLO


WORK_DIR = Path("/home/admin/peztz-ai")
IMAGE_NAMES = ("DOG1.jpg", "tapo_dog_frame.jpg", "rtsp_frame.jpg")


def iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def boxes(result, class_ids: set[int]) -> list[dict]:
    height, width = result.orig_shape
    output = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id not in class_ids:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        output.append(
            {
                "class_id": class_id,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
            }
        )
    return output


def main() -> None:
    custom = YOLO(str(WORK_DIR / "best_ncnn_model"), task="detect")
    verifier = YOLO(str(WORK_DIR / "yolov8n_ncnn_model"), task="detect")

    for image_name in IMAGE_NAMES:
        image = str(WORK_DIR / image_name)
        custom_result = custom.predict(image, imgsz=640, conf=0.50, verbose=False)[0]
        custom_dogs = boxes(custom_result, {0})

        verifier_result = verifier.predict(
            image,
            imgsz=640,
            conf=0.10,
            classes=[0, 16],
            verbose=False,
        )[0]
        people = boxes(verifier_result, {0})
        verifier_dogs = boxes(verifier_result, {16})

        matches = []
        for candidate in custom_dogs:
            best_iou = max(
                (iou(candidate["bbox"], dog["bbox"]) for dog in verifier_dogs),
                default=0.0,
            )
            matches.append(
                {
                    "custom_confidence": candidate["confidence"],
                    "best_dog_iou": round(best_iou, 3),
                    "accepted": best_iou >= 0.20,
                }
            )

        print(
            json.dumps(
                {
                    "image": image_name,
                    "custom_candidates": len(custom_dogs),
                    "verifier_people": [item["confidence"] for item in people],
                    "verifier_dogs": [item["confidence"] for item in verifier_dogs],
                    "matches": matches,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
