from pathlib import Path

from ultralytics import YOLO


WORK_DIR = Path("/home/admin/peztz-ai")
MODEL_PATH = WORK_DIR / "yolov8n.pt"
NCNN_DIR = WORK_DIR / "yolov8n_ncnn_model"


def main() -> None:
    print(f"loading={MODEL_PATH}", flush=True)
    model = YOLO(str(MODEL_PATH))

    print(f"class_0={model.names.get(0)}", flush=True)
    print(f"class_16={model.names.get(16)}", flush=True)

    if model.names.get(0) != "person" or model.names.get(16) != "dog":
        raise RuntimeError("COCO person/dog class mapping is not as expected")

    if NCNN_DIR.is_dir():
        print(f"ncnn_exists={NCNN_DIR}", flush=True)
        return

    exported = model.export(
        format="ncnn",
        imgsz=640,
        device="cpu",
        half=False,
    )
    print(f"exported={exported}", flush=True)


if __name__ == "__main__":
    main()
