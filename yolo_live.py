#!/usr/bin/env python3
"""PEZTZ headless latest-frame YOLO + anomaly detector for Raspberry Pi RTSP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

# =========================================================
# OpenCV import 전에 FFmpeg 설정
# =========================================================
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
)
os.environ.setdefault(
    "OPENCV_FFMPEG_LOGLEVEL",
    "-8"
)

import cv2
from ultralytics import YOLO

from anomaly_detector import AnomalyDetector


# =========================================================
# 최신 RTSP 프레임 Reader
# =========================================================
class LatestFrameReader:
    """RTSP를 계속 읽으면서 가장 최신 프레임 하나만 유지."""

    def __init__(
        self,
        source: str,
        reconnect_delay: float = 1.0
    ) -> None:

        self.source = source
        self.reconnect_delay = reconnect_delay

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._frame = None
        self._frame_id = 0
        self._captured_at = 0.0

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )


    def start(self) -> None:
        self._thread.start()


    def latest(self, after_frame_id: int):

        with self._lock:

            if (
                self._frame is None
                or self._frame_id <= after_frame_id
            ):
                return None

            return (
                self._frame_id,
                self._captured_at,
                self._frame.copy()
            )


    def close(self) -> None:

        self._stop.set()

        self._thread.join(
            timeout=3
        )


    def _run(self) -> None:

        capture = None

        while not self._stop.is_set():

            # =============================================
            # RTSP 연결
            # =============================================
            if (
                capture is None
                or not capture.isOpened()
            ):

                capture = cv2.VideoCapture(
                    self.source,
                    cv2.CAP_FFMPEG
                )

                capture.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )

                if not capture.isOpened():

                    capture.release()
                    capture = None

                    self._stop.wait(
                        self.reconnect_delay
                    )

                    continue


            # =============================================
            # 프레임 수신
            # =============================================
            success, frame = capture.read()


            if (
                not success
                or frame is None
            ):

                capture.release()
                capture = None

                self._stop.wait(
                    self.reconnect_delay
                )

                continue


            # =============================================
            # 오래된 프레임 쌓지 않고 최신 프레임만 저장
            # =============================================
            with self._lock:

                self._frame = frame

                self._frame_id += 1

                self._captured_at = (
                    time.time()
                )


        if capture is not None:
            capture.release()


# =========================================================
# 실행 옵션
# =========================================================
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--model",
        default=os.environ.get(
            "YOLO_MODEL_PATH",
            "/home/admin/peztz-ai/best_ncnn_model"
        ),
        help="PEZTZ custom NCNN YOLO model",
    )


    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )


    # =====================================================
    # 네 커스텀 모델 기준
    # =====================================================
    parser.add_argument(
        "--conf",
        type=float,
        default=0.85
    )


    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="0이면 Ctrl+C 전까지 계속 실행",
    )


    parser.add_argument(
        "--output",
        default="",
        help="최신 탐지 프레임 저장 경로(optional)",
    )


    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        help="DOG 미탐지 로그 출력 간격",
    )


    # =====================================================
    # 이상행동 옵션
    # =====================================================
    parser.add_argument(
        "--history-size",
        type=int,
        default=10
    )


    parser.add_argument(
        "--distance-margin",
        type=float,
        default=30.0
    )


    parser.add_argument(
        "--ratio-margin",
        type=float,
        default=0.3
    )


    parser.add_argument(
        "--anomaly-cooldown",
        type=float,
        default=10.0
    )


    return parser.parse_args()


# =========================================================
# JSON 로그
# =========================================================
def emit(event: dict) -> None:

    print(
        json.dumps(
            event,
            ensure_ascii=False
        ),
        flush=True
    )


# =========================================================
# MAIN
# =========================================================
def main() -> int:

    args = parse_args()


    # =====================================================
    # Tapo RTSP
    # 현재 환경변수에는 stream2를 넣어서 사용
    # =====================================================
    source = os.environ.get(
        "CAMERA_RTSP_URL"
    )


    if not source:

        print(
            "CAMERA_RTSP_URL is not set",
            file=sys.stderr
        )

        return 2


    if not 0.0 <= args.conf <= 1.0:

        print(
            "--conf must be between 0 and 1",
            file=sys.stderr
        )

        return 2


    # =====================================================
    # 출력 이미지 경로(optional)
    # =====================================================
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else None
    )


    if output_path:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


    # =====================================================
    # 종료 이벤트
    # =====================================================
    stop = threading.Event()


    def request_stop(
        _signum,
        _frame
    ) -> None:

        stop.set()


    signal.signal(
        signal.SIGINT,
        request_stop
    )

    signal.signal(
        signal.SIGTERM,
        request_stop
    )


    # =====================================================
    # YOLO NCNN 모델
    # =====================================================
    model = YOLO(
        args.model,
        task="detect"
    )


    # =====================================================
    # 공통 이상행동 모듈
    # =====================================================
    anomaly_detector = AnomalyDetector(
        history_size=args.history_size,
        distance_margin=args.distance_margin,
        ratio_margin=args.ratio_margin,
        cooldown_seconds=args.anomaly_cooldown,
    )


    # =====================================================
    # RTSP 최신 프레임 Reader
    # =====================================================
    reader = LatestFrameReader(
        source
    )

    reader.start()


    # =====================================================
    # 실행 상태
    # =====================================================
    started_at = time.monotonic()

    deadline = (
        started_at + args.duration
        if args.duration > 0
        else None
    )


    last_frame_id = 0
    last_status_at = 0.0
    processed = 0


    # =====================================================
    # 시작 로그
    # =====================================================
    emit(
        {
            "event": "detector_started",

            "model": args.model,

            "imgsz": args.imgsz,

            "conf": args.conf,

            "history_size": args.history_size,

            "distance_margin": args.distance_margin,

            "ratio_margin": args.ratio_margin,

            "anomaly_cooldown": args.anomaly_cooldown,

            "duration": args.duration,

            "source": "Tapo RTSP stream2",
        }
    )


    try:

        while not stop.is_set():

            # =================================================
            # 실행 시간 제한
            # =================================================
            if (
                deadline is not None
                and time.monotonic() >= deadline
            ):
                break


            # =================================================
            # 최신 프레임만 가져오기
            # =================================================
            packet = reader.latest(
                last_frame_id
            )


            if packet is None:

                time.sleep(0.02)
                continue


            (
                frame_id,
                captured_at,
                frame
            ) = packet


            last_frame_id = frame_id


            height, width = (
                frame.shape[:2]
            )


            # =================================================
            # YOLO 추론
            # =================================================
            wall_started = (
                time.perf_counter()
            )


            try:

                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    verbose=False,
                )[0]


            except Exception as exc:

                emit(
                    {
                        "event": "inference_error",
                        "message": str(exc),
                        "frame_id": frame_id,
                    }
                )

                time.sleep(0.5)

                continue


            wall_ms = (
                time.perf_counter()
                - wall_started
            ) * 1000


            processed += 1


            detections = []


            # =================================================
            # DOG 탐지
            # =================================================
            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )


                class_name = str(
                    model.names[class_id]
                )


                # DOG만 사용
                if (
                    "DOG"
                    not in
                    class_name.upper()
                ):

                    continue


                confidence = float(
                    box.conf[0]
                )


                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )


                # =================================================
                # 이상행동 판별
                # =================================================
                anomaly = (
                    anomaly_detector.update(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    )
                )


                # =================================================
                # DOG 탐지 결과
                # =================================================
                detection = {

                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        round(
                            confidence,
                            6
                        ),

                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],

                    "center": [
                        anomaly["center_x"],
                        anomaly["center_y"],
                    ],

                    "normalized_center": [
                        round(
                            anomaly["center_x"]
                            / width,
                            6
                        ),

                        round(
                            anomaly["center_y"]
                            / height,
                            6
                        ),
                    ],

                    # =============================================
                    # 이상행동 분석 결과
                    # =============================================
                    "anomaly": {

                        "is_pacing":
                            anomaly["is_pacing"],

                        "is_spinning":
                            anomaly["is_spinning"],

                        "event_types":
                            anomaly["event_types"],

                        "distance":
                            round(
                                anomaly["distance"],
                                2
                            ),

                        "avg_distance":
                            round(
                                anomaly["avg_distance"],
                                2
                            ),

                        "ratio_diff":
                            round(
                                anomaly["ratio_diff"],
                                4
                            ),

                        "avg_ratio_diff":
                            round(
                                anomaly["avg_ratio_diff"],
                                4
                            ),
                    },
                }


                detections.append(
                    detection
                )


                # =================================================
                # PACING 감지 로그
                # =================================================
                if anomaly["is_pacing"]:

                    print(
                        "⚠️ [배회 감지] "
                        f"평균 이동량: "
                        f"{anomaly['avg_distance']:.1f} "
                        "| "
                        f"현재 이동량: "
                        f"{anomaly['distance']:.1f}",
                        flush=True
                    )


                # =================================================
                # SPINNING 감지 로그
                # =================================================
                if anomaly["is_spinning"]:

                    print(
                        "⚠️ [맴돎 감지] "
                        f"평균 변화량: "
                        f"{anomaly['avg_ratio_diff']:.2f} "
                        "| "
                        f"현재 변화량: "
                        f"{anomaly['ratio_diff']:.2f}",
                        flush=True
                    )


                # =================================================
                # 이상행동 실시간 이벤트
                # =================================================
                if anomaly["event_types"]:

                    emit(
                        {
                            "event":
                                "anomaly_detected",

                            "frame_id":
                                frame_id,

                            "captured_at":
                                datetime.fromtimestamp(
                                    captured_at,
                                    timezone.utc
                                ).isoformat(),

                            "anomaly_types":
                                anomaly[
                                    "event_types"
                                ],

                            "confidence":
                                round(
                                    confidence,
                                    6
                                ),

                            "distance":
                                round(
                                    anomaly[
                                        "distance"
                                    ],
                                    2
                                ),

                            "avg_distance":
                                round(
                                    anomaly[
                                        "avg_distance"
                                    ],
                                    2
                                ),

                            "ratio_diff":
                                round(
                                    anomaly[
                                        "ratio_diff"
                                    ],
                                    4
                                ),

                            "avg_ratio_diff":
                                round(
                                    anomaly[
                                        "avg_ratio_diff"
                                    ],
                                    4
                                ),
                        }
                    )


                # =================================================
                # DB INSERT 예정 이벤트
                #
                # 쿨다운 통과했을 때만 발생한다.
                # =================================================
                if anomaly[
                    "should_log_pacing"
                ]:

                    emit(
                        {
                            "event":
                                "pet_log_ready",

                            "log_type":
                                "PACING",

                            "frame_id":
                                frame_id,

                            "captured_at":
                                datetime.fromtimestamp(
                                    captured_at,
                                    timezone.utc
                                ).isoformat(),

                            "confidence":
                                round(
                                    confidence,
                                    6
                                ),

                            "metadata":
                                {

                                    "distance":
                                        round(
                                            anomaly[
                                                "distance"
                                            ],
                                            2
                                        ),

                                    "avg_distance":
                                        round(
                                            anomaly[
                                                "avg_distance"
                                            ],
                                            2
                                        ),

                                    "bbox": [
                                        round(x1, 2),
                                        round(y1, 2),
                                        round(x2, 2),
                                        round(y2, 2),
                                    ],
                                },
                        }
                    )


                if anomaly[
                    "should_log_spinning"
                ]:

                    emit(
                        {
                            "event":
                                "pet_log_ready",

                            "log_type":
                                "SPINNING",

                            "frame_id":
                                frame_id,

                            "captured_at":
                                datetime.fromtimestamp(
                                    captured_at,
                                    timezone.utc
                                ).isoformat(),

                            "confidence":
                                round(
                                    confidence,
                                    6
                                ),

                            "metadata":
                                {

                                    "ratio_diff":
                                        round(
                                            anomaly[
                                                "ratio_diff"
                                            ],
                                            4
                                        ),

                                    "avg_ratio_diff":
                                        round(
                                            anomaly[
                                                "avg_ratio_diff"
                                            ],
                                            4
                                        ),

                                    "bbox": [
                                        round(x1, 2),
                                        round(y1, 2),
                                        round(x2, 2),
                                        round(y2, 2),
                                    ],
                                },
                        }
                    )


            # =================================================
            # DOG 탐지됨
            # =================================================
            if detections:

                emit(
                    {
                        "event":
                            "dog_detected",

                        "captured_at":
                            datetime.fromtimestamp(
                                captured_at,
                                timezone.utc
                            ).isoformat(),

                        "frame_id":
                            frame_id,

                        "frame_size": [
                            width,
                            height
                        ],

                        "wall_ms":
                            round(
                                wall_ms,
                                2
                            ),

                        "inference_ms":
                            round(
                                float(
                                    result.speed.get(
                                        "inference",
                                        0.0
                                    )
                                ),
                                2
                            ),

                        "detections":
                            detections,
                    }
                )


                # =============================================
                # optional annotated image
                # =============================================
                if output_path:

                    annotated = frame.copy()


                    for item in detections:

                        x1, y1, x2, y2 = (
                            item["bbox"]
                        )


                        anomaly_data = (
                            item["anomaly"]
                        )


                        abnormal = (
                            anomaly_data[
                                "is_pacing"
                            ]
                            or
                            anomaly_data[
                                "is_spinning"
                            ]
                        )


                        color = (
                            (0, 0, 255)
                            if abnormal
                            else (0, 255, 0)
                        )


                        cv2.rectangle(
                            annotated,
                            (
                                int(x1),
                                int(y1)
                            ),
                            (
                                int(x2),
                                int(y2)
                            ),
                            color,
                            3
                        )


                        if abnormal:

                            label = (
                                " + ".join(
                                    anomaly_data[
                                        "event_types"
                                    ]
                                )
                            )

                        else:

                            label = "DOG"


                        label += (
                            f" "
                            f"{item['confidence'] * 100:.1f}%"
                        )


                        cv2.putText(
                            annotated,
                            label,
                            (
                                int(x1),
                                max(
                                    30,
                                    int(y1) - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2
                        )


                    cv2.imwrite(
                        str(output_path),
                        annotated
                    )


            # =================================================
            # DOG 없음
            # =================================================
            elif (
                time.monotonic()
                - last_status_at
                >= args.status_interval
            ):

                emit(
                    {
                        "event":
                            "no_dog",

                        "frame_id":
                            frame_id,

                        "wall_ms":
                            round(
                                wall_ms,
                                2
                            ),
                    }
                )


                last_status_at = (
                    time.monotonic()
                )


    finally:

        reader.close()


    # =====================================================
    # 종료 로그
    # =====================================================
    emit(
        {
            "event":
                "detector_stopped",

            "processed_frames":
                processed,

            "elapsed_seconds":
                round(
                    time.monotonic()
                    - started_at,
                    2
                ),
        }
    )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())