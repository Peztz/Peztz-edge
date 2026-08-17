#!/usr/bin/env python3
"""PEZTZ single-model YOLO + anomaly dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlparse

# OpenCV import 전에 설정
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
# 기본 설정
# =========================================================
DEFAULT_MODEL = "/home/admin/peztz-ai/best_ncnn_model"

# 사용자용 고화질 MediaMTX 영상
DEFAULT_STREAM_URL = (
    "http://34.50.7.78:8889/"
    "cage-a1/?controls=true&muted=true&autoplay=true"
)


# =========================================================
# 최신 RTSP 프레임 Reader
# =========================================================
class LatestFrameReader:
    """RTSP를 계속 읽으면서 최신 프레임 하나만 유지."""

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
        self._thread.join(timeout=3)

    def _run(self) -> None:
        capture = None

        while not self._stop.is_set():

            if capture is None or not capture.isOpened():

                capture = cv2.VideoCapture(
                    self.source,
                    cv2.CAP_FFMPEG
                )

                capture.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )

                if not capture.isOpened():

                    print(
                        "⚠️ [RTSP] 연결 실패 - 재시도",
                        flush=True
                    )

                    capture.release()
                    capture = None

                    self._stop.wait(
                        self.reconnect_delay
                    )

                    continue

                print(
                    "✅ [RTSP] Tapo 스트림 연결 성공",
                    flush=True
                )

            success, frame = capture.read()

            if not success or frame is None:

                print(
                    "⚠️ [RTSP] 프레임 수신 실패 - 재연결",
                    flush=True
                )

                capture.release()
                capture = None

                self._stop.wait(
                    self.reconnect_delay
                )

                continue

            # 오래된 프레임을 쌓지 않고 최신 프레임만 보관
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._captured_at = time.time()

        if capture is not None:
            capture.release()


# =========================================================
# Dashboard 상태
# =========================================================
class DetectionState:

    def __init__(self) -> None:

        self._lock = threading.Lock()

        self._payload = {
            "event": "starting",
            "message": "AI 모델을 준비하고 있습니다.",
            "detections": [],
            "anomaly_events": [],
            "pet_log_ready": [],
            "updated_at": None,
        }

    def replace(self, payload: dict) -> None:
        with self._lock:
            self._payload = payload

    def snapshot(self) -> dict:
        with self._lock:

            payload = dict(self._payload)

            for key in (
                "detections",
                "anomaly_events",
                "pet_log_ready",
            ):
                payload[key] = [
                    dict(item)
                    for item in self._payload.get(key, [])
                ]

            return payload


# =========================================================
# 시간
# =========================================================
def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# =========================================================
# 실행 옵션
# =========================================================
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Tapo stream2
    parser.add_argument(
        "--source",
        default=os.environ.get(
            "CAMERA_RTSP_URL",
            ""
        )
    )

    # best_ncnn_model 하나만 사용
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "YOLO_MODEL_PATH",
            DEFAULT_MODEL
        )
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=512
    )

    # 네 커스텀 모델 기준
    parser.add_argument(
        "--conf",
        type=float,
        default=0.85
    )

    # 이상행동 설정
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

    # 대시보드
    parser.add_argument(
        "--bind",
        default="0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8090
    )

    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM_URL
    )

    return parser.parse_args()


# =========================================================
# bbox
# =========================================================
def normalized_detection(
    box,
    width: int,
    height: int,
    class_name: str
) -> dict:

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .tolist()
    )

    return {
        "class_name": class_name,

        "confidence": round(
            float(box.conf[0]),
            4
        ),

        "bbox_normalized": [
            round(x1 / width, 6),
            round(y1 / height, 6),
            round(x2 / width, 6),
            round(y2 / height, 6),
        ],
    }


def normalized_to_pixel_bbox(
    bbox: list[float],
    width: int,
    height: int
) -> list[float]:

    return [
        bbox[0] * width,
        bbox[1] * height,
        bbox[2] * width,
        bbox[3] * height,
    ]


# =========================================================
# AI 메인
# =========================================================
def run_detector(
    args: argparse.Namespace,
    state: DetectionState,
    stop: threading.Event
) -> None:

    # =====================================================
    # best_ncnn_model 하나만 로드
    # =====================================================
    try:
        model = YOLO(
            args.model,
            task="detect"
        )

    except Exception as exc:

        state.replace(
            {
                "event": "error",
                "message": f"모델 로드 실패: {exc}",
                "detections": [],
                "anomaly_events": [],
                "pet_log_ready": [],
                "updated_at": utc_now(),
            }
        )

        return


    # =====================================================
    # 이상행동 Detector
    # =====================================================
    anomaly_detector = AnomalyDetector(
        history_size=args.history_size,
        distance_margin=args.distance_margin,
        ratio_margin=args.ratio_margin,
        cooldown_seconds=args.anomaly_cooldown,
    )


    # =====================================================
    # RTSP
    # =====================================================
    reader = LatestFrameReader(
        args.source
    )

    reader.start()

    last_frame_id = 0


    try:

        while not stop.is_set():

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

            started = time.perf_counter()


            # =================================================
            # YOLO 1회만 실행
            # =================================================
            try:

                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    verbose=False,
                )[0]

            except Exception as exc:

                state.replace(
                    {
                        "event": "error",
                        "message": f"추론 실패: {exc}",
                        "detections": [],
                        "anomaly_events": [],
                        "pet_log_ready": [],
                        "updated_at": utc_now(),
                    }
                )

                time.sleep(1)
                continue


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

                if "DOG" not in class_name.upper():
                    continue

                detections.append(
                    normalized_detection(
                        box,
                        width,
                        height,
                        class_name
                    )
                )


            # =================================================
            # 이상행동
            # =================================================
            anomaly_events = []
            pet_log_ready = []


            if detections:

                # 케이지당 1마리 기준
                # confidence가 가장 높은 DOG 사용
                primary_detection = max(
                    detections,
                    key=lambda item:
                        item["confidence"]
                )

                pixel_bbox = (
                    normalized_to_pixel_bbox(
                        primary_detection[
                            "bbox_normalized"
                        ],
                        width,
                        height
                    )
                )

                anomaly = (
                    anomaly_detector.update(
                        pixel_bbox
                    )
                )

                primary_detection[
                    "anomaly"
                ] = {
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
                }


                # =============================================
                # PACING
                # =============================================
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

                    anomaly_events.append(
                        {
                            "type": "PACING",

                            "distance": round(
                                anomaly["distance"],
                                2
                            ),

                            "avg_distance": round(
                                anomaly["avg_distance"],
                                2
                            ),

                            "confidence":
                                primary_detection[
                                    "confidence"
                                ],

                            "frame_id":
                                frame_id,
                        }
                    )


                # =============================================
                # SPINNING
                # =============================================
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

                    anomaly_events.append(
                        {
                            "type": "SPINNING",

                            "ratio_diff": round(
                                anomaly["ratio_diff"],
                                4
                            ),

                            "avg_ratio_diff": round(
                                anomaly["avg_ratio_diff"],
                                4
                            ),

                            "confidence":
                                primary_detection[
                                    "confidence"
                                ],

                            "frame_id":
                                frame_id,
                        }
                    )


                # =============================================
                # 실시간 이상행동 로그
                # =============================================
                if anomaly["event_types"]:

                    print(
                        json.dumps(
                            {
                                "event":
                                    "anomaly_detected",

                                "anomaly_types":
                                    anomaly[
                                        "event_types"
                                    ],

                                "frame_id":
                                    frame_id,

                                "captured_at":
                                    datetime.fromtimestamp(
                                        captured_at,
                                        timezone.utc
                                    ).isoformat(),

                                "confidence":
                                    primary_detection[
                                        "confidence"
                                    ],
                            },
                            ensure_ascii=False
                        ),
                        flush=True
                    )


                # =============================================
                # DB 저장 예정 PACING
                # =============================================
                if anomaly[
                    "should_log_pacing"
                ]:

                    log_item = {
                        "log_type": "PACING",

                        "frame_id": frame_id,

                        "captured_at":
                            datetime.fromtimestamp(
                                captured_at,
                                timezone.utc
                            ).isoformat(),

                        "confidence":
                            primary_detection[
                                "confidence"
                            ],

                        "metadata": {
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

                            "bbox_normalized":
                                primary_detection[
                                    "bbox_normalized"
                                ],
                        },
                    }

                    pet_log_ready.append(
                        log_item
                    )

                    print(
                        json.dumps(
                            {
                                "event":
                                    "pet_log_ready",
                                **log_item,
                            },
                            ensure_ascii=False
                        ),
                        flush=True
                    )


                # =============================================
                # DB 저장 예정 SPINNING
                # =============================================
                if anomaly[
                    "should_log_spinning"
                ]:

                    log_item = {
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
                            primary_detection[
                                "confidence"
                            ],

                        "metadata": {
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

                            "bbox_normalized":
                                primary_detection[
                                    "bbox_normalized"
                                ],
                        },
                    }

                    pet_log_ready.append(
                        log_item
                    )

                    print(
                        json.dumps(
                            {
                                "event":
                                    "pet_log_ready",
                                **log_item,
                            },
                            ensure_ascii=False
                        ),
                        flush=True
                    )


            # =================================================
            # 시간
            # =================================================
            wall_ms = round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2
            )

            inference_ms = round(
                float(
                    result.speed.get(
                        "inference",
                        0.0
                    )
                ),
                2
            )


            # =================================================
            # 이벤트/상태
            # =================================================
            if anomaly_events:

                event = (
                    "anomaly_detected"
                )

                anomaly_names = ", ".join(
                    item["type"]
                    for item
                    in anomaly_events
                )

                message = (
                    f"이상행동 감지: "
                    f"{anomaly_names}"
                )

            elif detections:

                event = (
                    "dog_detected"
                )

                message = (
                    "강아지가 감지되었습니다."
                )

            else:

                event = (
                    "no_dog"
                )

                message = (
                    "강아지가 감지되지 않았습니다."
                )


            # =================================================
            # 터미널 상태 로그
            # =================================================
            print(
                f"[AI] "
                f"frame={frame_id} "
                f"DOG={'YES' if detections else 'NO'} "
                f"inference={inference_ms:.0f}ms "
                f"wall={wall_ms:.0f}ms",
                flush=True
            )


            # =================================================
            # API 상태 갱신
            # =================================================
            state.replace(
                {
                    "event":
                        event,

                    "message":
                        message,

                    "frame_id":
                        frame_id,

                    "frame_size": [
                        width,
                        height
                    ],

                    "captured_at":
                        datetime.fromtimestamp(
                            captured_at,
                            timezone.utc
                        ).isoformat(),

                    "updated_at":
                        utc_now(),

                    "wall_ms":
                        wall_ms,

                    "inference_ms":
                        inference_ms,

                    "detections":
                        detections,

                    "anomaly_events":
                        anomaly_events,

                    "pet_log_ready":
                        pet_log_ready,

                    "anomaly_config": {
                        "history_size":
                            args.history_size,

                        "distance_margin":
                            args.distance_margin,

                        "ratio_margin":
                            args.ratio_margin,

                        "cooldown_seconds":
                            args.anomaly_cooldown,
                    },
                }
            )


    finally:
        reader.close()


# =========================================================
# Dashboard HTML
# =========================================================
def build_page(
    stream_url: str
) -> bytes:

    safe_stream_url = html.escape(
        stream_url,
        quote=True
    )

    page = r'''
<!doctype html>

<html lang="ko">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
PEZTZ AI 반려견 모니터링
</title>

<style>

:root {
    color-scheme: dark;
    font-family:
        system-ui,
        -apple-system,
        "Segoe UI",
        sans-serif;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    background: #07111f;
    color: #eef6ff;
}

main {
    width: min(1180px, 96vw);
    margin: 24px auto;
}

h1 {
    margin: 0;
}

.sub {
    color: #9db0c7;
    margin-top: 6px;
}

.viewer {
    position: relative;
    width: 100%;
    aspect-ratio: 2688 / 1520;
    overflow: hidden;
    border-radius: 16px;
    background: #000;
    margin-top: 16px;
}

iframe {
    width: 100%;
    height: 100%;
    border: 0;
}

#overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
}

.box {
    position: absolute;
    border: 4px solid #22c55e;
    border-radius: 6px;
}

.box.anomaly {
    border-color: #ef4444;
    box-shadow:
        0 0 25px #ef444488;
}

.label {
    position: absolute;
    left: -4px;
    top: -35px;
    padding: 6px 10px;
    background: #16a34a;
    color: white;
    font-weight: 800;
    white-space: nowrap;
}

.box.anomaly .label {
    background: #dc2626;
}

#badge {
    position: absolute;
    top: 18px;
    right: 18px;
    z-index: 5;
    padding: 11px 15px;
    border-radius: 999px;
    font-weight: 800;
    background: #334155dc;
}

#badge.detected {
    background: #15803de8;
}

#badge.anomaly {
    background: #dc2626e8;
}

#badge.error {
    background: #b91c1ce8;
}

.panel {
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 12px;
    margin-top: 14px;
}

.card {
    background: #101d2e;
    border: 1px solid #24364d;
    border-radius: 12px;
    padding: 14px;
}

.card span {
    display: block;
    color: #91a5bd;
    font-size: 12px;
    margin-bottom: 5px;
}

.card strong {
    font-size: 18px;
}

</style>

</head>


<body>

<main>

<h1>
PEZTZ AI 반려견 모니터링
</h1>

<div class="sub">
MediaMTX 고화질 영상 +
Tapo stream2 AI 분석 +
best_ncnn_model +
PACING / SPINNING
</div>


<section class="viewer">

<iframe
    src="__STREAM_URL__"
    allow="autoplay; fullscreen; picture-in-picture"
    scrolling="no">
</iframe>

<div id="overlay"></div>

<div id="badge">
AI 준비 중
</div>

</section>


<section class="panel">

<div class="card">
<span>탐지 상태</span>
<strong id="status">준비 중</strong>
</div>

<div class="card">
<span>최고 신뢰도</span>
<strong id="confidence">-</strong>
</div>

<div class="card">
<span>이상행동</span>
<strong id="anomaly">-</strong>
</div>

<div class="card">
<span>추론 시간</span>
<strong id="latency">-</strong>
</div>

</section>

</main>


<script>

const badge =
    document.querySelector('#badge');

const overlay =
    document.querySelector('#overlay');

const statusText =
    document.querySelector('#status');

const confidenceText =
    document.querySelector('#confidence');

const anomalyText =
    document.querySelector('#anomaly');

const latencyText =
    document.querySelector('#latency');


function setBadge(
    text,
    kind = ''
) {
    badge.textContent = text;
    badge.className = kind;
}


function drawBoxes(
    detections
) {

    overlay
        .querySelectorAll('.box')
        .forEach(
            node => node.remove()
        );


    detections.forEach(
        item => {

            const [
                x1,
                y1,
                x2,
                y2
            ] = item.bbox_normalized;


            const anomaly =
                item.anomaly || {};


            const box =
                document.createElement(
                    'div'
                );


            if (
                anomaly.is_pacing
                ||
                anomaly.is_spinning
            ) {

                box.className =
                    'box anomaly';

            }

            else {

                box.className =
                    'box';

            }


            box.style.left =
                `${x1 * 100}%`;

            box.style.top =
                `${y1 * 100}%`;

            box.style.width =
                `${(x2 - x1) * 100}%`;

            box.style.height =
                `${(y2 - y1) * 100}%`;


            const label =
                document.createElement(
                    'div'
                );


            label.className =
                'label';


            let name = 'DOG';


            if (
                anomaly.is_pacing
                &&
                anomaly.is_spinning
            ) {

                name =
                    'PACING + SPINNING';

            }

            else if (
                anomaly.is_pacing
            ) {

                name =
                    'PACING';

            }

            else if (
                anomaly.is_spinning
            ) {

                name =
                    'SPINNING';

            }


            label.textContent =
                `${name} ${
                    (
                        item.confidence
                        * 100
                    ).toFixed(1)
                }%`;


            box.appendChild(
                label
            );


            overlay.appendChild(
                box
            );

        }
    );

}


async function refresh() {

    try {

        const response =
            await fetch(
                '/api/status',
                {
                    cache:
                        'no-store'
                }
            );


        const data =
            await response.json();


        const detections =
            data.detections || [];


        const anomalyEvents =
            data.anomaly_events || [];


        drawBoxes(
            detections
        );


        if (
            data.event
            ===
            'anomaly_detected'
        ) {

            const names =
                anomalyEvents
                .map(
                    item => item.type
                )
                .join(', ');


            setBadge(
                `이상행동 · ${names}`,
                'anomaly'
            );


            statusText.textContent =
                '이상행동 감지';


            anomalyText.textContent =
                names || '-';

        }


        else if (
            data.event
            ===
            'dog_detected'
            &&
            detections.length
        ) {

            const top =
                Math.max(
                    ...detections.map(
                        item =>
                            item.confidence
                    )
                );


            setBadge(
                `강아지 ${
                    (
                        top
                        *
                        100
                    ).toFixed(1)
                }%`,
                'detected'
            );


            statusText.textContent =
                '강아지 감지';


            anomalyText.textContent =
                '정상';

        }


        else if (
            data.event
            ===
            'no_dog'
        ) {

            setBadge(
                '강아지 감지 안 됨'
            );


            statusText.textContent =
                '감지 안 됨';


            anomalyText.textContent =
                '-';

        }


        else if (
            data.event
            ===
            'error'
        ) {

            setBadge(
                'AI 오류',
                'error'
            );


            statusText.textContent =
                data.message || '오류';

        }


        if (detections.length) {

            const top =
                Math.max(
                    ...detections.map(
                        item =>
                            item.confidence
                    )
                );


            confidenceText.textContent =
                `${
                    (
                        top
                        *
                        100
                    ).toFixed(1)
                }%`;

        }

        else {

            confidenceText.textContent =
                '-';

        }


        latencyText.textContent =
            data.inference_ms == null
            ?
            '-'
            :
            `${data.inference_ms.toFixed(0)} ms`;


    }

    catch (error) {

        setBadge(
            'AI 서버 연결 끊김',
            'error'
        );

        statusText.textContent =
            '연결 끊김';

    }

}


refresh();

setInterval(
    refresh,
    700
);

</script>

</body>

</html>
'''


    return page.replace(
        "__STREAM_URL__",
        safe_stream_url
    ).encode(
        "utf-8"
    )


# =========================================================
# HTTP
# =========================================================
def make_handler(
    state: DetectionState,
    page: bytes
):

    class DashboardHandler(
        BaseHTTPRequestHandler
    ):

        def do_GET(self) -> None:

            path = urlparse(
                self.path
            ).path


            if path in (
                "/",
                "/index.html"
            ):

                self._send(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    page
                )

                return


            if path == "/api/status":

                payload = json.dumps(
                    state.snapshot(),
                    ensure_ascii=False
                ).encode(
                    "utf-8"
                )

                self._send(
                    HTTPStatus.OK,
                    "application/json; charset=utf-8",
                    payload
                )

                return


            if path == "/healthz":

                self._send(
                    HTTPStatus.OK,
                    "text/plain; charset=utf-8",
                    b"ok\n"
                )

                return


            self._send(
                HTTPStatus.NOT_FOUND,
                "text/plain; charset=utf-8",
                b"not found\n"
            )


        def _send(
            self,
            status: HTTPStatus,
            content_type: str,
            body: bytes
        ) -> None:

            self.send_response(
                status
            )

            self.send_header(
                "Content-Type",
                content_type
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.send_header(
                "Cache-Control",
                "no-store"
            )

            self.end_headers()

            self.wfile.write(
                body
            )


        def log_message(
            self,
            fmt: str,
            *args
        ) -> None:
            return


    return DashboardHandler


# =========================================================
# MAIN
# =========================================================
def main() -> int:

    args = parse_args()


    if not args.source:

        raise SystemExit(
            "CAMERA_RTSP_URL is not set"
        )


    if not Path(
        args.model
    ).exists():

        raise SystemExit(
            f"model path does not exist: "
            f"{args.model}"
        )


    if not (
        0.0
        <= args.conf
        <= 1.0
    ):

        raise SystemExit(
            "--conf must be between 0 and 1"
        )


    state = DetectionState()

    stop = threading.Event()


    detector = threading.Thread(
        target=run_detector,
        args=(
            args,
            state,
            stop
        ),
        daemon=True
    )


    detector.start()


    server = ThreadingHTTPServer(
        (
            args.bind,
            args.port
        ),
        make_handler(
            state,
            build_page(
                args.stream_url
            )
        )
    )


    server.daemon_threads = True


    def request_stop(
        _signum,
        _frame
    ) -> None:

        stop.set()

        threading.Thread(
            target=server.shutdown,
            daemon=True
        ).start()


    signal.signal(
        signal.SIGINT,
        request_stop
    )


    signal.signal(
        signal.SIGTERM,
        request_stop
    )


    print(
        json.dumps(
            {
                "event":
                    "dashboard_started",

                "bind":
                    args.bind,

                "port":
                    args.port,

                "model":
                    args.model,

                "imgsz":
                    args.imgsz,

                "conf":
                    args.conf,

                "history_size":
                    args.history_size,

                "distance_margin":
                    args.distance_margin,

                "ratio_margin":
                    args.ratio_margin,

                "anomaly_cooldown":
                    args.anomaly_cooldown,

                "dashboard":
                    f"http://100.98.148.71:"
                    f"{args.port}/",
            },
            ensure_ascii=False
        ),
        flush=True
    )


    try:

        server.serve_forever(
            poll_interval=0.5
        )


    finally:

        stop.set()

        detector.join(
            timeout=5
        )

        server.server_close()


    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )