#!/usr/bin/env python3

import os

# OpenCV import 전에 설정
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO

from anomaly_detector import AnomalyDetector


# =========================================================
# 기본 설정
# =========================================================
MODEL_PATH = "/home/admin/peztz-ai/best_ncnn_model"
RTSP_URL = os.environ.get("CAMERA_RTSP_URL")

WEB_BIND = "0.0.0.0"
WEB_PORT = 8081

YOLO_IMGSZ = 640
YOLO_CONF = 0.80

if not RTSP_URL:
    raise SystemExit("CAMERA_RTSP_URL is not set")


# =========================================================
# 이상행동 탐지기
# =========================================================
anomaly_detector = AnomalyDetector(
    history_size=10,
    distance_margin=30.0,
    ratio_margin=0.3,
    cooldown_seconds=10.0,
)


# =========================================================
# 최신 카메라 프레임
# =========================================================
frame_lock = threading.Lock()

latest_frame = None
latest_frame_id = 0
latest_captured_at = 0.0


# =========================================================
# 최신 YOLO 결과
# =========================================================
detection_lock = threading.Lock()

latest_detections = []
latest_detection_frame_id = 0
latest_event = "NO_DOG"
latest_yolo_ms = 0.0


# =========================================================
# 브라우저 JPEG
# =========================================================
jpeg_lock = threading.Lock()
latest_jpeg = None


# =========================================================
# 종료
# =========================================================
stop_event = threading.Event()


# =========================================================
# RTSP 연결
# =========================================================
def open_camera():

    cap = cv2.VideoCapture(
        RTSP_URL,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    return cap


# =========================================================
# RTSP 최신 프레임 수신
# =========================================================
def camera_reader():

    global latest_frame
    global latest_frame_id
    global latest_captured_at

    cap = None

    print("📷 Tapo RTSP 수신 스레드 시작")

    while not stop_event.is_set():

        if cap is None or not cap.isOpened():

            print("📡 Tapo RTSP 연결 시도...")

            cap = open_camera()

            if not cap.isOpened():

                print("❌ RTSP 연결 실패 - 1초 후 재시도")

                cap.release()
                cap = None

                stop_event.wait(1.0)
                continue

            print("✅ Tapo RTSP 연결 성공")


        success, frame = cap.read()

        if not success or frame is None:

            print("⚠️ RTSP 프레임 수신 실패 - 재연결")

            cap.release()
            cap = None

            stop_event.wait(0.5)
            continue


        # 최신 프레임만 유지
        with frame_lock:

            latest_frame = frame
            latest_frame_id += 1
            latest_captured_at = time.time()


    if cap is not None:
        cap.release()


# =========================================================
# 최신 프레임 가져오기
# =========================================================
def get_latest_frame():

    with frame_lock:

        if latest_frame is None:
            return None

        return (
            latest_frame_id,
            latest_captured_at,
            latest_frame.copy()
        )


# =========================================================
# YOLO + 이상행동 분석
# =========================================================
def yolo_worker():

    global latest_detections
    global latest_detection_frame_id
    global latest_event
    global latest_yolo_ms


    print("🤖 YOLO NCNN 모델 로딩...")

    model = YOLO(
        MODEL_PATH,
        task="detect"
    )

    print("✅ YOLO 모델 준비 완료")


    last_processed_frame_id = -1


    while not stop_event.is_set():

        packet = get_latest_frame()

        if packet is None:

            time.sleep(0.01)
            continue


        frame_id, captured_at, frame = packet


        if frame_id == last_processed_frame_id:

            time.sleep(0.005)
            continue


        last_processed_frame_id = frame_id


        # =================================================
        # YOLO 추론
        # =================================================
        started = time.perf_counter()

        try:

            results = model.predict(
                source=frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                verbose=False
            )

        except Exception as exc:

            print(f"❌ YOLO 추론 오류: {exc}")

            time.sleep(0.5)
            continue


        yolo_ms = (
            time.perf_counter() - started
        ) * 1000


        detections = []
        dog_found = False


        # =================================================
        # DOG 탐지
        # =================================================
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = str(
                    model.names[class_id]
                )

                confidence = float(
                    box.conf[0]
                )


                if "DOG" not in class_name.upper():
                    continue


                dog_found = True


                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )


                # =========================================
                # 공통 이상행동 모듈 호출
                # =========================================
                anomaly = anomaly_detector.update(
                    [x1, y1, x2, y2]
                )


                is_pacing = anomaly["is_pacing"]
                is_spinning = anomaly["is_spinning"]


                # =========================================
                # 터미널 로그
                # =========================================
                if is_pacing:

                    print(
                        "⚠️ [배회 감지] "
                        f"평균 이동량: "
                        f"{anomaly['avg_distance']:.1f}"
                        " | "
                        f"현재 이동량: "
                        f"{anomaly['distance']:.1f}"
                    )


                if is_spinning:

                    print(
                        "⚠️ [맴돎 감지] "
                        f"평균 변화량: "
                        f"{anomaly['avg_ratio_diff']:.2f}"
                        " | "
                        f"현재 변화량: "
                        f"{anomaly['ratio_diff']:.2f}"
                    )


                event_types = anomaly["event_types"]


                if event_types:

                    print(
                        "🚨 [PEZTZ 이상행동] "
                        f"frame={frame_id} "
                        f"event={','.join(event_types)}"
                    )


                # =========================================
                # DB 연동 예정 포인트
                #
                # 쿨다운이 적용된 이벤트만 True
                # =========================================
                if anomaly["should_log_pacing"]:

                    print(
                        "💾 [DB 저장 예정] "
                        f"event=PACING "
                        f"frame={frame_id}"
                    )


                    # 나중에 여기에
                    # pet_logs INSERT 호출
                    #
                    # save_pet_log(
                    #     log_type="PACING",
                    #     ...
                    # )


                if anomaly["should_log_spinning"]:

                    print(
                        "💾 [DB 저장 예정] "
                        f"event=SPINNING "
                        f"frame={frame_id}"
                    )


                    # 나중에 여기에
                    # pet_logs INSERT 호출


                # =========================================
                # 화면 표시용 결과
                # =========================================
                if is_pacing and is_spinning:

                    event_name = (
                        "PACING + SPINNING"
                    )

                elif is_pacing:

                    event_name = "PACING"

                elif is_spinning:

                    event_name = "SPINNING"

                else:

                    event_name = "DOG"


                detections.append(
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),

                        "center_x": anomaly["center_x"],
                        "center_y": anomaly["center_y"],

                        "confidence": confidence,

                        "event": event_name,

                        "is_pacing": is_pacing,
                        "is_spinning": is_spinning,

                        "distance": anomaly["distance"],
                        "avg_distance": anomaly["avg_distance"],

                        "ratio_diff": anomaly["ratio_diff"],
                        "avg_ratio_diff": anomaly["avg_ratio_diff"],
                    }
                )


        # =================================================
        # 최신 YOLO 상태
        # =================================================
        with detection_lock:

            latest_detections = detections

            latest_detection_frame_id = frame_id

            latest_yolo_ms = yolo_ms


            if any(
                d["is_pacing"] or d["is_spinning"]
                for d in detections
            ):

                latest_event = "ANOMALY"

            elif dog_found:

                latest_event = "DOG"

            else:

                latest_event = "NO_DOG"


        print(
            f"[AI] "
            f"frame={frame_id} "
            f"DOG={'YES' if dog_found else 'NO'} "
            f"inference={yolo_ms:.0f}ms"
        )


# =========================================================
# 탐지 결과 가져오기
# =========================================================
def get_detection_state():

    with detection_lock:

        return (
            [
                dict(d)
                for d in latest_detections
            ],
            latest_detection_frame_id,
            latest_event,
            latest_yolo_ms
        )


# =========================================================
# 웹 화면 생성
# =========================================================
def display_worker():

    global latest_jpeg

    last_frame_id = -1


    while not stop_event.is_set():

        packet = get_latest_frame()

        if packet is None:

            time.sleep(0.01)
            continue


        frame_id, captured_at, frame = packet


        if frame_id == last_frame_id:

            time.sleep(0.005)
            continue


        last_frame_id = frame_id


        (
            detections,
            detection_frame_id,
            event,
            yolo_ms
        ) = get_detection_state()


        # =================================================
        # bbox 표시
        # =================================================
        for detection in detections:

            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]

            center_x = detection["center_x"]
            center_y = detection["center_y"]

            confidence = detection["confidence"]

            anomaly = (
                detection["is_pacing"]
                or detection["is_spinning"]
            )


            # 정상 DOG = 초록
            # 이상행동 = 빨강
            color = (
                (0, 0, 255)
                if anomaly
                else (0, 255, 0)
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )


            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                color,
                -1
            )


            label = (
                f"{detection['event']} "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )


        # =================================================
        # 상태 표시
        # =================================================
        if event == "ANOMALY":

            status_text = (
                "PEZTZ AI : ANOMALY"
            )

            status_color = (
                0,
                0,
                255
            )


        elif event == "DOG":

            status_text = (
                "PEZTZ AI : DOG"
            )

            status_color = (
                0,
                255,
                0
            )


        else:

            status_text = (
                "PEZTZ AI : NO DOG"
            )

            status_color = (
                255,
                255,
                255
            )


        cv2.putText(
            frame,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )


        cv2.putText(
            frame,
            f"YOLO {yolo_ms:.0f} ms",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255
            ),
            2
        )


        # =================================================
        # 웹 전송용 축소
        # =================================================
        height, width = frame.shape[:2]

        display_width = 720

        display_height = int(
            height
            *
            (
                display_width
                /
                width
            )
        )


        display_frame = cv2.resize(
            frame,
            (
                display_width,
                display_height
            )
        )


        ok, encoded = cv2.imencode(
            ".jpg",
            display_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                50
            ]
        )


        if not ok:
            continue


        with jpeg_lock:

            latest_jpeg = (
                encoded.tobytes()
            )


# =========================================================
# 웹 서버
# =========================================================
class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":

            page = """
<!DOCTYPE html>

<html lang="ko">

<head>

<meta charset="UTF-8">

<title>PEZTZ AI Vision Test</title>

<style>

body {
    margin: 0;
    background: #08111f;
    color: white;
    font-family: Arial, sans-serif;
}

main {
    width: min(1000px, 96vw);
    margin: 20px auto;
}

.viewer {
    background: black;
    border-radius: 10px;
    overflow: hidden;
}

img {
    display: block;
    width: 100%;
}

</style>

</head>

<body>

<main>

<h1>PEZTZ Edge AI Vision</h1>

<p>
Tapo stream2 · YOLO NCNN ·
PACING / SPINNING
</p>

<div class="viewer">

<img src="/video">

</div>

</main>

</body>

</html>
"""

            body = page.encode(
                "utf-8"
            )


            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(body)

            return


        if self.path == "/video":

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )

            self.send_header(
                "Cache-Control",
                "no-cache, no-store, must-revalidate"
            )

            self.end_headers()


            try:

                last_sent = None


                while not stop_event.is_set():

                    with jpeg_lock:

                        jpeg = latest_jpeg


                    if (
                        jpeg is None
                        or jpeg is last_sent
                    ):

                        time.sleep(0.01)
                        continue


                    last_sent = jpeg


                    self.wfile.write(
                        b"--frame\r\n"
                    )

                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                    )

                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                        .encode()
                    )

                    self.wfile.write(jpeg)

                    self.wfile.write(
                        b"\r\n"
                    )


            except (
                BrokenPipeError,
                ConnectionResetError
            ):

                pass


            return


        self.send_response(404)
        self.end_headers()


    def log_message(
        self,
        format,
        *args
    ):
        return


# =========================================================
# MAIN
# =========================================================
def main():

    print(
        "=========================================="
    )

    print(
        " PEZTZ Edge AI 이상행동 테스트"
    )

    print(
        "=========================================="
    )

    print(
        f"MODEL : {MODEL_PATH}"
    )

    print(
        f"CONF  : {YOLO_CONF}"
    )

    print(
        "SOURCE: Tapo stream2"
    )

    print(
        f"WEB   : http://100.98.148.71:{WEB_PORT}"
    )

    print(
        "=========================================="
    )


    camera_thread = threading.Thread(
        target=camera_reader,
        daemon=True
    )


    yolo_thread = threading.Thread(
        target=yolo_worker,
        daemon=True
    )


    display_thread = threading.Thread(
        target=display_worker,
        daemon=True
    )


    camera_thread.start()
    yolo_thread.start()
    display_thread.start()


    server = ThreadingHTTPServer(
        (
            WEB_BIND,
            WEB_PORT
        ),
        StreamHandler
    )


    try:

        server.serve_forever(
            poll_interval=0.2
        )


    except KeyboardInterrupt:

        print(
            "\n🛑 PEZTZ 종료 중..."
        )


    finally:

        stop_event.set()

        server.server_close()

        camera_thread.join(
            timeout=2
        )

        yolo_thread.join(
            timeout=2
        )

        display_thread.join(
            timeout=2
        )


        print(
            "✅ PEZTZ 종료 완료"
        )


if __name__ == "__main__":
    main()