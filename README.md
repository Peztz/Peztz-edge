# PEZTZ Vision Worker

IP 카메라 영상을 MediaMTX RTSP에서 받아 YOLO와 기존 PACING/SPINNING 판정 로직으로
분석하고, 생성된 이벤트를 PEZTZ FastAPI로 전송하는 비전 워커다.

## Docker 이미지 빌드

```bash
docker build -t peztz-vision:local .
```

## 단독 실행

백엔드와 MediaMTX가 연결된 `peztz-internal` Docker 네트워크에서 실행한다.

```bash
cp .env.example .env
docker run --rm \
  --name peztz-vision \
  --network peztz-internal \
  --env-file .env \
  -p 127.0.0.1:18081:8081 \
  peztz-vision:local
```

실제 운영에서는 `Peztz-backend/infra/docker-compose.yml`의 `vision` 프로필을
사용한다.

```bash
cd ~/Peztz-backend/infra
docker compose --profile vision up -d vision
docker compose --profile vision logs -f vision
```

## 설정값

| 환경변수 | 기본값 | 설명 |
|---|---:|---|
| `CAMERA_RTSP_URL` | 필수 | MediaMTX RTSP 주소 |
| `PEZTZ_EVENT_URL` | 필수 | FastAPI의 `/device/events` 주소 |
| `PEZTZ_DEVICE_API_KEY` | 필수 | FastAPI `DEVICE_API_KEY`와 동일한 값 |
| `PEZTZ_CAMERA_ID` | 필수 | Spring DB에 등록된 카메라 UUID |
| `VISION_MODEL_PATH` | `best.pt` | 컨테이너 안의 YOLO 모델 경로 |
| `VISION_DEVICE` | `cpu` | 추론 장치. GPU 전환 시에만 변경 |
| `VISION_FPS` | `5` | 초당 최대 분석 프레임 수 |
| `VISION_IMGSZ` | `640` | YOLO 입력 크기 |
| `VISION_CONF` | `0.80` | 객체 검출 신뢰도 기준 |
| `VISION_WEB_PORT` | `8081` | 상태/테스트 화면 포트 |

`VISION_FPS`는 입력 영상을 녹화하거나 재생하는 FPS가 아니다. 카메라 수신 스레드는
계속 최신 프레임을 유지하고, 분석 스레드만 설정된 주기로 최신 프레임을 가져간다.
따라서 처리가 밀렸을 때 오래된 프레임을 순서대로 분석하지 않는다.

## 상태 확인

```bash
curl http://127.0.0.1:18081/healthz
```

응답이 `ok`이면 프로세스의 HTTP 상태 서버가 동작 중이라는 뜻이다. 실제 분석 상태는
컨테이너 로그와 이벤트 저장 결과를 함께 확인한다.
