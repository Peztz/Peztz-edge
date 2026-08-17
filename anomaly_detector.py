import math
import time
from collections import deque


class AnomalyDetector:
    def __init__(
        self,
        history_size=10,
        distance_margin=30.0,
        ratio_margin=0.3,
        cooldown_seconds=10.0,
    ):
        self.distance_history = deque(maxlen=history_size)
        self.ratio_history = deque(maxlen=history_size)

        self.distance_margin = distance_margin
        self.ratio_margin = ratio_margin
        self.cooldown_seconds = cooldown_seconds

        self.prev_center_x = None
        self.prev_center_y = None
        self.prev_ratio = None

        self.last_event_time = {
            "PACING": 0.0,
            "SPINNING": 0.0,
        }

    def update(self, bbox):
        """
        bbox: [x1, y1, x2, y2]

        return:
        {
            "is_pacing": bool,
            "is_spinning": bool,
            "event_types": list[str],
            "distance": float,
            "avg_distance": float,
            "ratio_diff": float,
            "avg_ratio_diff": float,
            "should_log_pacing": bool,
            "should_log_spinning": bool
        }
        """

        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        width = x2 - x1
        height = y2 - y1

        current_ratio = width / height if height > 0 else 0.0

        is_pacing = False
        is_spinning = False

        distance = 0.0
        avg_distance = 0.0

        ratio_diff = 0.0
        avg_ratio_diff = 0.0

        # ==========================
        # PACING
        # ==========================
        if self.prev_center_x is not None:
            distance = math.sqrt(
                (center_x - self.prev_center_x) ** 2
                + (center_y - self.prev_center_y) ** 2
            )

            if len(self.distance_history) == self.distance_history.maxlen:
                avg_distance = (
                    sum(self.distance_history)
                    / len(self.distance_history)
                )

                if distance > (avg_distance + self.distance_margin):
                    is_pacing = True

            self.distance_history.append(distance)

        # ==========================
        # SPINNING
        # ==========================
        if self.prev_ratio is not None:
            ratio_diff = abs(current_ratio - self.prev_ratio)

            if len(self.ratio_history) == self.ratio_history.maxlen:
                avg_ratio_diff = (
                    sum(self.ratio_history)
                    / len(self.ratio_history)
                )

                if ratio_diff > (avg_ratio_diff + self.ratio_margin):
                    is_spinning = True

            self.ratio_history.append(ratio_diff)

        # ==========================
        # 이전 상태 갱신
        # ==========================
        self.prev_center_x = center_x
        self.prev_center_y = center_y
        self.prev_ratio = current_ratio

        # ==========================
        # 이벤트 타입
        # ==========================
        event_types = []

        if is_pacing:
            event_types.append("PACING")

        if is_spinning:
            event_types.append("SPINNING")

        # ==========================
        # DB/로그용 쿨다운
        # ==========================
        now = time.monotonic()

        should_log_pacing = False
        should_log_spinning = False

        if is_pacing:
            if now - self.last_event_time["PACING"] >= self.cooldown_seconds:
                should_log_pacing = True
                self.last_event_time["PACING"] = now

        if is_spinning:
            if now - self.last_event_time["SPINNING"] >= self.cooldown_seconds:
                should_log_spinning = True
                self.last_event_time["SPINNING"] = now

        return {
            "center_x": center_x,
            "center_y": center_y,

            "is_pacing": is_pacing,
            "is_spinning": is_spinning,

            "event_types": event_types,

            "distance": distance,
            "avg_distance": avg_distance,

            "ratio_diff": ratio_diff,
            "avg_ratio_diff": avg_ratio_diff,

            "should_log_pacing": should_log_pacing,
            "should_log_spinning": should_log_spinning,
        }

    def reset(self):
        self.distance_history.clear()
        self.ratio_history.clear()

        self.prev_center_x = None
        self.prev_center_y = None
        self.prev_ratio = None