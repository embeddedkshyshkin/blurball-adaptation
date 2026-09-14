from collections import deque
from math import atan2, degrees, hypot
from typing import Optional, Tuple

import numpy as np


class MotionEstimator:
    """Robust online kinematics estimator for calibrated table coordinates.

    This follows the useful parts of the supplied ball_kinematics.py approach:
    reject implausible pixel/metric jumps upstream, fit a local polynomial to
    position, and differentiate that fit instead of differentiating adjacent
    detections. Speed and direction use independent windows because they have
    different noise requirements.

    The estimator deliberately does NOT use a long-lived direction EMA. A
    bounce/reversal must become visible quickly; historical motion should not
    pull the arrow back toward the previous flight.
    """

    def __init__(
        self,
        fps: float,
        speed_window_frames: int = 9,
        direction_window_frames: int = 5,
        speed_smoothing_alpha: float = 0.35,
        direction_change_threshold_deg: float = 95.0,
        min_displacement_m: float = 0.012,
        min_speed_kmh: float = 3.0,
        reversal_confirm_frames: int = 2,
        direction_smoothing_alpha: float = 1.0,
    ):
        self.fps = float(fps)
        self.speed_window = max(3, int(speed_window_frames))
        self.direction_window = max(3, int(direction_window_frames))
        self.speed_alpha = float(np.clip(speed_smoothing_alpha, 0.05, 1.0))
        self.direction_threshold = float(direction_change_threshold_deg)
        self.min_displacement_m = float(min_displacement_m)
        self.min_speed_kmh = float(min_speed_kmh)
        self.reversal_confirm_frames = max(1, int(reversal_confirm_frames))
        self.points = deque(maxlen=max(self.speed_window, self.direction_window, 11))
        self.smoothed_speed_kmh: Optional[float] = None
        self.direction_rad: Optional[float] = None
        self.pending_direction_rad: Optional[float] = None
        self.pending_reversal_count = 0
        self.last_frame: Optional[int] = None

    @staticmethod
    def _local_poly_velocity(points, degree=2) -> Optional[Tuple[float, float]]:
        """Derivative at the newest point from a local weighted polynomial fit."""
        if len(points) < 3:
            return None
        t = np.asarray([p[0] for p in points], dtype=np.float64)
        x = np.asarray([p[1] for p in points], dtype=np.float64)
        y = np.asarray([p[2] for p in points], dtype=np.float64)
        t = t - t[-1]
        if np.ptp(t) <= 0:
            return None

        span = max(float(np.max(np.abs(t))), 1e-9)
        # The same compact weighting idea as the supplied reference script:
        # recent points matter most while the full window suppresses jitter.
        w = np.clip((1.0 - np.abs(t / span) ** 3) ** 3, 1e-3, None)
        deg = min(int(degree), len(t) - 1)
        V = np.vander(t, deg + 1, increasing=True)
        try:
            cx, *_ = np.linalg.lstsq(V * w[:, None], x * w, rcond=None)
            cy, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return float(cx[1]), float(cy[1])

    @staticmethod
    def _linear_velocity(points) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None
        t = np.asarray([p[0] for p in points], dtype=np.float64)
        x = np.asarray([p[1] for p in points], dtype=np.float64)
        y = np.asarray([p[2] for p in points], dtype=np.float64)
        t -= t.mean()
        denom = float(np.dot(t, t))
        if denom <= 0:
            return None
        return (
            float(np.dot(t, x - x.mean()) / denom),
            float(np.dot(t, y - y.mean()) / denom),
        )

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        return abs(atan2(np.sin(a - b), np.cos(a - b)))

    def _velocity_candidates(self):
        points = list(self.points)
        candidates = []
        # Direction is deliberately short and responsive. Using 5/7/9-frame
        # fits and taking a robust median is much less sensitive to one bad
        # TrackNet/BlurBall coordinate than one fit alone.
        for window in (self.direction_window, min(7, self.speed_window), min(9, self.speed_window)):
            if len(points) >= window:
                v = self._local_poly_velocity(points[-window:])
                if v is not None:
                    candidates.append(v)
        return candidates

    def reset(self):
        self.points.clear()
        self.smoothed_speed_kmh = None
        self.direction_rad = None
        self.pending_direction_rad = None
        self.pending_reversal_count = 0
        self.last_frame = None

    def update(self, frame_index: int, table_xy: Tuple[float, float]):
        frame_index = int(frame_index)
        x, y = float(table_xy[0]), float(table_xy[1])

        # A gap means we cannot safely differentiate across it. Start a new
        # flight segment instead of contaminating speed/direction with stale data.
        if self.last_frame is not None and frame_index <= self.last_frame:
            self.reset()
        elif self.last_frame is not None and frame_index - self.last_frame > 3:
            self.reset()

        self.points.append((frame_index, x, y))
        self.last_frame = frame_index

        if len(self.points) < 3:
            return 0.0, self.direction_rad

        # ---- speed: longer polynomial fit + mild EMA ----------------------
        speed_points = list(self.points)[-self.speed_window:]
        speed_velocity = self._local_poly_velocity(speed_points)
        if speed_velocity is None:
            speed_velocity = self._linear_velocity(speed_points)

        speed_kmh = 0.0
        if speed_velocity is not None:
            vx, vy = speed_velocity
            speed_mps = hypot(vx, vy) * self.fps
            raw_speed = speed_mps * 3.6
            if np.isfinite(raw_speed) and raw_speed >= 0:
                if self.smoothed_speed_kmh is None:
                    self.smoothed_speed_kmh = raw_speed
                else:
                    self.smoothed_speed_kmh = (
                        self.speed_alpha * raw_speed
                        + (1.0 - self.speed_alpha) * self.smoothed_speed_kmh
                    )
                speed_kmh = self.smoothed_speed_kmh

        # ---- direction: robust short-window velocity ----------------------
        candidates = self._velocity_candidates()
        candidate_direction = None
        candidate_speed_mps = 0.0
        if candidates:
            vx = float(np.median([v[0] for v in candidates]))
            vy = float(np.median([v[1] for v in candidates]))
            candidate_speed_mps = hypot(vx, vy) * self.fps
            # Direction needs real displacement, not just numerical derivative
            # noise. Approximate displacement over the shortest direction window.
            displacement = candidate_speed_mps * max(1, self.direction_window - 1) / self.fps
            if (displacement >= self.min_displacement_m
                    and candidate_speed_mps * 3.6 >= self.min_speed_kmh):
                candidate_direction = atan2(vy, vx)

        if candidate_direction is not None:
            if self.direction_rad is None:
                self.direction_rad = candidate_direction
                self.pending_direction_rad = None
                self.pending_reversal_count = 0
            else:
                delta_deg = degrees(self._angle_delta(candidate_direction, self.direction_rad))
                if delta_deg >= self.direction_threshold:
                    # Require only a very short confirmation. A table-tennis
                    # bounce happens over a few frames; three-frame hysteresis
                    # visibly delayed the arrow in the previous implementation.
                    if self.pending_direction_rad is None:
                        self.pending_direction_rad = candidate_direction
                        self.pending_reversal_count = 1
                    else:
                        pending_delta = degrees(
                            self._angle_delta(candidate_direction, self.pending_direction_rad)
                        )
                        if pending_delta <= 35.0:
                            self.pending_reversal_count += 1
                        else:
                            self.pending_direction_rad = candidate_direction
                            self.pending_reversal_count = 1

                    if self.pending_reversal_count >= self.reversal_confirm_frames:
                        self.direction_rad = self.pending_direction_rad
                        self.pending_direction_rad = None
                        self.pending_reversal_count = 0
                        # New flight: don't let the previous flight's speed EMA
                        # bleed through the bounce.
                        self.smoothed_speed_kmh = speed_kmh if speed_kmh > 0 else None
                else:
                    self.pending_direction_rad = None
                    self.pending_reversal_count = 0
                    # For normal curvature use the current measured tangent.
                    # No persistent EMA: direction should follow the ball.
                    self.direction_rad = candidate_direction

        return speed_kmh, self.direction_rad
