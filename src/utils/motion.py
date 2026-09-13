from collections import deque
from math import atan2, cos, degrees, hypot, radians, sin
from typing import Optional, Tuple

import numpy as np


class MotionEstimator:
    """Robust 2-D motion estimator for calibrated table coordinates.

    Direction is estimated from a local linear regression instead of a single
    frame-to-frame segment. Reversals must persist for several observations
    before the direction is changed, which prevents tracker jitter from
    flipping the HUD arrow.
    """

    def __init__(
        self,
        fps: float,
        speed_window_frames: int = 7,
        direction_window_frames: int = 5,
        speed_smoothing_alpha: float = 0.25,
        direction_change_threshold_deg: float = 115.0,
        min_displacement_m: float = 0.015,
        min_speed_kmh: float = 2.0,
        reversal_confirm_frames: int = 3,
        direction_smoothing_alpha: float = 0.35,
    ):
        self.fps = float(fps)
        self.speed_window = max(2, int(speed_window_frames))
        self.direction_window = max(2, int(direction_window_frames))
        self.speed_alpha = float(np.clip(speed_smoothing_alpha, 0.01, 1.0))
        self.direction_threshold = float(direction_change_threshold_deg)
        self.min_displacement_m = float(min_displacement_m)
        self.min_speed_kmh = float(min_speed_kmh)
        self.reversal_confirm_frames = max(1, int(reversal_confirm_frames))
        self.direction_alpha = float(np.clip(direction_smoothing_alpha, 0.01, 1.0))
        self.points = deque(maxlen=max(self.speed_window, self.direction_window) + 2)
        self.smoothed_speed_kmh: Optional[float] = None
        self.direction_rad: Optional[float] = None
        self.pending_direction_rad: Optional[float] = None
        self.pending_reversal_count = 0
        self.last_frame: Optional[int] = None

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        return abs(atan2(sin(a - b), cos(a - b)))

    @staticmethod
    def _angle_blend(a: float, b: float, alpha: float) -> float:
        x = (1.0 - alpha) * cos(a) + alpha * cos(b)
        y = (1.0 - alpha) * sin(a) + alpha * sin(b)
        return atan2(y, x)

    @staticmethod
    def _linear_velocity(points) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None
        t = np.asarray([p[0] for p in points], dtype=np.float64)
        x = np.asarray([p[1] for p in points], dtype=np.float64)
        y = np.asarray([p[2] for p in points], dtype=np.float64)
        t = t - t.mean()
        denom = float(np.dot(t, t))
        if denom <= 0:
            return None
        vx = float(np.dot(t, x - x.mean()) / denom)
        vy = float(np.dot(t, y - y.mean()) / denom)
        return vx, vy

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
        self.points.append((frame_index, x, y))

        speed_kmh = 0.0
        candidate_direction = None

        speed_points = list(self.points)[-self.speed_window:]
        velocity = self._linear_velocity(speed_points)
        if velocity is not None:
            vx, vy = velocity
            speed_mps = hypot(vx, vy) * self.fps
            if speed_mps > 0:
                raw_speed = speed_mps * 3.6
                if self.smoothed_speed_kmh is None:
                    self.smoothed_speed_kmh = raw_speed
                else:
                    self.smoothed_speed_kmh = (
                        self.speed_alpha * raw_speed
                        + (1.0 - self.speed_alpha) * self.smoothed_speed_kmh
                    )
                speed_kmh = self.smoothed_speed_kmh

        direction_points = list(self.points)[-self.direction_window:]
        velocity = self._linear_velocity(direction_points)
        if velocity is not None:
            vx, vy = velocity
            displacement_m = hypot(vx, vy) * max(1, len(direction_points) - 1) / self.fps
            speed_mps = hypot(vx, vy) * self.fps
            if displacement_m >= self.min_displacement_m and speed_mps * 3.6 >= self.min_speed_kmh:
                candidate_direction = atan2(vy, vx)

        if candidate_direction is not None:
            if self.direction_rad is None:
                self.direction_rad = candidate_direction
                self.pending_direction_rad = None
                self.pending_reversal_count = 0
            else:
                delta_deg = degrees(self._angle_delta(candidate_direction, self.direction_rad))
                if delta_deg >= self.direction_threshold:
                    if self.pending_direction_rad is None:
                        self.pending_direction_rad = candidate_direction
                        self.pending_reversal_count = 1
                    else:
                        pending_delta = degrees(
                            self._angle_delta(candidate_direction, self.pending_direction_rad)
                        )
                        if pending_delta < 45.0:
                            self.pending_reversal_count += 1
                        else:
                            self.pending_direction_rad = candidate_direction
                            self.pending_reversal_count = 1

                    if self.pending_reversal_count >= self.reversal_confirm_frames:
                        self.direction_rad = candidate_direction
                        self.pending_direction_rad = None
                        self.pending_reversal_count = 0
                        # A confirmed reversal is a new flight segment. Do not
                        # carry pre-bounce speed smoothing into the new segment.
                        self.smoothed_speed_kmh = speed_kmh if speed_kmh > 0 else None
                else:
                    self.pending_direction_rad = None
                    self.pending_reversal_count = 0
                    self.direction_rad = self._angle_blend(
                        self.direction_rad, candidate_direction, self.direction_alpha
                    )

        self.last_frame = frame_index
        return speed_kmh, self.direction_rad
