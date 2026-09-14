from __future__ import annotations

"""Physics-based ball speed/direction estimation.

The important difference from a simple homography derivative is that an airborne
ball is reconstructed in a per-flight vertical plane when camera intrinsics are
available.  The plane is chosen by reprojection error plus a gravity constraint,
then position is smoothed with a local polynomial and differentiated.

Without solved intrinsics the estimator falls back to the horizontal table-plane
homography, which is exact only when the ball is on/near the table.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

G = 9.80665
MAX_SLOPE = 0.8


@dataclass
class KinematicsCalibration:
    H_table_to_image: np.ndarray
    H_image_to_table: np.ndarray
    K: Optional[np.ndarray]
    R: Optional[np.ndarray]
    t: Optional[np.ndarray]
    camera_center: Optional[np.ndarray]
    length_m: float
    width_m: float

    @property
    def has_pose(self) -> bool:
        return self.K is not None and self.R is not None and self.t is not None


def calibration_from_pong_eye(raw: dict) -> KinematicsCalibration:
    cal = raw.get("calibration", raw)
    corners = cal["corners"]
    length = float(cal["tableDimensions"]["lengthMeters"])
    width = float(cal["tableDimensions"]["widthMeters"])

    image_points = np.array([
        [corners["topLeft"]["x"], corners["topLeft"]["y"]],
        [corners["topRight"]["x"], corners["topRight"]["y"]],
        [corners["bottomRight"]["x"], corners["bottomRight"]["y"]],
        [corners["bottomLeft"]["x"], corners["bottomLeft"]["y"]],
    ], dtype=np.float64)
    table_points = np.array([
        [0.0, 0.0], [length, 0.0], [length, width], [0.0, width]
    ], dtype=np.float64)
    H = cv2.getPerspectiveTransform(image_points.astype(np.float32), table_points.astype(np.float32))
    H /= H[2, 2]

    K = R = t = centre = None
    intr = raw.get("solvedIntrinsics") or cal.get("solvedIntrinsics") or {}
    if "focalLengthPx" in intr:
        f = float(intr["focalLengthPx"])
        size = raw.get("imageSize", {})
        cx = float(intr.get("principalPointXPx", size.get("width", 1920) / 2))
        cy = float(intr.get("principalPointYPx", size.get("height", 1080) / 2))
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        R, t = _pose_from_homography(K, np.linalg.inv(H))
        centre = -R.T @ t
        if centre[2] < 0:
            # Flip only the world z axis; x/y remain the calibrated table axes.
            F = np.diag([1.0, 1.0, -1.0])
            R = R @ F
            centre = -R.T @ t

    return KinematicsCalibration(H, np.linalg.inv(H), K, R, t, centre, length, width)


def _pose_from_homography(K: np.ndarray, H_table_to_image: np.ndarray):
    M = np.linalg.inv(K) @ H_table_to_image
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, tv = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    if tv[2] < 0:
        r1, r2, tv = -r1, -r2, -tv
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R, tv


def _rays_world(cal: KinematicsCalibration, uv: np.ndarray) -> np.ndarray:
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    hom = np.column_stack([uv, np.ones(len(uv))])
    d_cam = hom @ np.linalg.inv(cal.K).T
    return d_cam @ cal.R


def _vertical_points(cal: KinematicsCalibration, uv: np.ndarray, y0: float, slope: float):
    d = _rays_world(cal, uv)
    c = cal.camera_center
    denom = d[:, 1] - slope * d[:, 0]
    if np.any(np.abs(denom) < 1e-9):
        return None
    s = (y0 + slope * c[0] - c[1]) / denom
    P = c[None, :] + s[:, None] * d
    return P if np.isfinite(P).all() else None


def _project(cal: KinematicsCalibration, X: np.ndarray):
    cam = X @ cal.R.T + cal.t[None, :]
    if np.any(cam[:, 2] <= 1e-6):
        return None
    img = cam @ cal.K.T
    return img[:, :2] / img[:, 2:3]


def _horizontal_points(cal: KinematicsCalibration, uv: np.ndarray):
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    p = np.column_stack([uv, np.ones(len(uv))]) @ cal.H_image_to_table.T
    w = p[:, 2]
    w[np.abs(w) < 1e-12] = np.nan
    return np.column_stack([p[:, 0] / w, p[:, 1] / w, np.zeros(len(p))])


def _local_poly_state(t, y, half_window=3, degree=2):
    n = len(t)
    value = np.empty(n, dtype=np.float64)
    derivative = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        ts = t[lo:hi] - t[i]
        ys = y[lo:hi]
        deg = min(degree, len(ts) - 1)
        if deg < 1:
            value[i], derivative[i] = y[i], 0.0
            continue
        span = max(float(np.max(np.abs(ts))), 1e-9)
        w = np.clip((1.0 - np.abs(ts / span) ** 3) ** 3, 1e-3, None)
        V = np.vander(ts, deg + 1, increasing=True)
        try:
            coef, *_ = np.linalg.lstsq(V * w[:, None], ys * w, rcond=None)
            value[i], derivative[i] = coef[0], coef[1]
        except np.linalg.LinAlgError:
            value[i], derivative[i] = y[i], 0.0
    return value, derivative


def _plane_cost(cal, uv, t, y0, slope, bounds, gravity_weight=3.0):
    if not (bounds[0] <= y0 <= bounds[1]) or abs(slope) > MAX_SLOPE:
        return float("inf")
    P = _vertical_points(cal, uv, y0, slope)
    if P is None or np.min(np.linalg.norm(P - cal.camera_center[None, :], axis=1)) < 0.35:
        return float("inf")
    if np.max(np.abs(P[:, 0])) > 25 or np.max(np.abs(P[:, 2])) > 10:
        return float("inf")

    n = len(t)
    w = min(7, n)
    stride = max(1, (n - w) // 8 + 1)
    pixel_errors = []
    gravity_errors = []
    for a in range(0, n - w + 1, stride):
        b = a + w
        tt, xx, zz = t[a:b], P[a:b, 0], P[a:b, 2]
        A = np.column_stack([np.ones_like(tt), tt - tt[0]])
        cx, *_ = np.linalg.lstsq(A, xx, rcond=None)
        z_linear, *_ = np.linalg.lstsq(A, zz + 0.5 * G * (tt - tt[0]) ** 2, rcond=None)
        x_fit = A @ cx
        z_fit = A @ z_linear - 0.5 * G * (tt - tt[0]) ** 2
        model = np.column_stack([x_fit, y0 + slope * x_fit, z_fit])
        uv_fit = _project(cal, model)
        if uv_fit is None:
            return float("inf")
        pixel_errors.append(float(np.sqrt(np.mean(np.sum((uv_fit - uv[a:b]) ** 2, axis=1)))))
        if w >= 4:
            coef = np.polyfit(tt - tt[0], zz, 2)
            gravity_errors.append(abs((2.0 * coef[0]) + G) / G)

    if not pixel_errors:
        return float("inf")
    cost = float(np.median(pixel_errors))
    if gravity_errors:
        cost *= 1.0 + gravity_weight * float(np.median(gravity_errors))
    cost *= 1.0 + 0.6 * max(0.0, -y0, y0 - cal.width_m)
    return cost


def _solve_vertical_plane(cal, uv, t, margin=1.0, gravity_weight=3.0):
    bounds = (-margin, cal.width_m + margin)
    # Coarse-to-fine deterministic search. This avoids making scipy a runtime
    # requirement while retaining the important physical optimisation.
    best = (float("inf"), cal.width_m / 2.0, 0.0)
    for y0 in np.linspace(bounds[0], bounds[1], 17):
        for slope in np.linspace(-MAX_SLOPE, MAX_SLOPE, 9):
            c = _plane_cost(cal, uv, t, y0, slope, bounds, gravity_weight)
            if c < best[0]:
                best = (c, float(y0), float(slope))

    if not math.isfinite(best[0]):
        return None

    # Three local refinements around the best grid point.
    y0, slope = best[1], best[2]
    dy = (bounds[1] - bounds[0]) / 16.0
    ds = 2.0 * MAX_SLOPE / 8.0
    for _ in range(3):
        candidates = []
        for yy in np.linspace(y0 - dy, y0 + dy, 5):
            for ss in np.linspace(slope - ds, slope + ds, 5):
                c = _plane_cost(cal, uv, t, float(yy), float(ss), bounds, gravity_weight)
                candidates.append((c, float(yy), float(ss)))
        c, y0, slope = min(candidates, key=lambda x: x[0])
        dy *= 0.25
        ds *= 0.25

    P = _vertical_points(cal, uv, y0, slope)
    if P is None:
        return None

    # Final physical/reprojection quality checks.
    rms_m = float("inf")
    px_rms = float("inf")
    w = min(7, len(t))
    if w >= 3:
        residuals_m = []
        residuals_px = []
        for a in range(0, len(t) - w + 1, max(1, (len(t) - w) // 8 + 1)):
            b = a + w
            tt, xx, zz = t[a:b], P[a:b, 0], P[a:b, 2]
            A = np.column_stack([np.ones_like(tt), tt - tt[0]])
            cx, *_ = np.linalg.lstsq(A, xx, rcond=None)
            cz, *_ = np.linalg.lstsq(A, zz + 0.5 * G * (tt - tt[0]) ** 2, rcond=None)
            xfit = A @ cx
            zfit = A @ cz - 0.5 * G * (tt - tt[0]) ** 2
            residuals_m.extend(np.sqrt((xx - xfit) ** 2 + (zz - zfit) ** 2))
            uvfit = _project(cal, np.column_stack([xfit, y0 + slope * xfit, zfit]))
            if uvfit is not None:
                residuals_px.extend(np.linalg.norm(uvfit - uv[a:b], axis=1))
        if residuals_m:
            rms_m = float(np.sqrt(np.mean(np.square(residuals_m))))
        if residuals_px:
            px_rms = float(np.sqrt(np.mean(np.square(residuals_px))))

    if rms_m > 0.25 or px_rms > 8.0:
        return None
    return P, y0, slope, rms_m, px_rms


def _segment(frames, uv, max_gap=3, max_px_step=140.0):
    groups, cur = [], [0]
    for i in range(1, len(frames)):
        df = int(frames[i] - frames[i - 1])
        step = float(np.linalg.norm(uv[i] - uv[i - 1]))
        if df <= 0 or df > max_gap or step / max(df, 1) > max_px_step:
            groups.append(np.asarray(cur, dtype=int))
            cur = [i]
        else:
            cur.append(i)
    groups.append(np.asarray(cur, dtype=int))
    return groups


def _prune(frames, uv, max_px_step=140.0, accel_px_frame2=45.0):
    keep = np.ones(len(frames), dtype=bool)
    for i in range(2, len(frames)):
        prev = np.flatnonzero(keep[:i])
        if len(prev) < 2:
            continue
        a, b = prev[-2], prev[-1]
        d1, d2 = frames[b] - frames[a], frames[i] - frames[b]
        if d1 <= 0 or d2 <= 0:
            continue
        v = (uv[b] - uv[a]) / d1
        residual = float(np.linalg.norm(uv[i] - (uv[b] + v * d2)))
        if residual > 0.5 * accel_px_frame2 * d2 * d2 + 12.0 or np.linalg.norm(uv[i] - uv[b]) / d2 > max_px_step:
            keep[i] = False
    return keep


class BallKinematicsEstimator:
    """Estimate per-frame speed and heading from a complete trajectory."""

    def __init__(self, calibration: KinematicsCalibration, fps: float,
                 half_window: int = 3, degree: int = 2,
                 min_track: int = 8, margin: float = 1.0,
                 gravity_weight: float = 3.0):
        self.cal = calibration
        self.fps = float(fps)
        self.half_window = int(max(1, half_window))
        self.degree = int(max(1, degree))
        self.min_track = int(max(4, min_track))
        self.margin = float(margin)
        self.gravity_weight = float(gravity_weight)

    def estimate(self, frames: np.ndarray, uv: np.ndarray, visibility: np.ndarray):
        frames = np.asarray(frames, dtype=int)
        uv = np.asarray(uv, dtype=float)
        visibility = np.asarray(visibility, dtype=bool)
        speed = np.zeros(len(frames), dtype=float)
        heading = np.full(len(frames), np.nan, dtype=float)
        mode = np.full(len(frames), "none", dtype=object)
        confidence = np.zeros(len(frames), dtype=float)

        visible_idx = np.flatnonzero(visibility & np.isfinite(uv).all(axis=1))
        if len(visible_idx) < self.min_track:
            return speed, heading, mode, confidence

        vf, vu = frames[visible_idx], uv[visible_idx]
        for group in _segment(vf, vu):
            if len(group) < self.min_track:
                continue
            f, p = vf[group], vu[group]
            keep = _prune(f, p)
            f, p = f[keep], p[keep]
            if len(f) < self.min_track:
                continue
            t = f.astype(float) / self.fps

            solved = None
            if self.cal.has_pose:
                solved = _solve_vertical_plane(self.cal, p, t, self.margin, self.gravity_weight)

            if solved is not None:
                P = solved[0]
                fit_quality = max(0.0, 1.0 - solved[4] / 8.0)
                track_mode = "vertical"
            else:
                P = _horizontal_points(self.cal, p)
                fit_quality = 0.35
                track_mode = "plane"

            if P is None or not np.isfinite(P).all():
                continue

            tt = t
            smoothed = np.column_stack([
                _local_poly_state(tt, P[:, k], self.half_window, self.degree)[0]
                for k in range(3)
            ])
            V = np.column_stack([
                _local_poly_state(tt, P[:, k], self.half_window, self.degree)[1]
                for k in range(3)
            ])
            speed_mps = np.linalg.norm(V, axis=1)
            hd = np.degrees(np.arctan2(V[:, 1], V[:, 0]))

            # Reject derivative estimates that are dominated by numerical noise.
            reliable = np.isfinite(speed_mps) & (speed_mps >= 0.25)
            for j, original in enumerate(visible_idx[group][keep]):
                speed[original] = float(speed_mps[j] * 3.6) if reliable[j] else 0.0
                heading[original] = float(hd[j]) if reliable[j] else np.nan
                mode[original] = track_mode
                confidence[original] = fit_quality

        return speed, heading, mode, confidence
