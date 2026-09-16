from __future__ import annotations

"""Causal 3-D ball kinematics from a single calibrated camera.

Builds on top of the calibration parsing already validated in
``ball_kinematics.py`` (imported, not duplicated) and adds the pieces that
module does not have:

  * a pixel -> horizontal-plane (z = z0) intersection, the primitive needed
    to find where a ray meets the table surface or the ball's mid-flight
    height;
  * ``acrossAxisSign`` applied, so a session where the pose solver flipped the
    world-y axis does not silently reconstruct the ball off the table;
  * causal bounce detection: an image-space local peak in v (screen-down),
    confirmed 2 frames later once its ray is known to land on the table
    surface inside the table's bounds;
  * a trailing-window gravity + scalar-vertical-lift fit (7 parameters:
    initial position, initial velocity, one scalar correction to g), refit
    every frame using only frames up to and including the current one.

Coordinate convention used throughout this file: internally every point is
kept in the *raw* R/t frame (what ``calibration_from_pong_eye`` returns
directly). ``acrossAxisSign`` is applied exactly once, at the boundary,
by :meth:`Calibration3D.to_table` / :meth:`Calibration3D.from_table` --
every public method of this module speaks the *signed* table frame where
y always runs 0..width_m regardless of which way the solver's internal axis
pointed. Nothing outside this file needs to know the raw frame exists.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from utils.ball_kinematics import KinematicsCalibration, calibration_from_pong_eye

G = 9.80665
BALL_RADIUS_M = 0.020

# Physical plausibility gates, set from measuring all 155 arcs in
# video/2026-09-15_17-06-07/blurBall/segment_000.csv against the corner-
# validated calibration (see plan). A fit outside these is noise, not ball.
MAX_REPROJ_RMS_PX = 3.0
MIN_VERTICAL_ACCEL = 4.8   # m/s^2 -- half of g; anything looser is not gravity
MAX_VERTICAL_ACCEL = 14.8  # m/s^2 -- 1.5x g; anything tighter rules out spin
TABLE_MARGIN_M = 0.3
MIN_Z_M = -0.05

MIN_FIT_WINDOW = 8
# A bounce pins 3 of the 7 unknowns (position) at t=0, leaving velocity (3)
# and the lift scalar (1) to resolve from the trailing pixels alone -- and
# measuring all 155 arcs earlier, 7-13 anchored points already reached
# sub-pixel reprojection RMS. Anchored arcs may therefore start fitting
# sooner than the 8-frame default used for curvature-only (unanchored) arcs.
MIN_FIT_WINDOW_ANCHORED = 6
MAX_FIT_WINDOW = 45
BOUNCE_CONFIRM_LAG = 2
# Minimum image-v rise on each side of a candidate peak (px) for it to count
# as a bounce rather than sub-pixel detector noise. Picked from measuring
# segment_000: a hand-verified real bounce (frame 6021) has 12.4px of
# curvature; a 60-frame run of a low, flat, continuous shot produced
# "local maxima" every 5-8 frames topping out at 4.65px. 6px sits between.
MIN_BOUNCE_CURVATURE_PX = 6.0


@dataclass
class Calibration3D:
    """Wraps :class:`KinematicsCalibration` with the sign convention applied
    at the boundary, and exposes the ray primitives that module lacks."""

    base: KinematicsCalibration
    across_sign: float
    fps: float

    @property
    def has_pose(self) -> bool:
        return self.base.has_pose

    @property
    def length_m(self) -> float:
        return self.base.length_m

    @property
    def width_m(self) -> float:
        return self.base.width_m

    # ---- sign boundary -------------------------------------------------
    def to_table(self, p_raw: np.ndarray) -> np.ndarray:
        p = np.array(p_raw, dtype=np.float64, copy=True)
        p[..., 1] *= self.across_sign
        return p

    def from_table(self, p_table: np.ndarray) -> np.ndarray:
        # sign is its own inverse (+-1)
        return self.to_table(p_table)

    # ---- ray primitives (operate in the raw frame internally) ----------
    def camera_ray_raw(self, uv: np.ndarray) -> np.ndarray:
        """Unit ray direction(s) in the raw world frame for pixel(s) uv."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        hom = np.column_stack([uv, np.ones(len(uv))])
        d_cam = hom @ np.linalg.inv(self.base.K).T
        d_world = d_cam @ self.base.R
        return d_world / np.linalg.norm(d_world, axis=1, keepdims=True)

    def project_raw(self, P_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project raw-frame 3-D point(s) to pixels; also returns camera-frame
        depth Zc (the quantity that scales pixels via u = fx*Xc/Zc + cx)."""
        P_raw = np.atleast_2d(np.asarray(P_raw, dtype=np.float64))
        cam = P_raw @ self.base.R.T + self.base.t[None, :]
        img = cam @ self.base.K.T
        return img[:, :2] / img[:, 2:3], cam[:, 2]

    def intersect_z_raw(self, uv: np.ndarray, z0: float) -> tuple[np.ndarray, np.ndarray]:
        """Ray(s) through pixel(s) uv intersected with the raw-frame plane
        z = z0. Returns (points [N,3], camera-frame depth Zc [N])."""
        d = self.camera_ray_raw(uv)
        centre = self.base.camera_center
        s = (z0 - centre[2]) / d[:, 2]
        P = centre[None, :] + s[:, None] * d
        _, zc = self.project_raw(P)
        return P, zc

    # ---- table-frame convenience (sign applied) -------------------------
    def intersect_table_surface(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ray(s) through uv intersected with the physical table surface
        (z = 0), in signed table metres."""
        P_raw, zc = self.intersect_z_raw(uv, 0.0)
        return self.to_table(P_raw), zc

    def table_depth_at(self, uv: np.ndarray) -> np.ndarray:
        """Camera-frame depth (m) of the table-surface point under uv(s) --
        the monocular scale prior used before any bounce has been seen."""
        _, zc = self.intersect_table_surface(np.atleast_2d(uv))
        return zc

    def is_inside_table(self, p_table_xy: np.ndarray, margin: float = TABLE_MARGIN_M) -> bool:
        x, y = p_table_xy[0], p_table_xy[1]
        return (-margin <= x <= self.length_m + margin) and (-margin <= y <= self.width_m + margin)


def load_calibration_3d(raw_json: dict, fps: float) -> Calibration3D:
    base = calibration_from_pong_eye(raw_json)
    exposure = raw_json.get("exposure") or {}
    extrinsics = exposure.get("extrinsics") or raw_json.get("extrinsics") or {}
    sign = float(extrinsics.get("acrossAxisSign", 1.0))
    return Calibration3D(base, sign, fps)


# --------------------------------------------------------------------------
# Causal bounce detection
# --------------------------------------------------------------------------

@dataclass
class BounceEvent:
    frame: int
    point_table: np.ndarray  # (x, y, BALL_RADIUS_M) in signed table metres


def detect_bounce_causal(hist_frames: list[int], hist_uv: list[tuple[float, float]],
                          cal: Calibration3D) -> Optional[BounceEvent]:
    """Look for a confirmed bounce at index ``len(hist)-1-BOUNCE_CONFIRM_LAG``.

    A candidate is an image-v local maximum (screen-down peak -- the ball
    momentarily stops falling and starts rising) with 2 frames of causal
    confirmation on each side available in ``hist``, AND at least
    ``MIN_BOUNCE_CURVATURE_PX`` of actual rise on both sides of the peak --
    not just ``>=``. A bare ``>=`` comparison passes on sub-pixel detector
    noise riding a slowly-varying trend: measured against this file, a real
    bounce's curvature (``v[k] - v[k+-2]``) is 8-100px, while a 60-frame
    stretch of a low, flat, continuous shot produced a cluster of "local
    maxima" every 5-8 frames at 1-5px of curvature -- table-height noise, not
    bounces. It is confirmed only if the ray through that pixel also lands on
    z=0 inside the table's bounds -- a peak in the air (top of a lob) is a
    local max too, but does not land on the table, so this rejects it.

    Deliberately returns a *lag*ged event: the frame it reports is
    ``BOUNCE_CONFIRM_LAG`` frames behind the newest frame in ``hist``. The
    caller applies it going forward from the newest frame, never rewriting a
    frame already emitted -- see ``enrich_and_render.py``.
    """
    n = len(hist_frames)
    k = n - 1 - BOUNCE_CONFIRM_LAG
    if k < 2 or k + 2 >= n:
        return None
    if hist_frames[k + 2] - hist_frames[k - 2] > 8:
        return None  # the window has a gap; the "peak" is not trustworthy
    v = [uv[1] for uv in hist_uv]
    if not (v[k] >= v[k - 1] and v[k] >= v[k + 1]
            and v[k] - v[k - 2] >= MIN_BOUNCE_CURVATURE_PX
            and v[k] - v[k + 2] >= MIN_BOUNCE_CURVATURE_PX):
        return None
    if not cal.has_pose:
        return None
    p_table, _ = cal.intersect_table_surface(np.array([hist_uv[k]]))
    p_table = p_table[0]
    if not cal.is_inside_table(p_table[:2], margin=0.15):
        return None
    return BounceEvent(hist_frames[k], np.array([p_table[0], p_table[1], BALL_RADIUS_M]))


# --------------------------------------------------------------------------
# Trailing-window gravity + scalar-lift fit
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    ok: bool
    rms_px: float = float("nan")
    P0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    V0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a_z: float = -G
    t0_frame: int = 0

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) in the *raw* (unsigned-y) frame, metres, at
        seconds ``t`` since ``t0_frame``. Callers apply ``Calibration3D.to_table``
        before exposing either to anything outside this module."""
        a = np.array([0.0, 0.0, self.a_z])
        P = self.P0 + self.V0 * t + 0.5 * a * t * t
        V = self.V0 + a * t
        return P, V

    def plausible(self, cal: Calibration3D, t: float) -> bool:
        if not self.ok or self.rms_px > MAX_REPROJ_RMS_PX:
            return False
        if not (MIN_VERTICAL_ACCEL <= -self.a_z <= MAX_VERTICAL_ACCEL):
            return False
        P_raw, _ = self.state_at(t)
        p_tab = cal.to_table(P_raw)
        if not cal.is_inside_table(p_tab[:2], margin=TABLE_MARGIN_M):
            return False
        if p_tab[2] < MIN_Z_M:
            return False
        return True


def fit_arc_causal(cal: Calibration3D, frames: np.ndarray, uv_raw_frame: np.ndarray,
                    anchor: Optional[BounceEvent] = None) -> FitResult:
    """Fit gravity + scalar-vertical-lift over a trailing window ending at
    ``frames[-1]``. ``uv_raw_frame`` are pixel positions (already KF-filtered
    upstream); the fit itself only ever sees frames <= the current one.

    ``anchor``, if given, is a bounce whose table-frame 3-D point is known
    (z = ball radius) at ``anchor.frame``; the fit is penalised, not
    hard-constrained, to sit near it at that time, which is enough to pin
    scale from the very first frame of a post-bounce arc.
    """
    n = len(frames)
    min_window = MIN_FIT_WINDOW_ANCHORED if anchor is not None else MIN_FIT_WINDOW
    if n < min_window:
        return FitResult(ok=False)
    t0_frame = int(frames[0])
    fps = cal.fps
    t = (frames - t0_frame) / fps
    uv = np.asarray(uv_raw_frame, dtype=np.float64)

    t_anchor = None
    P_anchor_raw = None
    if anchor is not None:
        t_anchor = (anchor.frame - t0_frame) / fps
        if -5.0 / fps <= t_anchor <= t[-1] + 5.0 / fps:
            P_anchor_raw = cal.from_table(anchor.point_table)

    def unpack(p):
        return p[:3], p[3:6], np.array([0.0, 0.0, -G + p[6]])

    def resid(p):
        P0, V0, a = unpack(p)
        P = P0[None, :] + V0[None, :] * t[:, None] + 0.5 * a[None, :] * (t ** 2)[:, None]
        pr, zc = cal.project_raw(P)
        parts = [(pr - uv).ravel()]
        if P_anchor_raw is not None:
            Pa = P0 + V0 * t_anchor + 0.5 * a * t_anchor ** 2
            parts.append(30.0 * (Pa - P_anchor_raw))
        # keep the whole arc in front of the camera
        parts.append(1000.0 * np.minimum(zc - 0.3, 0.0))
        return np.concatenate(parts)

    best = None
    depth_guesses = (2.0, 3.0, 4.5) if P_anchor_raw is None else (None,)
    for zguess in depth_guesses:
        p0 = np.zeros(7)
        if P_anchor_raw is not None:
            p0[:3] = P_anchor_raw
        else:
            centre_uv = uv[len(uv) // 2]
            P_guess, _ = cal.intersect_z_raw(centre_uv[None, :], 0.3)
            p0[:3] = P_guess[0]
        try:
            sol = least_squares(resid, p0, max_nfev=2000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return FitResult(ok=False)

    P0, V0, a = unpack(best.x)
    P = P0[None, :] + V0[None, :] * t[:, None] + 0.5 * a[None, :] * (t ** 2)[:, None]
    pr, _ = cal.project_raw(P)
    rms = float(np.sqrt(np.mean(np.sum((pr - uv) ** 2, axis=1))))
    return FitResult(ok=True, rms_px=rms, P0=P0, V0=V0, a_z=float(a[2]), t0_frame=t0_frame)


# --------------------------------------------------------------------------
# Non-causal per-bounce arc
# --------------------------------------------------------------------------
#
# Everything above is causal. This last section is not, and is kept apart for
# that reason: it chooses its window by looking at frames the current one has
# not reached yet.
#
# What it is for. The causal reconstruction publishes a world position only
# once a trailing window of pixels already supports a fit, which on the
# reference recording is 8% of frames -- and never the opening frames of a
# flight, because MIN_FIT_WINDOW_ANCHORED of them have to accumulate first.
# Drawn as they come, those points read as a scatter rather than a ball.
# Freed from arriving in order, the same fit can start AT a bounce and run
# forward over pixels that have already been recorded.
#
# Why it is anchored to exactly one bounce, and why a bounce-to-bounce fit
# was tried first and rejected: a single camera cannot resolve the scale of a
# parabola from pixels alone. Measured over segment_000, an unanchored fit of
# a bounce-to-bounce window reprojects onto the detections at 1.4-2.4 px
# while placing the ball three to forty-five metres below the table at 264
# m/s^2 -- pixel-perfect and physically absurd. The bounce contact pins it,
# because there and only there the ball's height is known outright.
#
# The closing bounce cannot serve as a second anchor: bounce detection needs
# MIN_BOUNCE_CURVATURE_PX on both sides of an image-v vertex, a flat drive
# does not clear that bar, and a missed contact makes two flights look like
# one. Anchoring both ends of that pair fits one parabola to two flights: 5
# of 51 windows survived the plausibility gates. Growing from one bounce
# until the pixels stop agreeing instead lets the arc end wherever the ball
# actually left it -- a racket contact, a missed bounce -- and 34 of 82
# bounces then yield an arc at 0.87 px median reprojection error.
#
# That the gates are doing real work is checkable: vertical acceleration is
# free over 4.8-14.8 m/s^2 and comes back at 10.1 (quartiles 9.5-10.5)
# against gravity's 9.81, which no part of this fit was told.

# A guard, not a working limit: the arcs actually grown span a median of 16
# frames and at most 32. Same-track bounce pairs reach 66 frames at p90, and
# a 1.1 s ballistic flight peaks 1.5 m above the table -- past the top of a
# defensive lob, so nothing beyond it is one flight.
MAX_ARC_SPAN_SEC = 1.1


@dataclass
class BounceArc:
    """One gravity arc, anchored at a bounce and valid over [first, last]."""

    fit: FitResult
    bounce_frame: int
    first_frame: int
    last_frame: int
    n_detections: int
    after_bounce: bool

    def positions_table(self, cal: Calibration3D, frames) -> np.ndarray:
        """Positions [N,3] in signed table metres at the given frame numbers,
        including frames the detector never saw: inside a validated span the
        arc is the best statement available about where the ball was."""
        t = (np.atleast_1d(np.asarray(frames, dtype=np.float64)) - self.fit.t0_frame) / cal.fps
        return np.array([cal.to_table(self.fit.state_at(float(ti))[0]) for ti in t])


def grow_arc_from_bounce(cal: Calibration3D, frames, uv, bounce: BounceEvent,
                          after_bounce: bool = True) -> Optional[BounceArc]:
    """Longest arc from ``bounce`` that still explains the detector's pixels.

    ``frames``/``uv`` are detections in chronological order with the bounce at
    the near end -- the frames after it when ``after_bounce``, the frames
    before it otherwise, so that the flight into a bounce and the flight out
    of it are each fitted on their own side of whatever separates them.

    The window grows a frame at a time and stops at the first one that breaks
    either reprojection (``MAX_REPROJ_RMS_PX``) or physical plausibility.
    Returns None when even the shortest window fails, which is the honest
    answer for a bounce whose flight the detector did not follow.
    """
    frames = np.atleast_1d(np.asarray(frames))
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    n = len(frames)
    if n < MIN_FIT_WINDOW_ANCHORED:
        return None

    best = None
    for k in range(MIN_FIT_WINDOW_ANCHORED, n + 1):
        win_f, win_uv = (frames[:k], uv[:k]) if after_bounce else (frames[n - k:], uv[n - k:])
        fit = fit_arc_causal(cal, win_f, win_uv, anchor=bounce)
        if not fit.ok or fit.rms_px > MAX_REPROJ_RMS_PX:
            break
        span_s = (win_f[-1] - fit.t0_frame) / cal.fps
        # Plausible at both ends and at the apex, not at one sample: a fit can
        # sit on the table at the frames it was given and dive through it in
        # between.
        if not all(fit.plausible(cal, t) for t in (0.0, span_s / 2.0, span_s)):
            break
        best = (fit, win_f)

    if best is None:
        return None
    fit, win_f = best
    return BounceArc(fit=fit, bounce_frame=int(bounce.frame),
                      first_frame=int(win_f[0]), last_frame=int(win_f[-1]),
                      n_detections=len(win_f), after_bounce=after_bounce)
