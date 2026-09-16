from __future__ import annotations

"""Causal image-space ball tracker.

A constant-acceleration Kalman filter per pixel axis (x, y), independent and
decoupled. It exists to answer three things every frame, using only the
current and earlier measurements:

  * a position for every frame, including the ~48% that have no detection
    (predict-only steps, tagged ``predicted`` rather than ``measured``);
  * an outlier gate, so the spurious multi-hundred-pixel jumps present in the
    raw BlurBall trajectory are rejected instead of corrupting the velocity
    estimate;
  * a velocity estimate whose direction is the screen-space heading the HUD
    draws — no 3-D, no table-frame angle, so there is nothing to project and
    nothing that can be drawn in the wrong frame.

This module knows nothing about calibration, gravity, or the table. Bounce
handling is a single method, ``inflate_after_bounce``, that the orchestrator
calls once a bounce is confirmed by ``ball_kinematics_3d``; it does not detect
bounces itself.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# chi-square critical value, 2 dof, p=0.999 -- a measurement further than this
# from the predicted position (in units of its own uncertainty) is treated as
# a tracking glitch rather than motion.
#
# Raised from 13.8 (p=0.999) to 25.0 after measuring what the tighter gate
# actually cost. Gate width and estimator smoothness are both driven by Q, so
# they were conflated; opening the gate separately is a free win. Judged
# against the desktop auto-label detector on segment_000 by the only metric
# that nets retention against admitted outliers -- accepted frames that land
# within 15px of the independent detector:
#
#   gate 13.8 : 4469 accepted, median err 1.20px, 3.6% >100px -> 4296 accurate
#   gate 25.0 : 4579 accepted, median err 1.20px, 3.6% >100px -> 4405 accurate
#
# i.e. 110 more real detections kept, with the outlier rate and localisation
# error both unchanged. Lowering jerk_std instead does reduce velocity noise
# but rejects genuine detections (4171 accurate at 60k, 3654 at 25k), so the
# speed jitter is handled downstream by a causal EMA on the reported number
# rather than by detuning the filter that produces the positions.
_GATE_CHI2 = 25.0

# Process/measurement noise. ``dt`` is real seconds (1/fps), so state is
# [pos_px, vel_px_per_s, accel_px_per_s^2] and jerk is px/s^3 -- NOT
# px/frame^3. This matters: with dt properly small, the white-noise-jerk
# discretisation's dt^3/dt^4/dt^5 terms correctly shrink the process noise
# injected by a single frame, so a real bounce's sudden reversal (needs a
# generous jerk budget) can be told apart from a single-frame ~1000px
# detector glitch (which must still be rejected). Passing dt=1 ("frame
# units") degenerates that discretisation -- every dt power collapses to 1
# and the gate stops discriminating anything. Values below were picked by
# sweeping against video/2026-09-15_17-06-07/blurBall/segment_000.csv:
# at this setting a known ~250px double-jump (frames 5990-5991) is rejected
# while 84.5% of real detections are kept, and isolated single-frame blips
# (run length <= 3) are accepted at a much lower rate (66%) than points
# that are part of a real run (86%) -- the gate is discriminating, not just
# open or shut.
_DEFAULT_JERK_STD = 150_000.0
# Measurement noise: 1-sigma pixel localisation error of the detector.
_DEFAULT_MEAS_STD = 2.0

# Per-predict-step decay applied to the acceleration point estimate (not its
# covariance, which already grows correctly). See the comment on
# _AxisKF.predict for why this exists: a long run of predict-only steps
# integrating a constant, un-decaying acceleration is quadratically unstable.
# 0.90 kills a runaway acceleration to ~12% within 20 steps, after which a
# coast is bounded (linear in the surviving velocity, not quadratic) and
# max_gap_frames caps how long it can run at all.
#
# Velocity is deliberately NOT decayed the same way. A Kalman update only
# undoes a bias by (gain x innovation); in steady, well-tracked motion the
# velocity gain is small (P has converged near the measurement-noise floor),
# so a per-frame velocity haircut is not "corrected each frame" the way the
# acceleration one effectively is -- it would instead read as a persistent
# few-percent low bias in SpeedKmh across the whole clip. Acceleration decay
# alone already removes the instability; adding velocity decay bought
# boundedness the gap length already provides, at the cost of a real bias.
_ACCEL_DECAY_PER_STEP = 0.90

# The accel decay above bounds a coast from quadratic to linear growth, but
# "linear" over up to max_gap_frames steps is still not "small": measured on
# segment_000 (a segment already validated on every OTHER metric -- direction
# accuracy, retention, bounce count) after adding the decay, 610 rows still
# carried |Xf| or |Yf| beyond 3000px, some past 20000px, all on Source =
# "predicted" gap frames following a bounce inflate(). The mechanism: inflate()
# widens the acceleration/velocity covariance right when a gap is about to
# start, so if the gap begins immediately after, the filter free-runs from an
# already-large velocity/acceleration estimate for up to max_gap_frames steps.
# The resulting SpeedKmh values (as low as ~95-150 km/h on this data) are not
# reliably caught by enrich_and_render.py's MAX_PLAUSIBLE_SPEED_KMH clamp,
# since the decayed velocity can sit just under that threshold even while the
# position it came from is many frames' worth of real motion outside the
# frame -- a bounds check on position is the direct fix, independent of
# whatever the derived speed happens to compute to. 500px is generous enough
# that a ball genuinely leaving frame near an edge is not penalised.
_BOUNDS_MARGIN_PX = 500.0

# Prior spread on a *new* track's velocity/acceleration. reset() starts the
# state at zero velocity, so these say how wrong that is allowed to be.
#
# The previous 20 px/s velocity seed was off by two orders of magnitude: a
# rally ball crosses the frame at 1500-4000 px/s, so the filter began each
# track almost certain the ball was standing still, and the chi-square gate
# then rejected the very measurement that would have taught it the velocity.
# Reproduced directly -- feed a fresh tracker two clean measurements 28px
# apart and the second comes back "predicted", not "measured".
#
# 3000 px/s is ~8 m/s at this recording's depth/focal length (~30 km/h), so a
# 100 km/h smash still sits inside 3 sigma. Swept on segment_000; every
# metric improves together and then saturates, which is what fixing a bug
# looks like rather than trading one off:
#
#   sigma_v   retention  restarts  steady median  steady p90  within 25px
#      20        0.886      126        2.96          13.15       0.944
#     800        0.918      118        2.92          12.01       0.953
#    3000        0.918      120        2.91          12.16       0.952
#
_INIT_VEL_STD = 3000.0   # px/s
_INIT_ACC_STD = 20000.0  # px/s^2 -- gravity plus racket impulse, generously


def _cv_matrices(dt: float, jerk_std: float, meas_std: float):
    """Constant-acceleration transition/noise matrices for one scalar axis."""
    F = np.array([[1.0, dt, 0.5 * dt * dt],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]])
    # Discretised white-noise-jerk process covariance (standard result for a
    # 3rd-order kinematic model driven by white jerk noise).
    dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
    q = jerk_std ** 2
    Q = q * np.array([
        [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
        [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
        [dt3 / 6.0, dt2 / 2.0, dt],
    ])
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[meas_std ** 2]])
    return F, Q, H, R


@dataclass
class _AxisKF:
    F: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e6)
    initialised: bool = False

    def predict(self):
        # Decay acceleration slightly on every predict, not just during a
        # gap. A constant-acceleration model integrated blindly for many
        # consecutive predict-only steps (a real gap) is quadratically
        # unstable: found empirically as Xf/Yf reaching +-10000+ px and
        # SpeedKmh reaching 5 figures on segment_001, always on a long
        # predict-only run where the last fitted acceleration (often
        # widened right after `inflate()`) got integrated unchecked. Decaying
        # the ACCELERATION state (not the covariance -- that already grows
        # correctly) means a long coast degrades toward constant *velocity*
        # (bounded, since max_gap_frames caps how long it can run) rather
        # than diverging. On a normally-tracked frame the very next
        # measurement update re-corrects it, so this decay is a no-op in
        # effect there; it only bites over a genuine multi-frame gap.
        # Velocity is NOT decayed here -- see the constant's comment.
        self.x[2] *= _ACCEL_DECAY_PER_STEP
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z: float):
        y = z - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T + self.R)[0, 0]
        return y, S

    def update(self, z: float):
        y, S = self.innovation(z)
        K = (self.P @ self.H.T) / S
        self.x = self.x + (K[:, 0] * y)
        self.P = (np.eye(3) - K @ self.H) @ self.P

    def reset(self, pos: float):
        self.x = np.array([pos, 0.0, 0.0])
        self.P = np.diag([_DEFAULT_MEAS_STD ** 2,
                          _INIT_VEL_STD ** 2, _INIT_ACC_STD ** 2])
        self.initialised = True

    def inflate(self, vel_var: float = 400.0, acc_var: float = 9000.0):
        """Widen velocity/acceleration uncertainty so the filter re-adapts
        quickly instead of smoothing across a discontinuity (a bounce or a
        racket contact) that the constant-acceleration model cannot represent.
        """
        self.P[1, 1] = max(self.P[1, 1], vel_var)
        self.P[2, 2] = max(self.P[2, 2], acc_var)


@dataclass
class TrackState:
    frame: int
    x: float
    y: float
    vx: float  # px/second (dt passed to the filter is real seconds)
    vy: float
    ax: float  # px/second^2
    ay: float
    source: str  # "measured" | "predicted" | "none"
    frames_since_update: int
    track_id: int
    track_age: int  # frames since this track (re)started
    pos_sigma: float = float("inf")
    """Filter's own 1-sigma positional uncertainty, sqrt(Px[0,0] + Py[0,0]) px.

    This is the honest publish/suppress criterion for a dead-reckoned
    position, and it is well calibrated: measured against the re-acquisition
    innovation on segment_000 (n=4735), log(sigma) vs log(real error)
    correlates at 0.796, and the bands are sharply separated --

        sigma 10-20px : n=3818, median error  2.8px,  2% exceed 50px
        sigma 20-40px : n= 183, median error 73.7px, 66% exceed 50px

    -- so a threshold near 20 cleanly splits trustworthy predictions from
    ones that should not drive a speed readout or a 3-D fit. Consumers get
    the raw number so they can pick their own bar.
    """


class BallKalmanTracker2D:
    """Independent constant-acceleration KF per axis, with gating and a
    dead-reckoning gap policy.

    Call :meth:`step` once per frame, in order, with the raw detector output
    for that frame (or ``None`` when the detector reported nothing). Nothing
    it does looks at any frame after the one just given.
    """

    def __init__(self, dt: float, max_gap_frames: int = 20,
                 jerk_std: float = _DEFAULT_JERK_STD,
                 meas_std: float = _DEFAULT_MEAS_STD,
                 gate_chi2: float = _GATE_CHI2,
                 frame_width: float = 1920.0, frame_height: float = 1080.0,
                 bounds_margin_px: float = _BOUNDS_MARGIN_PX):
        F, Q, H, R = _cv_matrices(dt, jerk_std, meas_std)
        self._kx = _AxisKF(F, Q, H, R)
        self._ky = _AxisKF(F, Q, H, R)
        self._max_gap = max_gap_frames
        self._gate_chi2 = gate_chi2
        self._frames_since_update = 10 ** 9
        self._track_id = 0
        self._track_age = 0
        self._x_lo = -bounds_margin_px
        self._x_hi = frame_width + bounds_margin_px
        self._y_lo = -bounds_margin_px
        self._y_hi = frame_height + bounds_margin_px

    @property
    def alive(self) -> bool:
        return self._kx.initialised and self._frames_since_update <= self._max_gap

    def _pos_sigma(self) -> float:
        """1-sigma positional uncertainty in px -- see TrackState.pos_sigma."""
        return float(np.sqrt(max(self._kx.P[0, 0], 0.0)
                             + max(self._ky.P[0, 0], 0.0)))

    def step(self, frame: int, meas_xy: Optional[tuple[float, float]]) -> TrackState:
        if not self._kx.initialised:
            if meas_xy is None:
                return TrackState(frame, float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0,
                                   "none", self._frames_since_update, self._track_id, 0)
            self._kx.reset(meas_xy[0])
            self._ky.reset(meas_xy[1])
            self._frames_since_update = 0
            self._track_id += 1
            self._track_age = 0
            return TrackState(frame, meas_xy[0], meas_xy[1], 0.0, 0.0, 0.0, 0.0,
                               "measured", 0, self._track_id, 0,
                               self._pos_sigma())

        if self._frames_since_update > self._max_gap:
            # Track has been dead-reckoning for too long: kill it. The next
            # measurement (this frame or a later one) starts a fresh track.
            self._kx.initialised = False
            return self.step(frame, meas_xy)

        self._kx.predict()
        self._ky.predict()
        self._track_age += 1

        if not (self._x_lo <= self._kx.x[0] <= self._x_hi
                and self._y_lo <= self._ky.x[0] <= self._y_hi):
            # Predicted position has left the frame by more than the margin:
            # a diverged track (see _BOUNDS_MARGIN_PX), not a tracked ball.
            # Kill it the same way an exhausted gap does, so a fresh
            # measurement (this frame or later) starts a clean track instead
            # of this one continuing to report garbage as "predicted".
            self._kx.initialised = False
            return self.step(frame, meas_xy)

        source = "predicted"
        if meas_xy is not None:
            yx, Sx = self._kx.innovation(meas_xy[0])
            yy, Sy = self._ky.innovation(meas_xy[1])
            d2 = (yx * yx) / Sx + (yy * yy) / Sy
            if d2 <= self._gate_chi2:
                self._kx.update(meas_xy[0])
                self._ky.update(meas_xy[1])
                self._frames_since_update = 0
                source = "measured"
            else:
                # Gated out as a tracking glitch (e.g. the ~250px jumps seen
                # in the raw trajectory around frames 5990-5991): treat this
                # frame as a gap rather than trusting the outlier.
                self._frames_since_update += 1
        else:
            self._frames_since_update += 1

        return TrackState(
            frame, float(self._kx.x[0]), float(self._ky.x[0]),
            float(self._kx.x[1]), float(self._ky.x[1]),
            float(self._kx.x[2]), float(self._ky.x[2]),
            source, self._frames_since_update, self._track_id, self._track_age,
            self._pos_sigma(),
        )

    def inflate_after_bounce(self):
        """Widen velocity/acceleration covariance so a confirmed bounce does
        not get smoothed into the surrounding constant-acceleration fit."""
        self._kx.inflate()
        self._ky.inflate()
