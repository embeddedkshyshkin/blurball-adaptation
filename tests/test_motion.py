import math

from utils.motion import MotionEstimator


def test_direction_is_stable_with_small_tracking_jitter():
    estimator = MotionEstimator(
        fps=240,
        speed_window_frames=7,
        direction_window_frames=5,
        reversal_confirm_frames=3,
    )

    directions = []
    for frame in range(20):
        x = 0.004 * frame + (0.0005 if frame % 2 else -0.0005)
        y = 0.20 + (0.0004 if frame % 3 == 0 else -0.0002)
        _, direction = estimator.update(frame, (x, y))
        if direction is not None:
            directions.append(direction)

    assert directions
    assert abs(math.degrees(directions[-1])) < 10.0


def test_reversal_requires_confirmation():
    estimator = MotionEstimator(
        fps=240,
        speed_window_frames=7,
        direction_window_frames=5,
        reversal_confirm_frames=3,
        direction_change_threshold_deg=100,
    )

    for frame in range(12):
        estimator.update(frame, (0.005 * frame, 0.2))

    before = estimator.direction_rad
    assert before is not None

    # A single noisy point in the opposite direction must not flip the arrow.
    estimator.update(12, (0.03, 0.2))
    estimator.update(13, (0.029, 0.2))
    assert estimator.direction_rad is not None
    assert abs(math.degrees(estimator.direction_rad)) < 30.0


def test_real_reversal_eventually_changes_direction():
    estimator = MotionEstimator(
        fps=240,
        speed_window_frames=5,
        direction_window_frames=5,
        reversal_confirm_frames=3,
        direction_change_threshold_deg=100,
    )

    for frame in range(12):
        estimator.update(frame, (0.005 * frame, 0.2))

    for frame in range(12, 24):
        direction = -0.005 * (frame - 12) + 0.055
        _, current = estimator.update(frame, (direction, 0.2))

    assert current is not None
    assert abs(math.degrees(current) - 180.0) < 25.0 or abs(math.degrees(current) + 180.0) < 25.0


def test_speed_uses_calibrated_distance_and_fps():
    estimator = MotionEstimator(
        fps=100,
        speed_window_frames=7,
        direction_window_frames=5,
        speed_smoothing_alpha=1.0,
    )

    speed = 0.0
    for frame in range(15):
        # 1 m/s along x => 3.6 km/h.
        speed, _ = estimator.update(frame, (frame * 0.01, 0.0))

    assert 3.0 < speed < 4.2
