"""Calibration helpers shared by the gravity-constrained ball solver.

The solver works in PongEye's table frame: ``+x`` along the table, ``+y``
across it, and ``+z`` away from the playing surface.  This module deliberately
only normalises calibration input; it does not choose or tune a motion model.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibration:
    H: np.ndarray              # table metres (z=0) -> image pixels
    H_inv: np.ndarray          # image pixels -> table metres (z=0)
    K: np.ndarray | None       # table-frame camera intrinsics
    R: np.ndarray | None       # table frame -> camera frame rotation
    t: np.ndarray | None       # table-frame camera translation
    cam_centre: np.ndarray | None
    image_size: tuple[int, int]
    length_m: float
    width_m: float
    px_per_m: float

    @property
    def has_pose(self) -> bool:
        return self.K is not None and self.R is not None and self.t is not None


def _as_matrix(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 9:
        raise ValueError(f"calibration JSON needs a 9-element '{name}'")
    return np.asarray(values, dtype=float).reshape(3, 3)


def _pongeye_homography(corners, length_m, width_m):
    """Build table -> image H from PongEye's named table corners.

    PongEye's extrinsics define bottomLeft as table origin, not topLeft.
    Keeping that order makes the resulting plane agree with the supplied pose.
    """
    image_points = np.asarray([
        [corners["bottomLeft"]["x"], corners["bottomLeft"]["y"]],
        [corners["bottomRight"]["x"], corners["bottomRight"]["y"]],
        [corners["topRight"]["x"], corners["topRight"]["y"]],
        [corners["topLeft"]["x"], corners["topLeft"]["y"]],
    ], dtype=np.float64)
    table_points = np.asarray([
        [0.0, 0.0], [length_m, 0.0],
        [length_m, width_m], [0.0, width_m],
    ], dtype=np.float64)
    return cv2.getPerspectiveTransform(table_points.astype(np.float32), image_points.astype(np.float32))


def _pose_from_homography(K, H):
    """Recover table-frame pose for the older homography-only schema."""
    M = np.linalg.inv(K) @ H
    scale = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * scale, M[:, 1] * scale, M[:, 2] * scale
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack((r1, r2, np.cross(r1, r2)))
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return R, t


def load_calibration(path: str | Path) -> Calibration:
    """Load either the original solver schema or PongEye's recording schema."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    cal = raw.get("calibration", raw)
    dims = cal.get("tableDimensions", {})
    length_m = float(dims.get("lengthMeters", 2.74))
    width_m = float(dims.get("widthMeters", 1.525))
    px_per_m = float(raw.get("pixelsPerMetreAtTableCentre",
                             cal.get("pixelsPerMetreAtTableCentre", float("nan"))))

    if cal.get("homographyTableToImage") is not None:
        H = _as_matrix(cal["homographyTableToImage"], "homographyTableToImage")
    elif cal.get("corners") is not None:
        H = _pongeye_homography(cal["corners"], length_m, width_m)
    else:
        raise ValueError("calibration JSON needs 'homographyTableToImage' or 'calibration.corners'")
    H /= H[2, 2]

    # Recordings written by PongEye have used both top-level ``exposure`` and
    # ``device.exposure``. They carry the same camera calibration fields.
    exposure = raw.get("exposure") or raw.get("device", {}).get("exposure", {})
    if not np.isfinite(px_per_m):
        px_per_m = float(exposure.get("pixelsPerMetreAtTableCentre", float("nan")))
    image = raw.get("imageSize", {})
    image_size = (
        int(exposure.get("width", image.get("width", 1920))),
        int(exposure.get("height", image.get("height", 1080))),
    )

    K = R = t = centre = None
    intr = raw.get("solvedIntrinsics") or exposure.get("extrinsicsIntrinsics") or {}
    if intr:
        fx = intr.get("focalLengthXPx", intr.get("focalLengthPx"))
        fy = intr.get("focalLengthYPx", intr.get("focalLengthPx", fx))
        if fx is not None and fy is not None:
            K = np.array([
                [float(fx), 0.0, float(intr.get("principalPointXPx", image_size[0] / 2))],
                [0.0, float(fy), float(intr.get("principalPointYPx", image_size[1] / 2))],
                [0.0, 0.0, 1.0],
            ])
            extrinsics = exposure.get("extrinsics") or raw.get("extrinsics") or {}
            if extrinsics.get("rotation") is not None and extrinsics.get("translationMetres") is not None:
                R = _as_matrix(extrinsics["rotation"], "rotation")
                t = np.asarray(extrinsics["translationMetres"], dtype=float)
                if t.shape != (3,):
                    raise ValueError("calibration JSON needs a 3-element 'translationMetres'")
            else:
                R, t = _pose_from_homography(K, H)
            centre = -R.T @ t
            declared_centre = extrinsics.get("cameraPositionMetres")
            if declared_centre is not None:
                declared_centre = np.asarray(declared_centre, dtype=float)
                if declared_centre.shape != (3,):
                    raise ValueError("calibration JSON needs a 3-element 'cameraPositionMetres'")
                # The declared centre is a useful integrity check, but R/t are
                # authoritative because they are the projection pose.
                if not np.allclose(centre, declared_centre, atol=1e-3):
                    raise ValueError("PongEye camera pose is inconsistent: -R.T @ t != cameraPositionMetres")

    return Calibration(H, np.linalg.inv(H), K, R, t, centre, image_size,
                       length_m, width_m, px_per_m)


def pixels_to_height_plane(calibration: Calibration, uv, height_m: float):
    """Back-project pixels to a horizontal table-frame plane.

    This is the pose interface consumed by the existing 3-D solver.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    if height_m == 0.0 or not calibration.has_pose:
        p = np.c_[uv, np.ones(len(uv))] @ calibration.H_inv.T
        return np.c_[p[:, :2] / p[:, 2:3], np.full(len(p), height_m)]
    directions_camera = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(calibration.K).T
    directions_world = directions_camera @ calibration.R
    scale = (height_m - calibration.cam_centre[2]) / directions_world[:, 2]
    return calibration.cam_centre + scale[:, None] * directions_world
