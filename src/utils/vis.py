import os
import os.path as osp
from typing import Tuple, Optional
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import Center


def draw_frame(img_or_path, center: Center, color: Tuple, radius: int = 5, thickness: int = -1, angle=None, l=None):
    if isinstance(img_or_path, np.ndarray):
        img = img_or_path
    elif isinstance(img_or_path, (str, bytes, os.PathLike)) and osp.isfile(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path
    xy = center.xy
    visi = center.is_visible
    if visi:
        x, y = map(int, xy)
        img = cv2.circle(img, (x, y), radius, color, thickness=thickness)
        if angle is not None and l != 0:
            angle_rad = np.deg2rad(angle)
            x1 = int(x + l * np.cos(angle_rad))
            y1 = int(y + l * np.sin(angle_rad))
            x2 = int(x - l * np.cos(angle_rad))
            y2 = int(y - l * np.sin(angle_rad))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def draw_speed_direction_hud(img, speed_kmh: float = 0.0, angle_rad: Optional[float] = None, position="top_center", arrow_length=70):
    """Draw the current ball speed and, when reliable, the movement direction."""
    h, w = img.shape[:2]
    cx, cy = w // 2, 55 if position == "top_center" else 55

    if angle_rad is not None:
        half = arrow_length * 0.5
        x1 = int(cx - half * np.cos(angle_rad))
        y1 = int(cy - half * np.sin(angle_rad))
        x2 = int(cx + half * np.cos(angle_rad))
        y2 = int(cy + half * np.sin(angle_rad))
        cv2.arrowedLine(img, (x1, y1), (x2, y2), (0, 255, 255), 4, tipLength=0.3)

    label = f"{speed_kmh:.0f} km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.0, 2
    (tw, _), _ = cv2.getTextSize(label, font, scale, thickness)
    tx, ty = cx - tw // 2, cy + 42
    cv2.putText(img, label, (tx + 2, ty + 2), font, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


# Common nominal camera/video frame rates. Container metadata can expose a
# mathematically equivalent but unsuitable timebase (e.g. 1000/240229),
# which OpenCV's MPEG-4 writer rejects because the timebase denominator is
# larger than the MPEG-4 limit. Snap only when the reported FPS is very close
# to a known nominal rate; otherwise preserve the reported value.
_NOMINAL_FPS = (
    23.976,
    24.0,
    25.0,
    29.97,
    30.0,
    50.0,
    59.94,
    60.0,
    100.0,
    119.88,
    120.0,
    200.0,
    239.76,
    240.0,
)


def normalize_video_fps(fps: float, tolerance: float = 0.5) -> float:
    """Return a sane nominal FPS for video output.

    Some MOV/MP4 containers expose FPS values such as 240.229 because of a
    pathological source timebase. OpenCV can read that value but may fail to
    create an MPEG-4 VideoWriter from it. If the value is sufficiently close
    to a standard nominal camera rate, use that nominal rate instead.
    """
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps!r}")

    nominal = min(_NOMINAL_FPS, key=lambda value: abs(value - fps))
    if abs(nominal - fps) <= tolerance:
        return nominal
    return fps


def gen_video(video_path, vis_dir, resize=1.0, fps=30.0, fourcc="mp4v"):
    fnames = os.listdir(vis_dir)
    fnames.sort()
    h, w, _ = cv2.imread(osp.join(vis_dir, fnames[0])).shape
    im_size = (int(w * resize), int(h * resize))

    normalized_fps = normalize_video_fps(fps)
    if normalized_fps != float(fps):
        print(f"Normalizing output video FPS: {float(fps):.6f} -> {normalized_fps:.6f}")

    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(video_path, fourcc, normalized_fps, im_size)
    if not out.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for {video_path!r} at {normalized_fps:.6f} FPS"
        )

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im = cv2.imread(im_path)
        if im is not None:
            im = cv2.resize(im, None, fx=resize, fy=resize)
            out.write(im)
        else:
            print("COuldn't read image")
            print(fname)
    out.release()
