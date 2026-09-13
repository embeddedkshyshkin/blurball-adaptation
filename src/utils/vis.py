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


def draw_speed_direction_hud(
    img,
    speed_kmh: float = 0.0,
    angle_rad: Optional[float] = None,
    position="top_center",
    arrow_length=72,
):
    """Draw a compact, readable speed/direction HUD.

    The direction arrow is intentionally independent from BlurBall's optical
    blur angle. ``angle_rad`` must represent the calibrated movement vector.
    """
    h, w = img.shape[:2]
    speed_kmh = max(0.0, float(speed_kmh))

    font = cv2.FONT_HERSHEY_SIMPLEX
    speed_text = f"{speed_kmh:.0f}"
    unit_text = " km/h"
    speed_scale = max(0.75, min(1.25, w / 1100.0))
    unit_scale = speed_scale * 0.55
    speed_thickness = max(2, int(round(2.5 * speed_scale)))
    unit_thickness = max(1, int(round(1.7 * speed_scale)))

    (sw, sh), _ = cv2.getTextSize(speed_text, font, speed_scale, speed_thickness)
    (uw, uh), _ = cv2.getTextSize(unit_text, font, unit_scale, unit_thickness)

    gap = max(4, int(7 * speed_scale))
    text_w = sw + gap + uw
    text_h = max(sh, uh)

    arrow_space = int(arrow_length + 24) if angle_rad is not None else 0
    panel_w = max(text_w + 42, arrow_space + 34)
    panel_h = text_h + 30 + (arrow_space if angle_rad is not None else 0)

    if position == "top_left":
        x0, y0 = 22, 18
    elif position == "top_right":
        x0, y0 = w - panel_w - 22, 18
    else:
        x0, y0 = (w - panel_w) // 2, 18

    x0 = max(8, min(x0, w - panel_w - 8))
    y0 = max(8, min(y0, h - panel_h - 8))

    overlay = img.copy()
    radius = max(10, int(14 * speed_scale))
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x0, y0), (x0 + panel_w, y0 + panel_h), 255, -1)
    cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0, dst=mask)
    img[:] = np.where(mask[..., None] > 0, cv2.addWeighted(overlay, 0.76, img, 0.24, 0), img)

    if angle_rad is not None:
        acx = x0 + panel_w // 2
        acy = y0 + int(arrow_space * 0.5)
        half = arrow_length * 0.5
        dx = np.cos(angle_rad) * half
        dy = np.sin(angle_rad) * half
        start = (int(acx - dx), int(acy - dy))
        end = (int(acx + dx), int(acy + dy))
        line_width = max(3, int(round(4 * speed_scale)))
        tip = max(9, int(round(12 * speed_scale)))
        cv2.line(img, start, end, (255, 255, 255), line_width + 3, cv2.LINE_AA)
        cv2.line(img, start, end, (0, 220, 255), line_width, cv2.LINE_AA)
        # A compact custom arrowhead gives a cleaner result than a large
        # arrowedLine tip and remains visually stable at 120/240 FPS.
        ux, uy = np.cos(angle_rad), np.sin(angle_rad)
        px, py = -uy, ux
        base_x, base_y = end[0] - ux * tip, end[1] - uy * tip
        left = (int(base_x + px * tip * 0.55), int(base_y + py * tip * 0.55))
        right = (int(base_x - px * tip * 0.55), int(base_y - py * tip * 0.55))
        cv2.fillConvexPoly(img, np.array([end, left, right], dtype=np.int32), (0, 220, 255), cv2.LINE_AA)

    text_y = y0 + panel_h - max(10, int(12 * speed_scale))
    text_x = x0 + (panel_w - text_w) // 2
    cv2.putText(img, speed_text, (text_x, text_y), font, speed_scale, (255, 255, 255), speed_thickness, cv2.LINE_AA)
    cv2.putText(img, unit_text, (text_x + sw + gap, text_y), font, unit_scale, (220, 220, 220), unit_thickness, cv2.LINE_AA)
    return img


# Common nominal camera/video frame rates. Container metadata can expose a
# mathematically equivalent but unsuitable timebase (e.g. 1000/240229),
# which OpenCV's MPEG-4 writer rejects because the timebase denominator is
# larger than the MPEG-4 limit.
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
    """Return a sane nominal FPS for video output."""
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
