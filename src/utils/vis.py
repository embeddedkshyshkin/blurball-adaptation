import os
import os.path as osp
from typing import Tuple, Optional
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import Center


def draw_frame(img_or_path, center: Center, color: Tuple, radius: int = 5, thickness: int = -1, angle=None, l=None):
    if osp.isfile(img_or_path):
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


def draw_speed_direction_hud(img, speed_kmh: Optional[float], angle_rad: Optional[float], position="top_center", arrow_length=70):
    """Draw a rotating direction arrow and current ball speed."""
    if speed_kmh is None or angle_rad is None:
        return img
    h, w = img.shape[:2]
    cx, cy = w // 2, 55 if position == "top_center" else 55
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


def gen_video(video_path, vis_dir, resize=1.0, fps=30.0, fourcc="mp4v"):
    fnames = os.listdir(vis_dir)
    fnames.sort()
    h, w, _ = cv2.imread(osp.join(vis_dir, fnames[0])).shape
    im_size = (int(w * resize), int(h * resize))
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(video_path, fourcc, fps, im_size)
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
