from __future__ import annotations

"""A fixed-position, always-on speed/direction gauge -- a radar-gun readout,
not an annotation glued to the ball.

Deliberately independent from ``vis.draw_speed_direction_hud``: this module
is new so the file actually driving today's (wrong) HUD stays untouched.

Direction here is the causal Kalman-filter image-space heading and nothing
else -- see ``ball_tracker_kf.py`` and the plan's rationale for not falling
back to a projected-3-D angle (the two sources agree when the 3-D fit is
good, but switching between them at every confidence-gate flip would make
the arrow jump exactly the way the current build already does).
"""

import cv2
import numpy as np

MAX_SPEED_KMH = 100.0
MIN_ARROW_PX = 4          # below this length we draw a point instead
POINT_RADIUS_PX = 6


def draw_radar_hud(img, speed_kmh: float, heading_rad: float | None,
                    source: str, confidence: float,
                    position: str = "top_center", max_arrow_len: int = 90) -> None:
    """Mutates ``img`` in place. Always draws something -- a point at 0 km/h,
    growing to a full arrow at ``max_arrow_len`` px by ``MAX_SPEED_KMH``.

    ``source``: "measured" | "predicted" | "none" -- from the KF track state.
    ``confidence``: 0..1, from the 3-D fit's plausibility (0 when no 3-D fit
    is trusted; the number and arrow are still shown, just visually muted).
    """
    h, w = img.shape[:2]
    scale = float(np.clip(w / 1100.0, 0.75, 1.6))
    panel_w, panel_h = int(230 * scale), int(150 * scale)
    margin = int(24 * scale)

    if position == "top_center":
        x0, y0 = (w - panel_w) // 2, margin
    elif position == "top_right":
        x0, y0 = w - panel_w - margin, margin
    else:  # top_left
        x0, y0 = margin, margin

    overlay = img[y0:y0 + panel_h, x0:x0 + panel_w].copy()
    panel = overlay.copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (18, 18, 22), -1, cv2.LINE_AA)
    blended = cv2.addWeighted(panel, 0.78, overlay, 0.22, 0)
    img[y0:y0 + panel_h, x0:x0 + panel_w] = blended
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (70, 70, 78), max(1, int(scale)), cv2.LINE_AA)

    acx, acy = x0 + panel_w // 2, y0 + int(panel_h * 0.42)
    speed_clamped = float(np.clip(speed_kmh, 0.0, MAX_SPEED_KMH))
    arrow_len = (speed_clamped / MAX_SPEED_KMH) * max_arrow_len * scale

    is_live = source != "none"
    dim = source == "predicted" or confidence < 0.35
    colour = (0, 220, 255) if not dim else (0, 150, 190)
    halo = (255, 255, 255) if not dim else (170, 170, 170)

    if not is_live or heading_rad is None or arrow_len < MIN_ARROW_PX:
        r = int(POINT_RADIUS_PX * scale)
        cv2.circle(img, (acx, acy), r, halo, -1, cv2.LINE_AA)
        cv2.circle(img, (acx, acy), max(1, r - 2), colour, -1, cv2.LINE_AA)
    else:
        half = arrow_len * 0.5
        dx, dy = np.cos(heading_rad) * half, np.sin(heading_rad) * half
        start = (int(acx - dx), int(acy - dy))
        end = (int(acx + dx), int(acy + dy))
        lw = max(2, int(round(4 * scale)))
        tip = max(8, int(round(11 * scale)))
        line_style = cv2.LINE_AA
        if source == "predicted":
            # Dashed shaft: predicted (dead-reckoned) position/heading, not a
            # real detection this frame.
            _dashed_line(img, start, end, halo, lw + 2)
            _dashed_line(img, start, end, colour, lw)
        else:
            cv2.line(img, start, end, halo, lw + 3, line_style)
            cv2.line(img, start, end, colour, lw, line_style)
        ux, uy = np.cos(heading_rad), np.sin(heading_rad)
        px, py = -uy, ux
        bx, by = end[0] - ux * tip, end[1] - uy * tip
        left = (int(bx + px * tip * 0.55), int(by + py * tip * 0.55))
        right = (int(bx - px * tip * 0.55), int(by - py * tip * 0.55))
        cv2.fillConvexPoly(img, np.array([end, left, right], dtype=np.int32), colour, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    speed_scale = 1.15 * scale
    label = f"{speed_clamped:.0f}" if is_live else "--"
    (tw, th), _ = cv2.getTextSize(label, font, speed_scale, max(2, int(3 * scale)))
    tx = x0 + (panel_w - tw) // 2
    ty = y0 + panel_h - int(14 * scale)
    text_colour = (255, 255, 255) if not dim else (190, 190, 190)
    cv2.putText(img, label, (tx, ty), font, speed_scale, text_colour, max(2, int(3 * scale)), cv2.LINE_AA)
    unit_scale = 0.5 * scale
    cv2.putText(img, "km/h", (x0 + panel_w - int(52 * scale), ty), font, unit_scale,
                (200, 200, 200), max(1, int(scale)), cv2.LINE_AA)
    if source == "predicted":
        cv2.putText(img, "predicted", (x0 + int(10 * scale), y0 + int(18 * scale)), font,
                    0.4 * scale, (150, 190, 210), 1, cv2.LINE_AA)


def _dashed_line(img, p0, p1, colour, thickness, dash_len=10, gap_len=7):
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    total = np.linalg.norm(p1 - p0)
    if total < 1e-6:
        return
    direction = (p1 - p0) / total
    pos = 0.0
    while pos < total:
        seg_end = min(pos + dash_len, total)
        a = tuple((p0 + direction * pos).astype(int))
        b = tuple((p0 + direction * seg_end).astype(int))
        cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)
        pos = seg_end + gap_len
