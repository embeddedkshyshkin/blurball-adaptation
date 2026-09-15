"""Causal enrichment + rendering for an existing BlurBall trajectory CSV.

Reads ``<recording>/blurBall/<segment>.csv`` (the model pipeline's own
output: ``Frame,X,Y,Visibility,L,Theta``) plus ``<recording>/calibration.json``,
and produces, causally and frame-by-frame:

  * a gap-filled, outlier-rejected image-space track (Kalman filter);
  * a physically-gated 3-D reconstruction where the data supports one
    (``ball_kinematics_3d.py``);
  * an extended CSV with world position, speed, both headings, and a
    confidence/provenance trail (task 3);
  * optionally, a re-rendered video with the always-on radar-gun HUD
    (task 1), whose direction is the screen-space Kalman heading -- see
    ``radar_hud.py`` for why that is the only source used.

This is deliberately a plain, dependency-light script -- it consumes the
BlurBall model's output rather than running the model, so it needs no CUDA,
no torch, and no Hydra config surface, and it never touches
``ball_kinematics.py`` / ``motion.py`` / ``vis.py`` / ``inference.py``, which
the existing pipeline still uses unmodified. Outputs land under a sibling
``blurBall_causal/`` directory; nothing under the existing ``blurBall/`` is
ever written to.

Usage:
    python src/enrich_and_render.py --recording-dir <path> --segment segment_000
    python src/enrich_and_render.py --recording-dir <path> --segment segment_000 --no-video
    python src/enrich_and_render.py --recording-dir <path> --segment segment_000 \\
        --frame-range 5900 6150   # render only this slice, for a quick spot-check
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.ball_tracker_kf import BallKalmanTracker2D
from utils.ball_kinematics_3d import (
    load_calibration_3d,
    detect_bounce_causal,
    fit_arc_causal,
    BounceEvent,
    MIN_FIT_WINDOW,
    MIN_FIT_WINDOW_ANCHORED,
    MAX_FIT_WINDOW,
)
from utils.radar_hud import draw_radar_hud

log = logging.getLogger("enrich_and_render")

EXTENDED_COLUMNS = [
    "Frame", "X", "Y", "Visibility", "L", "Theta",
    "Source", "TrackId", "Xf", "Yf",
    "WorldX", "WorldY", "WorldZ", "Vx", "Vy", "Vz",
    "SpeedKmh", "ScreenDirDeg", "TableHeadingDeg",
    "Confidence", "ScaleSource", "IsBounce",
]

HIST_LEN = 8
MAX_PROPAGATE_GAP_FRAMES = 90  # ~1.5s at 60fps before a held depth is distrusted
# Hard ceiling on the *reported* speed -- defense in depth, independent of
# the KF's own accel/velocity decay (ball_tracker_kf.py). No detector- or
# fit-internal failure should ever reach a CSV consumer or the HUD as a
# physically impossible number: found empirically on segment_001, where a
# long predict-only KF coast (now damped, but this stays as a floor) reached
# five-figure SpeedKmh. 150 km/h is well above any real table-tennis shot.
MAX_PLAUSIBLE_SPEED_KMH = 150.0


def _load_calibration(recording_dir: Path, fps_hint: float):
    calib_path = recording_dir / "calibration.json"
    if not calib_path.is_file():
        log.warning("No calibration.json at %s -- 3-D and physical speed are unavailable; "
                    "screen-space direction still works.", calib_path)
        return None
    raw = json.loads(calib_path.read_text())
    fps = float((raw.get("exposure") or {}).get("frameRate", fps_hint))
    try:
        return load_calibration_3d(raw, fps)
    except Exception as exc:
        log.warning("Calibration present but unusable (%s) -- falling back to screen-space only.", exc)
        return None


def process_segment(recording_dir: Path, segment: str, render_video: bool,
                     frame_range: tuple[int, int] | None, overwrite: bool) -> Path:
    csv_in = recording_dir / "blurBall" / f"{segment}.csv"
    if not csv_in.is_file():
        raise FileNotFoundError(f"Input trajectory CSV not found: {csv_in}")
    video_in = recording_dir / "segments" / f"{segment}.mov"

    out_dir = recording_dir / "blurBall_causal"
    out_dir.mkdir(exist_ok=True)
    csv_out = out_dir / f"{segment}.csv"
    if csv_out.exists() and not overwrite and frame_range is None:
        log.info("%s already exists; pass --overwrite to redo.", csv_out)
        return csv_out

    df = pd.read_csv(csv_in)
    n = len(df)
    log.info("Loaded %d rows from %s", n, csv_in)

    cap = None
    fps_hint = 60.0
    if video_in.is_file():
        cap = cv2.VideoCapture(str(video_in))
        fps_hint = cap.get(cv2.CAP_PROP_FPS) or fps_hint

    cal = _load_calibration(recording_dir, fps_hint)
    fps = cal.fps if cal is not None else fps_hint

    if cap is not None:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    else:
        # No .mov on disk (e.g. CSV-only testing): fall back to this
        # recording's known resolution. Only used for the KF's off-frame
        # divergence gate, so an approximate value here is fine.
        frame_w, frame_h = 1920, 1080
    kf = BallKalmanTracker2D(dt=1.0 / fps, frame_width=frame_w, frame_height=frame_h)

    writer = None
    if render_video and cap is not None:
        seg_dir = out_dir / "segments"
        seg_dir.mkdir(exist_ok=True)
        video_out = seg_dir / f"{segment}.mp4"
        w, h = frame_w, frame_h
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))
        log.info("Rendering annotated video to %s", video_out)

    prev_track_id = None
    hist_frames: list[int] = []
    hist_uv: list[tuple[float, float]] = []
    arc_buffer: list[tuple[int, float, float]] = []
    anchor: BounceEvent | None = None
    last_good_depth: tuple[int, float] | None = None

    rows = []
    lo, hi = frame_range if frame_range else (0, n - 1)
    t_start = time.time()

    for i in range(n):
        r = df.iloc[i]
        visible = int(r["Visibility"]) > 0
        meas = (float(r["X"]), float(r["Y"])) if visible else None
        ts = kf.step(i, meas)

        if ts.track_id != prev_track_id:
            hist_frames, hist_uv = [], []
            arc_buffer = []
            anchor = None
            last_good_depth = None
            prev_track_id = ts.track_id

        source = ts.source
        heading_rad = None
        img_speed_px = 0.0
        world = vel = None
        speed_kmh = 0.0
        table_heading = None
        scale_source = "none"
        confidence = 0.0
        is_bounce_row = False

        if source != "none":
            hist_frames.append(i)
            hist_uv.append((ts.x, ts.y))
            if len(hist_frames) > HIST_LEN:
                hist_frames.pop(0)
                hist_uv.pop(0)

            img_speed_px = math.hypot(ts.vx, ts.vy)
            if img_speed_px > 1e-6:
                heading_rad = math.atan2(ts.vy, ts.vx)

            ev = None
            if cal is not None and cal.has_pose:
                ev = detect_bounce_causal(hist_frames, hist_uv, cal)
            if ev is not None and (anchor is None or ev.frame != anchor.frame):
                anchor = ev
                kf.inflate_after_bounce()
                arc_buffer = [(f, x, y) for f, (x, y) in zip(hist_frames, hist_uv) if f >= ev.frame]
                is_bounce_row = True
            else:
                arc_buffer.append((i, ts.x, ts.y))
                if len(arc_buffer) > MAX_FIT_WINDOW:
                    arc_buffer.pop(0)

            active_anchor = anchor if (anchor is not None and arc_buffer and anchor.frame >= arc_buffer[0][0]) else None
            min_window = MIN_FIT_WINDOW_ANCHORED if active_anchor is not None else MIN_FIT_WINDOW
            if cal is not None and cal.has_pose and len(arc_buffer) >= min_window:
                frames_arr = np.array([f for f, _, _ in arc_buffer])
                uv_arr = np.array([[x, y] for _, x, y in arc_buffer])
                fit = fit_arc_causal(cal, frames_arr, uv_arr, anchor=active_anchor)
                t_now = (i - fit.t0_frame) / fps if fit.ok else None
                if fit.ok and fit.plausible(cal, t_now):
                    P_raw, V_raw = fit.state_at(t_now)
                    P_tab, V_tab = cal.to_table(P_raw), cal.to_table(V_raw)
                    _, zc = cal.project_raw(P_raw[None, :])
                    last_good_depth = (i, float(zc[0]))
                    world, vel = P_tab, V_tab
                    speed_kmh = float(np.linalg.norm(V_raw) * 3.6)
                    table_heading = float(np.degrees(np.arctan2(V_tab[1], V_tab[0])))
                    scale_source = "bounce" if active_anchor is not None else "curvature"
                    confidence = float(np.clip(1.0 - fit.rms_px / 3.0, 0.4, 1.0))

            if world is None and cal is not None and cal.has_pose:
                fx = cal.base.K[0, 0]
                if last_good_depth is not None and (i - last_good_depth[0]) <= MAX_PROPAGATE_GAP_FRAMES:
                    age = i - last_good_depth[0]
                    z_hat = last_good_depth[1]
                    scale_source = "propagated"
                    confidence = max(0.1, 0.5 * (1.0 - age / MAX_PROPAGATE_GAP_FRAMES))
                else:
                    z_hat = float(cal.table_depth_at(np.array([ts.x, ts.y]))[0])
                    scale_source = "default"
                    confidence = 0.15
                # ts.vx/vy (and hence img_speed_px) are already px/second: the
                # KF's dt is real seconds, not frames. No extra *fps here.
                speed_kmh = float(img_speed_px * z_hat / fx * 3.6)

            if abs(speed_kmh) > MAX_PLAUSIBLE_SPEED_KMH:
                # abs(), not >: a diverged/near-parallel-to-table-plane ray
                # in the propagated/default fallback (table_depth_at can
                # return a negative depth for a pixel whose ray barely
                # grazes or misses the table plane) produces a negative
                # z_hat and hence a negative speed_kmh -- equally bogus and
                # equally worth suppressing.
                # A KF or fit failure, not a fast ball -- see the constant's
                # comment. Report a track without a trustworthy speed rather
                # than a number nobody should act on.
                if scale_source in ("bounce", "curvature"):
                    # This path already passed fit.plausible() (reprojection
                    # RMS + physical accel gates) at line ~198. Getting here
                    # anyway means a gate gap, not a known KF-coast failure
                    # mode -- worth a log line rather than only silent
                    # suppression, since silently zeroing it would hide a
                    # bug in the plausibility gates themselves.
                    log.warning("frame %d: %s fit passed plausibility gates "
                                "but yielded %.1f km/h (> %.0f); clamped to 0",
                                i, scale_source, speed_kmh, MAX_PLAUSIBLE_SPEED_KMH)
                speed_kmh = 0.0
                confidence = 0.0
                world = vel = None
                table_heading = None
                scale_source = "none"

        rows.append({
            "Frame": i, "X": r["X"], "Y": r["Y"], "Visibility": r["Visibility"],
            "L": r["L"], "Theta": r["Theta"],
            "Source": source, "TrackId": ts.track_id,
            "Xf": round(ts.x, 2) if source != "none" else "",
            "Yf": round(ts.y, 2) if source != "none" else "",
            "WorldX": round(float(world[0]), 4) if world is not None else "",
            "WorldY": round(float(world[1]), 4) if world is not None else "",
            "WorldZ": round(float(world[2]), 4) if world is not None else "",
            "Vx": round(float(vel[0]), 4) if vel is not None else "",
            "Vy": round(float(vel[1]), 4) if vel is not None else "",
            "Vz": round(float(vel[2]), 4) if vel is not None else "",
            "SpeedKmh": round(speed_kmh, 2),
            "ScreenDirDeg": round(math.degrees(heading_rad), 2) if heading_rad is not None else "",
            "TableHeadingDeg": round(table_heading, 2) if table_heading is not None else "",
            "Confidence": round(confidence, 3),
            "ScaleSource": scale_source,
            "IsBounce": is_bounce_row,
        })

        if writer is not None and lo <= i <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            draw_radar_hud(frame, speed_kmh, heading_rad, source, confidence)
            if source != "none":
                colour = (60, 220, 60) if source == "measured" else (60, 160, 230)
                marker = cv2.MARKER_CROSS if source == "predicted" else -1
                if marker == -1:
                    cv2.circle(frame, (int(ts.x), int(ts.y)), 5, colour, 2, cv2.LINE_AA)
                else:
                    cv2.drawMarker(frame, (int(ts.x), int(ts.y)), colour, marker, 10, 2)
            writer.write(frame)
        elif writer is not None:
            ok = cap.grab()
            if not ok:
                break

        if i % 2000 == 0:
            log.info("frame %d/%d (%.0fs elapsed)", i, n, time.time() - t_start)

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()

    out_df = pd.DataFrame(rows, columns=EXTENDED_COLUMNS)
    out_df.to_csv(csv_out, index=False)
    log.info("Wrote %s (%d rows) in %.1fs", csv_out, len(out_df), time.time() - t_start)
    return csv_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording-dir", required=True, type=Path)
    ap.add_argument("--segment", required=True, help="e.g. segment_000 (no extension)")
    ap.add_argument("--no-video", action="store_true", help="write only the extended CSV")
    ap.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"),
                     help="render only this inclusive frame range (CSV still covers the whole segment)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)

    recording_dir = args.recording_dir.resolve()
    if not recording_dir.is_dir():
        print(f"Recording folder not found: {recording_dir}", file=sys.stderr)
        return 1

    frame_range = tuple(args.frame_range) if args.frame_range else None
    process_segment(recording_dir, args.segment, render_video=not args.no_video,
                     frame_range=frame_range, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
