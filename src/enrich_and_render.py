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
    # whole recording, CSV only -- every segment that has a BlurBall CSV
    python src/enrich_and_render.py --recording-dir <path> --no-video

    # whole recording, CSV + HUD video (slower: decodes and re-encodes)
    python src/enrich_and_render.py --recording-dir <path>

    # named segment(s) only
    python src/enrich_and_render.py --recording-dir <path> --segment segment_000
    python src/enrich_and_render.py --recording-dir <path> \\
        --segment segment_000 segment_001 --no-video

    # re-render one slice for a quick spot-check (CSV still covers the whole
    # segment; --frame-range only limits the video)
    python src/enrich_and_render.py --recording-dir <path> --segment segment_000 \\
        --frame-range 5900 6150

Existing outputs are skipped unless ``--overwrite`` is given, so a batch can
be re-run as the detector finishes more segments and only the new ones cost
anything.
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
    "Confidence", "ScaleSource", "IsBounce", "BounceConfirmFrame", "PosSigmaPx",
    "SpeedKmhRaw",
]

# Above this the Kalman filter's own positional uncertainty says its
# dead-reckoned position is not worth building anything on. Measured on
# segment_000 against an independent detector (the desktop app's auto-label
# sightings) and against re-acquisition innovations (n=4735):
#
#   first predicted frame  (sigma ~11) : median error   3.0px, 94% within 25px
#   second predicted frame (sigma ~30) : median error 109.6px, 17% within 25px
#
# The cliff is real and it is not a model-choice artefact: coasting at
# constant velocity instead of constant acceleration makes it worse (185.8px
# at the second frame), because dropouts coincide with motion no kinematic
# model can follow -- a blurred smash is ~130px/frame, and a racket contact
# inside the gap is a velocity discontinuity by definition.
#
# So a row past this bar still reports its position (the user explicitly
# asked for predicted positions in the CSV, marked as predicted), but it no
# longer feeds a speed, a 3-D fit, a bounce search, or an on-video arrow.
MAX_TRUSTED_POS_SIGMA_PX = 20.0

# How far a detector CSV may fall short of the manifest's frame count before
# it is treated as a half-written file rather than an end-of-stream quirk.
# The two disagree by one frame legitimately -- segment_012 of the reference
# recording has 10820 detector rows against a manifest frameCount of 10821,
# with the detector finished -- while a CSV caught mid-write is short by
# hundreds or thousands. 5 separates those without denying a good segment.
MANIFEST_ROW_TOLERANCE = 5

# Causal exponential smoothing of the *reported* speed.
#
# The raw number jitters +-20% frame to frame on a smoothly-varying arc,
# which reads as broken on the video even when the mean is right. Measured on
# the 5992-6021 lob: SpeedKmh and img_speed_px have identical variation
# (cv 0.095, 7.1% median frame-to-frame) while z_hat is flat to 3 decimal
# places (3.034-3.035 m), so every bit of the jitter is KF velocity noise and
# none of it is the depth path.
#
# The filter itself is left alone: detuning Q to smooth velocity costs real
# detections (see _GATE_CHI2's note), so the smoothing belongs on the readout.
# alpha=0.3 is an effective ~3.3-frame window, cutting the jitter to about
# 4% -- as smooth as the most detuned filter tried, with no coverage lost --
# for ~2.3 frames (39ms) of lag. It is reset on a track change, on a
# confirmed bounce and on any untrusted frame, so it never smooths across a
# genuine velocity discontinuity; that lag-at-a-bounce was the original
# complaint about the old non-causal estimator and is not worth reintroducing.
# SpeedKmhRaw keeps the unsmoothed value for anyone who wants it.
SPEED_EMA_ALPHA = 0.3

HIST_LEN = 8
MAX_PROPAGATE_GAP_FRAMES = 90  # ~1.5s at 60fps before a held depth is distrusted
# Hard ceiling on the *reported* speed -- defense in depth, independent of
# the KF's own accel/velocity decay (ball_tracker_kf.py). No detector- or
# fit-internal failure should ever reach a CSV consumer or the HUD as a
# physically impossible number: found empirically on segment_001, where a
# long predict-only KF coast (now damped, but this stays as a floor) reached
# five-figure SpeedKmh. 150 km/h is well above any real table-tennis shot.
MAX_PLAUSIBLE_SPEED_KMH = 150.0


def _manifest_frame_count(recording_dir: Path, segment: str) -> int | None:
    """Frame count this segment should have, from manifest.json, or None if
    the manifest is missing/unreadable/silent about it."""
    manifest_path = recording_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None
    for seg in manifest.get("segments", []):
        name = str(seg.get("fileName", ""))
        if name.rsplit(".", 1)[0] == segment:
            count = seg.get("frameCount")
            return int(count) if isinstance(count, (int, float)) else None
    return None


def _video_looks_complete(video_out: Path, recording_dir: Path, segment: str) -> bool:
    """True when an existing rendered video covers the whole segment.

    A render interrupted part-way (Ctrl-C, OOM, a full disk) leaves a short
    but perfectly readable mp4. Treating "the file exists" as "the video is
    done" would make that partial result stick, silently, for every later
    run -- and a whole-recording render is ~30 minutes, so being interrupted
    is not a remote possibility. One frame-count check costs nothing on the
    skip path and is the same reasoning as the manifest check on inputs.
    """
    if not video_out.is_file():
        return False
    expected = _manifest_frame_count(recording_dir, segment)
    if expected is None:
        return True  # nothing to compare against; take the file at face value
    cap = cv2.VideoCapture(str(video_out))
    try:
        if not cap.isOpened():
            return False
        have = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if expected - have > MANIFEST_ROW_TOLERANCE:
        log.info("%s: existing video has %d of %d frames -- incomplete, "
                 "re-rendering.", segment, have, expected)
        return False
    return True


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
    video_out = out_dir / "segments" / f"{segment}.mp4"
    # Skip only when everything this invocation was asked for already exists.
    # Keying the skip on the CSV alone silently broke the natural sequence of
    # "enrich the whole recording with --no-video, then render the videos":
    # the second pass saw the CSVs, skipped every segment and produced no
    # video at all, exit 0. Asking for the video now re-runs the pass even
    # though the CSV is there, and asking only for the CSV still costs
    # nothing on a second run.
    have_wanted = csv_out.exists() and (not render_video
                                        or _video_looks_complete(video_out, recording_dir, segment))
    if have_wanted and not overwrite and frame_range is None:
        log.info("%s already has the requested output(s); pass --overwrite "
                 "to redo.", segment)
        return csv_out

    df = pd.read_csv(csv_in)
    n = len(df)
    log.info("Loaded %d rows from %s", n, csv_in)

    # Refuse a half-written input. The normal workflow is to enrich a whole
    # recording while the detector is still working through it, so a segment's
    # CSV may be mid-write when the glob picks it up. pandas reads a truncated
    # file without complaint, and the short result would then be skipped on
    # the next run (it exists), making the bad output sticky. manifest.json
    # states each segment's true frame count, so compare and skip instead.
    expected = _manifest_frame_count(recording_dir, segment)
    if expected is not None and expected - n > MANIFEST_ROW_TOLERANCE:
        log.warning("%s: %d rows but manifest says %d frames -- input looks "
                    "incomplete (detector still running?); skipping. Re-run "
                    "once it has finished.", segment, n, expected)
        return csv_out
    if expected is not None and n != expected:
        log.info("%s: %d rows vs manifest %d -- within tolerance, proceeding.",
                 segment, n, expected)

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
        video_out.parent.mkdir(exist_ok=True)
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
    speed_ema: float | None = None

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
            speed_ema = None
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

        # A measured frame is trustworthy by construction; a dead-reckoned one
        # only while the filter's own uncertainty stays under the bar. An
        # untrusted row is still emitted with its position and PosSigmaPx --
        # it just stops propagating into anything downstream, including the
        # history that feeds bounce detection and the 3-D arc buffer. Letting
        # a 110px-wrong position into those was what put a visibly wrong ball
        # and a jumping speed on the rendered video.
        trusted = source == "measured" or (
            source == "predicted" and ts.pos_sigma <= MAX_TRUSTED_POS_SIGMA_PX)

        if trusted:
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
                # Mark the bounce on the frame it actually happened on
                # (ev.frame, the image-v vertex), not on the frame we found
                # out about it. detect_bounce_causal deliberately reports a
                # lagged event -- it needs BOUNCE_CONFIRM_LAG frames on the
                # far side of the peak before it can tell a bounce from
                # noise -- so `i` here is the *confirmation* frame, typically
                # vertex+2 but further when the history has gaps. Marking `i`
                # put every bounce visibly late against the video, measured
                # as a +2 mode (63 of 86 matched bounces on segment_000)
                # versus the image-v vertex in an independent detector's
                # trace. Backfilling by ev.frame rather than a constant -2
                # also gets the gappy cases right.
                #
                # This backfills a marker into an already-computed row, which
                # is fine precisely because it is only a marker: the CSV is
                # written after the whole pass, and none of the kinematic
                # columns are touched. The anchor itself is still applied
                # strictly forward from this frame, so speed/direction/3-D on
                # every row stay causal.
                if 0 <= ev.frame < len(rows):
                    rows[ev.frame]["IsBounce"] = True
                    rows[ev.frame]["BounceConfirmFrame"] = i
                else:
                    is_bounce_row = True
                # Speed genuinely steps across a bounce; don't smooth over it.
                speed_ema = None
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
                if not math.isfinite(z_hat) or z_hat <= 0.0:
                    # The ray through this pixel never meets the table plane
                    # in front of the camera -- the ball is above the table's
                    # horizon in the image, so this depth prior simply does
                    # not apply. It is undefined, not negative: multiplying
                    # by it produced 200 rows of negative km/h across
                    # segments 000/001, every one of them on a perfectly
                    # tracked frame (sigma 2.74), and the magnitude clamp
                    # below only caught the ones past +-150. Report no speed
                    # rather than a sign-flipped one.
                    scale_source = "none"
                    confidence = 0.0
                else:
                    # ts.vx/vy (and hence img_speed_px) are already px/second:
                    # the KF's dt is real seconds, not frames. No extra *fps.
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

        # Causal EMA on the reported speed -- see SPEED_EMA_ALPHA. Only a
        # trusted row with a real speed feeds it; anything else drops the
        # state so the next rally starts clean instead of easing out of a
        # stale value.
        speed_kmh_raw = speed_kmh
        if trusted and speed_kmh > 0.0:
            speed_ema = (speed_kmh if speed_ema is None
                         else SPEED_EMA_ALPHA * speed_kmh
                         + (1.0 - SPEED_EMA_ALPHA) * speed_ema)
            speed_kmh = speed_ema
        else:
            speed_ema = None

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
            "BounceConfirmFrame": "",
            "PosSigmaPx": round(ts.pos_sigma, 2) if math.isfinite(ts.pos_sigma) else "",
            "SpeedKmhRaw": round(speed_kmh_raw, 2),
        })

        if writer is not None and lo <= i <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            # scale_source "none" means no depth was usable this frame, so
            # there is no speed to show -- read "--" rather than "0", same
            # reasoning as a suppressed prediction.
            draw_radar_hud(frame, speed_kmh, heading_rad, source, confidence,
                            trusted=trusted and scale_source != "none")
            # Only mark a position the filter can actually defend. Gating on
            # `source != "none"` drew a cross on every dead-reckoned frame,
            # including the ones whose position is a median 369px from where
            # an independent detector puts the ball -- that stray cross
            # wandering off the ball is what "predicted balls is quite not
            # accurate" was describing. An untrusted frame now draws no
            # marker at all and the gauge collapses to a point, which is the
            # honest statement: we do not know where the ball is.
            if trusted:
                colour = (60, 220, 60) if source == "measured" else (60, 160, 230)
                if source == "measured":
                    cv2.circle(frame, (int(ts.x), int(ts.y)), 5, colour, 2, cv2.LINE_AA)
                else:
                    cv2.drawMarker(frame, (int(ts.x), int(ts.y)), colour,
                                    cv2.MARKER_CROSS, 10, 2)
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
    ap.add_argument("--segment", nargs="+", metavar="NAME",
                     help="one or more segment names, e.g. --segment segment_000 "
                          "segment_001 (no extension). Omit to process every "
                          "segment in the recording that has a BlurBall CSV.")
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

    if args.segment:
        segments = list(args.segment)
    else:
        # Whole-recording batch: every segment the detector has produced a
        # trajectory for. Segments still queued in the detector simply are
        # not there yet, so this picks up whatever is ready and can be
        # re-run later to fill in the rest.
        segments = sorted(p.stem for p in (recording_dir / "blurBall").glob("*.csv"))
        if not segments:
            print(f"No BlurBall CSVs found in {recording_dir / 'blurBall'}",
                  file=sys.stderr)
            return 1
        log.info("batch: %d segment(s) with a BlurBall CSV: %s",
                 len(segments), ", ".join(segments))

    failures = []
    for name in segments:
        try:
            process_segment(recording_dir, name, render_video=not args.no_video,
                             frame_range=frame_range, overwrite=args.overwrite)
        except Exception as exc:  # one bad segment must not lose the batch
            log.error("segment %s failed: %s", name, exc, exc_info=True)
            failures.append(name)

    if failures:
        print(f"{len(failures)} of {len(segments)} segment(s) failed: "
              f"{', '.join(failures)}", file=sys.stderr)
        return 1
    if len(segments) > 1:
        log.info("batch complete: %d segment(s)", len(segments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
