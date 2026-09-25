"""RAM-resident inference pipeline: extract -> Queue A -> model -> Queue B -> post -> CSV.

For one video segment, three daemon threads overlap frame extraction, model
forward+transfer, and blob-detection/tracker/CSV writing instead of running
them strictly sequentially through a directory of extracted PNGs (see
``inference_video`` in ``inference.py`` for the disk-based baseline this
mirrors).

For several segments (``run_ram_pipeline_segments``), one extract thread and
one post thread run per segment, but all segments share a single model
thread and GPU -- not one model copy per segment. Queue A is shared across
the active segments (items are tagged with ``seg_id``); each segment keeps
its own Queue B so a segment's post thread never has to filter another
segment's output.

Preprocessing (affine transform, resize/normalize) and the windowing/step
logic are copied 1:1 from ``inference_video`` so the CSV this produces
matches the disk-based path frame for frame. The postprocessor math
(x/y/L/theta) itself is untouched: the model thread does sigmoid + GPU->CPU
transfer only (``detector.to_heatmaps``) and hands heatmaps to the post
thread, which runs blob detection (``detector.results_from_heatmaps``) --
the same CPU-bound work ``detector.run_tensor`` does inline, just moved off
the shared GPU thread so it doesn't compete with the next window's forward
pass, and so it can run concurrently across segments.

A single-segment error (bad video, a step outside 1/3, an exception inside
the detector) is isolated to that segment: its own extract/post threads stop
and no CSV is written for it, but other segments in the same batch keep
running. A global ``RunControl.cancel_event`` -- the user's Cancel button --
stops every segment.
"""
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from trackers import build_tracker
from utils.image import get_affine_transform

_SENTINEL_KEY = "__sentinel__"


class RunControl:
    """Thread-safe pause/cancel flags shared by the pipeline threads."""

    def __init__(self):
        self.pause_event = threading.Event()
        self.pause_event.set()  # set = running, clear = paused
        self.cancel_event = threading.Event()

    def wait_if_paused(self):
        while not self.pause_event.is_set() and not self.cancel_event.is_set():
            self.pause_event.wait(timeout=0.2)

    @property
    def cancelled(self):
        return self.cancel_event.is_set()


class _SegmentControl:
    """A ``RunControl``-shaped view scoped to one segment.

    ``cancelled`` is true if the user cancelled the whole run *or* this
    segment failed on its own; either way the segment's threads should stop
    without writing a CSV. ``local_cancel`` never touches the shared
    ``RunControl``, so one segment's failure doesn't stop its siblings.
    """

    def __init__(self, run_control):
        self._run_control = run_control
        self._local_stop = threading.Event()

    @property
    def cancelled(self):
        return self._run_control.cancelled or self._local_stop.is_set()

    def wait_if_paused(self):
        self._run_control.wait_if_paused()

    def local_cancel(self):
        self._local_stop.set()


def _sentinel(seg_id):
    return {_SENTINEL_KEY: True, "seg_id": seg_id}


def _is_sentinel(item):
    return isinstance(item, dict) and item.get(_SENTINEL_KEY, False)


def _put_with_cancel(q, item, control, timeout=0.2):
    while not control.cancelled:
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _get_with_cancel(q, control, timeout=0.2):
    while not control.cancelled:
        try:
            return q.get(timeout=timeout), True
        except queue.Empty:
            continue
    return None, False


def _put_best_effort(q, item, timeout=1.0):
    """Put ignoring any cancel flag -- used for sentinels, which must be
    delivered even when the sender just cancelled itself (local or global),
    since the receiver's bookkeeping (e.g. active_seg_ids) depends on it."""
    try:
        q.put(item, timeout=timeout)
        return True
    except queue.Full:
        return False


def _build_preprocess(inp_height, inp_width):
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((inp_height, inp_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_affine(w, h, inp_width, inp_height, frames_in):
    c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = max(h, w) * 1.0
    trans = np.stack(
        [get_affine_transform(c, s, 0, [inp_width, inp_height], inv=1) for _ in range(frames_in)],
        axis=0,
    )
    return torch.tensor(trans)[None, :]


def _extract_worker(seg_id, video_path, cfg, seg_control, queue_a, error_box, timing):
    frames_in = cfg["model"]["frames_in"]
    step = cfg["detector"]["step"]
    if step not in (1, 3):
        error_box[seg_id] = ValueError(f"Unsupported detector.step={step}; expected 1 or 3")
        seg_control.local_cancel()
        _put_best_effort(queue_a, _sentinel(seg_id))
        return
    inp_h, inp_w = cfg["model"]["inp_height"], cfg["model"]["inp_width"]
    preprocess = _build_preprocess(inp_h, inp_w)

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timing["total_frames"] = total_frames if total_frames > 0 else None
        trans = _build_affine(w, h, inp_w, inp_h, frames_in)

        frames_buffer = []
        idx_buffer = []
        frame_idx = 0
        while not seg_control.cancelled:
            seg_control.wait_if_paused()
            if seg_control.cancelled:
                break
            t0 = time.time()
            ret, frame = cap.read()
            timing["t_extract"] += time.time() - t0
            if not ret:
                break
            frames_buffer.append(frame)
            idx_buffer.append(frame_idx)
            frame_idx += 1
            if len(frames_buffer) == frames_in:
                t0 = time.time()
                tensors = [preprocess(f) for f in frames_buffer]
                input_tensor = torch.cat(tensors, dim=0).unsqueeze(0)
                timing["t_extract"] += time.time() - t0
                item = {
                    "seg_id": seg_id,
                    "frame_indices": list(idx_buffer),
                    "tensor": input_tensor,
                    "affine": trans,
                }
                if not _put_with_cancel(queue_a, item, seg_control):
                    break
                if step == 1:
                    frames_buffer.pop(0)
                    idx_buffer.pop(0)
                else:  # step == 3
                    frames_buffer = []
                    idx_buffer = []
    except Exception as exc:
        error_box[seg_id] = exc
        seg_control.local_cancel()
    finally:
        if cap is not None:
            cap.release()
        _put_best_effort(queue_a, _sentinel(seg_id))


def _model_worker_shared(detector, control, queue_a, queue_b_map, seg_controls, active_seg_ids, error_box, timing):
    """GPU-only: forward pass + sigmoid/transfer, shared across all active
    segments' windows. Blob detection runs downstream in each segment's post
    thread so this thread spends less time per window and never blocks on
    one segment while others have work waiting (Queue A is shared/FIFO)."""
    try:
        with torch.no_grad():
            while active_seg_ids and not control.cancelled:
                item, ok = _get_with_cancel(queue_a, control)
                if not ok:
                    break  # global cancel
                seg = item["seg_id"]
                if _is_sentinel(item):
                    # Forward the completion signal so the post thread knows
                    # to flush and write its CSV, rather than treat this as
                    # a cancel/error (which skips the CSV entirely).
                    if seg in queue_b_map:
                        _put_best_effort(queue_b_map[seg], item)
                    active_seg_ids.discard(seg)
                    continue
                if seg not in active_seg_ids:
                    continue  # stale item from an already-failed/cancelled segment
                seg_ctrl = seg_controls[seg]
                if seg_ctrl.cancelled:
                    active_seg_ids.discard(seg)
                    continue
                try:
                    t0 = time.time()
                    hms, affine_np = detector.to_heatmaps(item["tensor"], item["affine"])
                    timing[seg]["t_model"] += time.time() - t0
                except Exception as exc:
                    error_box[seg] = exc
                    seg_ctrl.local_cancel()
                    active_seg_ids.discard(seg)
                    continue
                out_item = {
                    "seg_id": seg,
                    "frame_indices": item["frame_indices"],
                    "hms": hms,
                    "affine_np": affine_np,
                }
                if not _put_with_cancel(queue_b_map[seg], out_item, seg_ctrl):
                    active_seg_ids.discard(seg)
    except Exception as exc:
        # A failure in the shared plumbing itself (not attributable to one
        # segment's data) has to stop everyone -- no segment can make
        # progress without this thread.
        error_box["__model__"] = exc
        control.cancel_event.set()


def _post_worker_seg(detector, cfg, seg_control, queue_b, csv_path, error_box, timing, seg_id, progress_cb=None):
    """CPU-only: blob detection over heatmaps from Queue B, then tracker + CSV."""
    step = cfg["detector"]["step"]
    model_name = cfg["model"]["name"]
    tracker = build_tracker(cfg)
    tracker.refresh()
    pending = {}
    rows = []

    def finalize(fidx):
        preds = pending.pop(fidx, [])
        t0 = time.time()
        result = tracker.update(preds)
        timing["t_post"] += time.time() - t0
        row = {
            "Frame": fidx,
            "X": int(min(max(result["x"], 0), 100000)),
            "Y": int(min(max(result["y"], 0), 100000)),
            "Visibility": int(result["visi"]),
        }
        if model_name == "blurball":
            row["L"] = result["length"]
            row["Theta"] = result["angle"]
        rows.append(row)
        if progress_cb is not None:
            progress_cb(seg_id, len(rows), timing.get("total_frames"))

    try:
        while not seg_control.cancelled:
            item, ok = _get_with_cancel(queue_b, seg_control)
            if not ok or _is_sentinel(item):
                break
            seg_control.wait_if_paused()
            if seg_control.cancelled:
                break
            frame_indices = item["frame_indices"]
            t0 = time.time()
            batch_results, _hms_vis = detector.results_from_heatmaps(item["hms"], item["affine_np"])
            timing["t_post"] += time.time() - t0
            preds_per_pos = [batch_results[0].get(pos, []) for pos in range(len(frame_indices))]
            for pos, fidx in enumerate(frame_indices):
                pending.setdefault(fidx, []).extend(preds_per_pos[pos])
            if step == 3:
                for fidx in frame_indices:
                    finalize(fidx)
            else:  # step == 1: frame at buffer position 0 has now received its
                   # last contribution (from windows fidx-2, fidx-1, fidx).
                finalize(frame_indices[0])

        if not seg_control.cancelled:
            # Sliding-window tail: the last (frames_in - 1) frames only ever
            # appear at buffer positions > 0, so they're finalized here.
            for fidx in sorted(pending.keys()):
                finalize(fidx)

        if seg_control.cancelled:
            return  # no partial CSV on cancel or segment-local error

        if not rows:
            raise ValueError(
                f"No complete window of {cfg['model']['frames_in']} frames "
                f"(segment shorter than frames_in, or unreadable)"
            )

        df = pd.DataFrame(rows).sort_values("Frame").reset_index(drop=True)
        tmp_path = Path(str(csv_path) + ".tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(csv_path)
    except Exception as exc:
        # Covers both the accumulation loop above and CSV writing: any
        # failure here must be reported as this segment's error, not leave
        # the thread to crash silently while the wave reports "ok" (that
        # DataFrame().sort_values on an empty `rows` used to do exactly
        # that -- KeyError escaped uncaught and the segment was still
        # counted as processed).
        error_box[seg_id] = exc
        seg_control.local_cancel()


def _run_wave(detector, cfg, wave, control, progress_cb):
    queue_maxsize = max(1, int(cfg.get("runner", {}).get("queue_maxsize", 128)))
    queue_a = queue.Queue(maxsize=queue_maxsize * max(1, len(wave)))
    queue_b_map = {}
    seg_controls = {}
    timing = {}
    active_seg_ids = set()
    error_box = {}

    for seg_id, _video_path, _csv_path in wave:
        queue_b_map[seg_id] = queue.Queue(maxsize=queue_maxsize)
        seg_controls[seg_id] = _SegmentControl(control)
        timing[seg_id] = {"t_extract": 0.0, "t_model": 0.0, "t_post": 0.0, "total_frames": None}
        active_seg_ids.add(seg_id)

    threads = []
    for seg_id, video_path, _csv_path in wave:
        threads.append(threading.Thread(
            target=_extract_worker,
            args=(seg_id, video_path, cfg, seg_controls[seg_id], queue_a, error_box, timing[seg_id]),
            name=f"ram-pipeline-extract-{seg_id}",
            daemon=True,
        ))
    threads.append(threading.Thread(
        target=_model_worker_shared,
        args=(detector, control, queue_a, queue_b_map, seg_controls, active_seg_ids, error_box, timing),
        name="ram-pipeline-model",
        daemon=True,
    ))
    for seg_id, _video_path, csv_path in wave:
        threads.append(threading.Thread(
            target=_post_worker_seg,
            args=(detector, cfg, seg_controls[seg_id], queue_b_map[seg_id], csv_path, error_box, timing[seg_id], seg_id, progress_cb),
            name=f"ram-pipeline-post-{seg_id}",
            daemon=True,
        ))

    t_wall_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_wall_start

    out = {}
    for seg_id, video_path, _csv_path in wave:
        timing[seg_id]["t_wall"] = wall
        if seg_id in error_box:
            out[seg_id] = {"status": "error", "error": error_box[seg_id], "timing": timing[seg_id]}
        elif "__model__" in error_box:
            out[seg_id] = {"status": "error", "error": error_box["__model__"], "timing": timing[seg_id]}
        elif control.cancelled:
            out[seg_id] = {"status": "cancelled", "timing": timing[seg_id]}
        else:
            out[seg_id] = {"status": "ok", "timing": timing[seg_id]}
        print(
            f"[{out[seg_id]['status']}] {Path(video_path).name} "
            f"(extract={timing[seg_id]['t_extract']:.1f}s model={timing[seg_id]['t_model']:.1f}s "
            f"post={timing[seg_id]['t_post']:.1f}s wave_wall={wall:.1f}s)"
        )
    return out


def run_ram_pipeline_segments(detector, cfg, segments, control=None, progress_cb=None):
    """Run the RAM pipeline over several segments, sharing one model/GPU.

    ``segments`` is a list of ``(seg_id, video_path, csv_path)``. Up to
    ``runner.num_segment_workers`` (clamped to 1-6) segments are in flight at
    once; segments beyond that run in a later wave. A segment's own error
    (or the user's cancel) means no CSV is written for it, but a single
    segment's error does not stop the others in its wave, and later waves
    still run unless the user cancelled.

    Returns ``{seg_id: {"status": "ok"|"error"|"cancelled", "timing": {...},
    "error": Exception | None}}``.
    """
    control = control or RunControl()
    num_workers = max(1, min(6, int(cfg.get("runner", {}).get("num_segment_workers", 1))))
    results = {}
    idx = 0
    while idx < len(segments):
        if control.cancelled:
            for seg_id, _video_path, _csv_path in segments[idx:]:
                results[seg_id] = {"status": "cancelled", "timing": {}, "error": None}
            break
        wave = segments[idx: idx + num_workers]
        results.update(_run_wave(detector, cfg, wave, control, progress_cb))
        idx += num_workers
    return results


def run_ram_pipeline(detector, cfg, input_video_path, traj_output_path, control=None):
    """Run the RAM pipeline for a single video segment.

    Returns a timing dict (t_extract, t_model, t_post, t_wall) on success. On
    cancel, no CSV is written. On a worker exception, that exception is
    re-raised here after all threads join.
    """
    control = control or RunControl()
    results = run_ram_pipeline_segments(
        detector, cfg, [("segment", input_video_path, traj_output_path)], control=control
    )
    status = results["segment"]
    timing = status["timing"]
    if status["status"] == "error":
        raise status["error"]
    if status["status"] == "cancelled":
        print(f"RAM pipeline cancelled: {input_video_path}")
    else:
        print(
            f"RAM pipeline finished: {input_video_path} "
            f"(extract={timing['t_extract']:.1f}s model={timing['t_model']:.1f}s "
            f"post={timing['t_post']:.1f}s wall={timing['t_wall']:.1f}s)"
        )
    return timing
