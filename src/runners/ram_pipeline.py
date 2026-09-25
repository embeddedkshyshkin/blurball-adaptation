"""RAM-resident inference pipeline: extract -> Queue A -> model -> Queue B -> post -> CSV.

Three daemon threads overlap frame extraction, model forward + postprocess, and
tracker/CSV writing for a single video segment, instead of running them strictly
sequentially through a directory of extracted PNGs (see ``inference_video`` in
``inference.py`` for the disk-based baseline this mirrors).

Preprocessing (affine transform, resize/normalize) and the windowing/step logic
are copied 1:1 from ``inference_video`` so the CSV this produces matches the
disk-based path frame for frame. The postprocessor math (x/y/L/theta) itself is
untouched -- it already runs inside ``detector.run_tensor``.
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

from utils.image import get_affine_transform

_SENTINEL = None


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
    return _SENTINEL, False


def _put_sentinel(q, timeout=1.0):
    try:
        q.put(_SENTINEL, timeout=timeout)
    except queue.Full:
        pass  # downstream thread is already exiting via its own cancel check


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


def _extract_worker(video_path, cfg, control, queue_a, error_box, timing):
    frames_in = cfg["model"]["frames_in"]
    step = cfg["detector"]["step"]
    if step not in (1, 3):
        error_box.append(ValueError(f"Unsupported detector.step={step}; expected 1 or 3"))
        control.cancel_event.set()
        _put_sentinel(queue_a)
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
        trans = _build_affine(w, h, inp_w, inp_h, frames_in)

        frames_buffer = []
        idx_buffer = []
        frame_idx = 0
        while not control.cancelled:
            control.wait_if_paused()
            if control.cancelled:
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
                    "frame_indices": list(idx_buffer),
                    "tensor": input_tensor,
                    "affine": trans,
                }
                if not _put_with_cancel(queue_a, item, control):
                    break
                if step == 1:
                    frames_buffer.pop(0)
                    idx_buffer.pop(0)
                else:  # step == 3
                    frames_buffer = []
                    idx_buffer = []
    except Exception as exc:
        error_box.append(exc)
        control.cancel_event.set()
    finally:
        if cap is not None:
            cap.release()
        _put_sentinel(queue_a)


def _model_worker(detector, control, queue_a, queue_b, error_box, timing):
    try:
        with torch.no_grad():
            while not control.cancelled:
                item, ok = _get_with_cancel(queue_a, control)
                if not ok or item is _SENTINEL:
                    break
                control.wait_if_paused()
                if control.cancelled:
                    break
                t0 = time.time()
                batch_results, _hms_vis = detector.run_tensor(item["tensor"], item["affine"])
                timing["t_model"] += time.time() - t0
                num_positions = len(item["frame_indices"])
                preds_per_pos = [batch_results[0].get(pos, []) for pos in range(num_positions)]
                out_item = {
                    "frame_indices": item["frame_indices"],
                    "preds_per_pos": preds_per_pos,
                }
                if not _put_with_cancel(queue_b, out_item, control):
                    break
    except Exception as exc:
        error_box.append(exc)
        control.cancel_event.set()
    finally:
        _put_sentinel(queue_b)


def _post_worker(tracker, cfg, control, queue_b, csv_path, error_box, timing):
    step = cfg["detector"]["step"]
    model_name = cfg["model"]["name"]
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

    try:
        while not control.cancelled:
            item, ok = _get_with_cancel(queue_b, control)
            if not ok or item is _SENTINEL:
                break
            control.wait_if_paused()
            if control.cancelled:
                break
            frame_indices = item["frame_indices"]
            preds_per_pos = item["preds_per_pos"]
            for pos, fidx in enumerate(frame_indices):
                pending.setdefault(fidx, []).extend(preds_per_pos[pos])
            if step == 3:
                for fidx in frame_indices:
                    finalize(fidx)
            else:  # step == 1: frame at buffer position 0 has now received its
                   # last contribution (from windows fidx-2, fidx-1, fidx).
                finalize(frame_indices[0])

        if not control.cancelled:
            # Sliding-window tail: the last (frames_in - 1) frames only ever
            # appear at buffer positions > 0, so they're finalized here.
            for fidx in sorted(pending.keys()):
                finalize(fidx)
    except Exception as exc:
        error_box.append(exc)
        control.cancel_event.set()
        return

    if control.cancelled:
        return  # no partial CSV on cancel

    df = pd.DataFrame(rows).sort_values("Frame").reset_index(drop=True)
    tmp_path = Path(str(csv_path) + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)


def run_ram_pipeline(detector, tracker, cfg, input_video_path, traj_output_path, control=None):
    """Run the 3-thread RAM pipeline for one video segment.

    Returns a timing dict (t_extract, t_model, t_post, t_wall) on success. On
    cancel, no CSV is written and the dict still reports partial timings. On a
    worker exception, that exception is re-raised here after all threads join.
    """
    control = control or RunControl()
    queue_maxsize = max(1, int(cfg.get("runner", {}).get("queue_maxsize", 128)))
    queue_a = queue.Queue(maxsize=queue_maxsize)
    queue_b = queue.Queue(maxsize=queue_maxsize)
    error_box = []
    timing = {"t_extract": 0.0, "t_model": 0.0, "t_post": 0.0}

    t_wall_start = time.time()

    threads = [
        threading.Thread(
            target=_extract_worker,
            args=(input_video_path, cfg, control, queue_a, error_box, timing),
            name="ram-pipeline-extract",
            daemon=True,
        ),
        threading.Thread(
            target=_model_worker,
            args=(detector, control, queue_a, queue_b, error_box, timing),
            name="ram-pipeline-model",
            daemon=True,
        ),
        threading.Thread(
            target=_post_worker,
            args=(tracker, cfg, control, queue_b, traj_output_path, error_box, timing),
            name="ram-pipeline-post",
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    timing["t_wall"] = time.time() - t_wall_start

    if error_box:
        raise error_box[0]

    if control.cancelled:
        print(f"RAM pipeline cancelled: {input_video_path}")
    else:
        print(
            f"RAM pipeline finished: {input_video_path} "
            f"(extract={timing['t_extract']:.1f}s model={timing['t_model']:.1f}s "
            f"post={timing['t_post']:.1f}s wall={timing['t_wall']:.1f}s)"
        )
    return timing
