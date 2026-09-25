"""Multi-segment RAM pipeline: shared model thread, per-segment error isolation,
and global cancel. Uses a fake VideoCapture and fake detector so these run
without GPU/weights, but exercise the real threading/queue/windowing code in
runners/ram_pipeline.py end to end.
"""
import threading
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import runners.inference as inference
import runners.ram_pipeline as ram_pipeline
from runners.ram_pipeline import RunControl


class FakeDetector:
    """No detections, ever -- enough to exercise the pipeline plumbing
    (windowing, queues, tracker calls, CSV writing) without a real model."""

    def to_heatmaps(self, tensor, affine):
        return {"n": tensor.shape[0]}, {}

    def results_from_heatmaps(self, hms, affine_np):
        return {0: {0: [], 1: [], 2: []}}, {}


class _FakeCapture:
    def __init__(self, num_frames, w=32, h=32):
        self._n = num_frames
        self._i = 0
        self._w = w
        self._h = h

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._w
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._h
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._n
        return 0.0

    def read(self):
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def release(self):
        pass


class _FailingCapture:
    def isOpened(self):
        return False

    def get(self, prop):
        return 0.0

    def read(self):
        return False, None

    def release(self):
        pass


def _fake_capture_factory(frame_counts, w=32, h=32):
    def factory(path):
        stem = Path(path).stem
        n = frame_counts[stem]
        return _FailingCapture() if n is None else _FakeCapture(n, w, h)
    return factory


def _ram_cfg(recording, workers=1, step=3, queue_maxsize=8):
    return OmegaConf.create({
        "input_folder": str(recording),
        "input_vid": None,
        "calibration_file": None,
        "overwrite": False,
        "runner": {
            "mode": "csv_only",
            "use_ram_pipeline": True,
            "num_segment_workers": workers,
            "queue_maxsize": queue_maxsize,
            "visualization": {"show_speed_direction": False},
        },
        "model": {"name": "blurball", "frames_in": 3, "frames_out": 3, "inp_height": 32, "inp_width": 32},
        "detector": {"step": step},
        "tracker": {"name": "online_blur", "max_disp": 1000},
    })


def _recording(tmp_path, names):
    recording = tmp_path / "recording"
    segments = recording / "segments"
    segments.mkdir(parents=True)
    for name in names:
        (segments / name).write_bytes(b"source-video")
    return recording


def test_parallel_segments_write_csvs_and_isolate_a_bad_segment(tmp_path, monkeypatch):
    recording = _recording(tmp_path, ("segment_000.mov", "segment_001.mov", "segment_002.mov"))
    monkeypatch.setattr(inference, "build_detector", lambda *a, **k: FakeDetector())
    monkeypatch.setattr(
        ram_pipeline.cv2, "VideoCapture",
        _fake_capture_factory({"segment_000": 9, "segment_001": 12, "segment_002": None}),
    )

    result = inference.RecordingInferenceRunner(_ram_cfg(recording, workers=2)).run()

    assert result["processed"] == 2
    assert len(result["failures"]) == 1
    assert result["failures"][0][0] == "segment_002.mov"

    out_dir = recording / "blurBall"
    assert len(list(out_dir.glob("segment_000.csv"))) == 1
    assert len(list(out_dir.glob("segment_001.csv"))) == 1
    assert not (out_dir / "segment_002.csv").exists()

    import pandas as pd
    assert len(pd.read_csv(out_dir / "segment_000.csv")) == 9
    assert len(pd.read_csv(out_dir / "segment_001.csv")) == 12


def test_a_bad_segment_does_not_hang_the_shared_model_thread(tmp_path, monkeypatch):
    """Regression test: a segment whose extract thread fails immediately must
    still deliver its sentinel so the shared model thread's active_seg_ids
    bookkeeping clears it, instead of waiting forever for data that will
    never arrive."""
    recording = _recording(tmp_path, ("segment_000.mov",))
    monkeypatch.setattr(inference, "build_detector", lambda *a, **k: FakeDetector())
    monkeypatch.setattr(
        ram_pipeline.cv2, "VideoCapture",
        _fake_capture_factory({"segment_000": None}),
    )

    done = threading.Event()
    result = {}

    def go():
        result["value"] = inference.RecordingInferenceRunner(_ram_cfg(recording, workers=1)).run()
        done.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    assert done.wait(timeout=10), "run() did not return -- likely deadlocked"
    assert result["value"]["processed"] == 0
    assert len(result["value"]["failures"]) == 1


def test_cancel_stops_all_active_segments_without_writing_csv(tmp_path, monkeypatch):
    recording = _recording(tmp_path, ("segment_000.mov", "segment_001.mov", "segment_002.mov"))
    monkeypatch.setattr(inference, "build_detector", lambda *a, **k: FakeDetector())
    # Large frame counts so the run is still in progress when cancel fires.
    monkeypatch.setattr(
        ram_pipeline.cv2, "VideoCapture",
        _fake_capture_factory({"segment_000": 100000, "segment_001": 100000, "segment_002": 100000}),
    )

    control = RunControl()

    def cancel_soon():
        time.sleep(0.3)
        control.cancel_event.set()

    threading.Thread(target=cancel_soon, daemon=True).start()

    done = threading.Event()
    result = {}

    def go():
        result["value"] = inference.RecordingInferenceRunner(_ram_cfg(recording, workers=3)).run(control=control)
        done.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    assert done.wait(timeout=10), "run() did not return after cancel -- likely deadlocked"
    assert result["value"]["processed"] == 0
    out_dir = recording / "blurBall"
    assert not any(out_dir.glob("*.csv"))
