from pathlib import Path

from omegaconf import OmegaConf

import runners.inference as inference


def _cfg(recording, mode="standard", overwrite=False):
    return OmegaConf.create({
        "input_folder": str(recording),
        "input_vid": None,
        "calibration_file": None,
        "overwrite": overwrite,
        "runner": {
            "mode": mode,
            "visualization": {"show_speed_direction": False},
        },
    })


def _recording(tmp_path, names=("segment_010.mov", "segment_002.mp4", "notes.txt")):
    recording = tmp_path / "recording"
    segments = recording / "segments"
    segments.mkdir(parents=True)
    for name in names:
        (segments / name).write_bytes(b"source-video")
    return recording


def test_discovers_supported_segments_in_deterministic_order(tmp_path):
    recording = _recording(tmp_path)
    segments = inference.RecordingInferenceRunner.discover_segments(recording)
    assert [path.name for path in segments] == ["segment_002.mp4", "segment_010.mov"]


def test_folder_run_generates_expected_output_paths_and_keeps_sources(tmp_path, monkeypatch):
    recording = _recording(tmp_path, ("segment_000.mov", "segment_001.mov"))
    calls = []

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, source, csv_path, mode, annotated_video_path=None, **kwargs):
            calls.append((Path(source), Path(csv_path), mode, Path(annotated_video_path)))
            Path(csv_path).write_text("Frame,X,Y,Visibility,L,Theta\n0,1,2,1,0,0\n")
            Path(annotated_video_path).write_bytes(b"annotated")

    monkeypatch.setattr(inference, "VideoInferenceProcessor", FakeProcessor)
    result = inference.RecordingInferenceRunner(_cfg(recording)).run()

    assert result["processed"] == 2
    assert [(call[1].name, call[3].name) for call in calls] == [
        ("segment_000.csv", "segment_000.mov"),
        ("segment_001.csv", "segment_001.mov"),
    ]
    assert (recording / "blurBall" / "segment_000.csv").is_file()
    assert (recording / "blurBall" / "segments" / "segment_000.mov").is_file()
    assert (recording / "segments" / "segment_000.mov").read_bytes() == b"source-video"


def test_csv_only_does_not_request_annotated_video(tmp_path, monkeypatch):
    recording = _recording(tmp_path, ("segment_000.mov",))
    requested_video_paths = []

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, source, csv_path, mode, annotated_video_path=None, **kwargs):
            requested_video_paths.append(annotated_video_path)
            Path(csv_path).write_text("Frame,X,Y,Visibility\n0,1,2,1\n")

    monkeypatch.setattr(inference, "VideoInferenceProcessor", FakeProcessor)
    inference.RecordingInferenceRunner(_cfg(recording, mode="csv_only")).run()
    assert requested_video_paths == [None]
    assert (recording / "blurBall" / "segment_000.csv").is_file()
    assert not (recording / "blurBall" / "segments").exists()


def test_csv_only_uses_and_removes_a_temporary_frame_workspace(tmp_path, monkeypatch):
    source = tmp_path / "segment_000.mov"
    source.write_bytes(b"source-video")
    frame_dirs = []

    def fake_extract(video_path, filter=False, output_dir=None):
        frame_dir = Path(output_dir)
        frame_dirs.append(frame_dir)
        frame_dir.mkdir(parents=True)
        (frame_dir / "00000.png").write_bytes(b"frame")
        return str(frame_dir)

    def fake_inference(detector, tracker, input_video, frame_dir, cfg, **kwargs):
        Path(kwargs["traj_output_path"]).write_text("Frame,X,Y,Visibility\n0,1,2,1\n")
        assert Path(frame_dir).is_dir()
        return {"t_elapsed": 0, "num_frames": 1}

    monkeypatch.setattr(inference, "build_detector", lambda *args, **kwargs: object())
    monkeypatch.setattr(inference, "build_tracker", lambda *args, **kwargs: object())
    monkeypatch.setattr(inference, "process_video", fake_extract)
    monkeypatch.setattr(inference, "inference_video", fake_inference)
    processor = inference.VideoInferenceProcessor(_cfg(tmp_path, mode="csv_only"))
    processor.process(source, tmp_path / "out.csv", "csv_only")

    assert frame_dirs and not frame_dirs[0].exists()
    assert not list(tmp_path.glob("frames_segment_000"))
    assert source.read_bytes() == b"source-video"


def test_resume_overwrite_and_failure_do_not_stop_later_segments(tmp_path, monkeypatch):
    recording = _recording(tmp_path, ("segment_000.mov", "segment_001.mov", "segment_002.mov"))
    output = recording / "blurBall"
    output.mkdir()
    (output / "segment_000.csv").write_text("existing")
    calls = []

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, source, csv_path, mode, **kwargs):
            calls.append(Path(source).name)
            if Path(source).stem == "segment_001":
                raise RuntimeError("synthetic failure")
            Path(csv_path).write_text("Frame,X,Y,Visibility\n0,1,2,1\n")

    monkeypatch.setattr(inference, "VideoInferenceProcessor", FakeProcessor)
    result = inference.RecordingInferenceRunner(_cfg(recording, mode="csv_only")).run()
    assert calls == ["segment_001.mov", "segment_002.mov"]
    assert result["skipped"] == 1
    assert len(result["failures"]) == 1
    assert (output / "segment_002.csv").is_file()

    calls.clear()
    inference.RecordingInferenceRunner(_cfg(recording, mode="csv_only", overwrite=True)).run()
    assert calls == ["segment_000.mov", "segment_001.mov", "segment_002.mov"]


def test_folder_metrics_require_the_recording_calibration(tmp_path):
    recording = _recording(tmp_path, ("segment_000.mov",))
    cfg = _cfg(recording, mode="csv_only")
    cfg.runner.visualization.show_speed_direction = True
    try:
        inference.RecordingInferenceRunner(cfg).run()
    except ValueError as exc:
        assert str(recording / "calibration.json") in str(exc)
    else:
        raise AssertionError("missing calibration should fail when metrics are enabled")
