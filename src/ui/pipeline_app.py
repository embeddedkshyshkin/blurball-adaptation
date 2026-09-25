"""Desktop UI for the RAM inference pipeline (S2.9).

Entry point: ``python src/ui/pipeline_app.py`` or ``python -m src.ui.pipeline_app``
from the repo root, with the project venv active.

Only wraps runner.use_ram_pipeline=true (see runners/ram_pipeline.py) --
mode=csv_only always, no annotated-video/heatmap rendering, and only the
"blurball" detector (the only one wired to the RAM pipeline's
to_heatmaps/results_from_heatmaps split). The Hydra CLI path is untouched
by anything in this file.
"""
import os
import platform
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from detectors import build_detector  # noqa: E402
from runners.inference import RecordingInferenceRunner  # noqa: E402
from runners.ram_pipeline import RunControl, run_ram_pipeline_segments  # noqa: E402

_CONFIG_DIR = str((_SRC_DIR / "configs").resolve())
_VIDEO_SUFFIXES = {".mp4", ".mov"}


def _compose_cfg(model_path, step, num_segment_workers, queue_maxsize):
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="inference_blurball",
            overrides=[
                f"detector.model_path={model_path}",
                f"detector.step={step}",
                "runner.mode=csv_only",
                "runner.use_ram_pipeline=true",
                f"runner.num_segment_workers={num_segment_workers}",
                f"runner.queue_maxsize={queue_maxsize}",
                "runner.visualization.show_speed_direction=false",
            ],
        )
    if torch.cuda.is_available():
        cfg["runner"]["device"] = "cuda"
        cfg["runner"]["gpus"] = [0]
    return cfg


def _open_in_file_manager(path):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif system == "Windows":
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        messagebox.showerror("Open output folder", f"Could not open {path}: {exc}")


class PipelineApp:
    def __init__(self, root):
        self.root = root
        root.title("BlurBall RAM Pipeline")

        self.control = None
        self.worker_thread = None
        self.ui_queue = queue.Queue()
        self.seg_totals = {}   # seg_id -> total_frames (None if unknown)
        self.seg_done = {}     # seg_id -> frames finalized so far
        self.segments_total_count = None  # set once "files" mode knows its segment count
        self.output_dir_for_open = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.step_var = tk.StringVar(value="3")
        self.workers_var = tk.IntVar(value=2)
        self.queue_var = tk.IntVar(value=128)

        self._build_layout()
        self._update_button_states(running=False)
        self.root.after(100, self._drain_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- layout -----------------------------------------------------

    def _build_layout(self):
        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True)

        row = 0
        ttk.Label(frm, text="Input (folder or files):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.input_var, width=60).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse folder", command=self._browse_input_folder).grid(row=row, column=2, **pad)
        ttk.Button(frm, text="Browse files", command=self._browse_input_files).grid(row=row, column=3, **pad)
        row += 1

        ttk.Label(frm, text="Output folder:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.output_var, width=60).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=self._browse_output).grid(row=row, column=2, **pad)
        ttk.Label(
            frm, text="(ignored for a PongEye recording folder -- output goes to <input>/blurBall)",
            foreground="gray",
        ).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="Model weights (.pth/.pth.tar):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.model_path_var, width=60).grid(row=row, column=1, **pad)
        ttk.Button(frm, text="Browse", command=self._browse_weights).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(frm, text="Model:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Label(frm, text="blurball (only detector wired to the RAM pipeline)").grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1

        opts = ttk.Frame(frm)
        opts.grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        ttk.Label(opts, text="Step:").pack(side="left")
        ttk.Combobox(opts, textvariable=self.step_var, values=["1", "3"], width=3, state="readonly").pack(
            side="left", padx=(2, 12)
        )
        ttk.Label(opts, text="Workers (1-6):").pack(side="left")
        ttk.Spinbox(opts, from_=1, to=6, textvariable=self.workers_var, width=4).pack(side="left", padx=(2, 12))
        ttk.Label(opts, text="Queue size:").pack(side="left")
        ttk.Spinbox(opts, from_=8, to=1024, increment=8, textvariable=self.queue_var, width=6).pack(
            side="left", padx=(2, 12)
        )
        self.device_label = ttk.Label(opts, text=self._device_text())
        self.device_label.pack(side="left", padx=(20, 0))
        row += 1

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=4)
        self.pause_btn = ttk.Button(btns, text="Pause", command=self._on_pause)
        self.pause_btn.pack(side="left", padx=4)
        self.resume_btn = ttk.Button(btns, text="Resume", command=self._on_resume)
        self.resume_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._on_cancel)
        self.cancel_btn.pack(side="left", padx=4)
        row += 1

        self.progress_label = ttk.Label(frm, text="Progress: idle")
        self.progress_label.grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        row += 1
        self.progress = ttk.Progressbar(frm, orient="horizontal", length=500, mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=4, sticky="we", **pad)
        row += 1

        ttk.Label(frm, text="Log:").grid(row=row, column=0, sticky="w", **pad)
        row += 1
        self.log_widget = scrolledtext.ScrolledText(frm, width=100, height=16, state="disabled")
        self.log_widget.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        row += 1

        self.open_output_btn = ttk.Button(
            frm, text="Open output folder", command=self._on_open_output, state="disabled"
        )
        self.open_output_btn.grid(row=row, column=0, sticky="w", **pad)

    def _device_text(self):
        if torch.cuda.is_available():
            return f"Device: CUDA ({torch.cuda.get_device_name(0)})"
        return "Device: CUDA not available"

    # ---- browse handlers ---------------------------------------------

    def _browse_input_folder(self):
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self.input_var.set(path)

    def _browse_input_files(self):
        paths = filedialog.askopenfilenames(
            title="Select video files", filetypes=[("Video", "*.mp4 *.mov"), ("All files", "*.*")]
        )
        if paths:
            self.input_var.set("|".join(paths))

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _browse_weights(self):
        path = filedialog.askopenfilename(
            title="Select model weights", filetypes=[("Weights", "*.pth *.pth.tar"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)

    def _on_open_output(self):
        if self.output_dir_for_open is not None:
            _open_in_file_manager(self.output_dir_for_open)

    # ---- input resolution ----------------------------------------------

    def _resolve_segments(self):
        """Returns (kind, payload). kind is 'recording' (payload = folder Path)
        or 'files' (payload = list of (seg_id, video_path, csv_path))."""
        raw = self.input_var.get().strip()
        if not raw:
            raise ValueError("Choose an input folder or select video files")
        parts = raw.split("|")
        if len(parts) == 1 and Path(parts[0]).is_dir():
            folder = Path(parts[0])
            if (folder / "segments").is_dir():
                return "recording", folder
            videos = sorted(
                p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
            )
            if not videos:
                raise ValueError(f"No .mp4/.mov files found directly in {folder}")
            return "files", self._files_to_segments(videos)
        videos = [Path(p) for p in parts]
        for v in videos:
            if not v.is_file():
                raise ValueError(f"Not a file: {v}")
        return "files", self._files_to_segments(videos)

    def _files_to_segments(self, videos):
        output_dir = self.output_var.get().strip()
        if not output_dir:
            raise ValueError("Choose an output folder for individual video files")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_for_open = output_dir
        return [(v.stem, v, output_dir / f"{v.stem}.csv") for v in videos]

    # ---- start/pause/resume/cancel -------------------------------------

    def _on_start(self):
        try:
            model_path = self.model_path_var.get().strip()
            if not model_path or not Path(model_path).is_file():
                raise ValueError(f"Model weights file not found: {model_path!r}")
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available; BlurBall requires an NVIDIA GPU")
            kind, payload = self._resolve_segments()
        except ValueError as exc:
            messagebox.showerror("Cannot start", str(exc))
            return

        cfg = _compose_cfg(model_path, self.step_var.get(), self.workers_var.get(), self.queue_var.get())
        self.control = RunControl()
        self.seg_totals = {}
        self.seg_done = {}
        self.segments_total_count = None
        self.progress.configure(value=0, maximum=1)
        self._clear_log()
        self._log(f"Starting ({kind})...")
        self._update_button_states(running=True)

        self.worker_thread = threading.Thread(
            target=self._run_worker, args=(cfg, kind, payload), daemon=True
        )
        self.worker_thread.start()

    def _on_pause(self):
        if self.control is not None:
            self.control.pause_event.clear()
            self._log("Paused")

    def _on_resume(self):
        if self.control is not None:
            self.control.pause_event.set()
            self._log("Resumed")

    def _on_cancel(self):
        if self.control is None:
            return
        if messagebox.askyesno("Cancel", "Cancel the running pipeline? No CSV will be written for in-progress segments."):
            self.control.cancel_event.set()
            self.control.pause_event.set()  # unblock anything waiting on pause
            self._log("Cancel requested...")

    def _on_close(self):
        """Closing the window while the worker thread is still running native
        CUDA calls can abort the process at interpreter shutdown. Cancel and
        wait for the thread to actually finish before destroying the window."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            if not messagebox.askyesno("Quit", "A pipeline run is in progress. Cancel it and quit?"):
                return
            if self.control is not None:
                self.control.cancel_event.set()
                self.control.pause_event.set()  # unblock anything waiting on pause
            self._wait_for_worker_then_destroy()
        else:
            self.root.destroy()

    def _wait_for_worker_then_destroy(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.root.after(100, self._wait_for_worker_then_destroy)
        else:
            self.root.destroy()

    def _update_button_states(self, running):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.pause_btn.configure(state="normal" if running else "disabled")
        self.resume_btn.configure(state="normal" if running else "disabled")
        self.cancel_btn.configure(state="normal" if running else "disabled")

    # ---- background worker ---------------------------------------------

    def _progress_cb(self, seg_id, frames_done, frames_total):
        self.ui_queue.put(("progress", seg_id, frames_done, frames_total))

    def _run_worker(self, cfg, kind, payload):
        try:
            if kind == "recording":
                self.output_dir_for_open = payload / "blurBall"
                cfg["input_folder"] = str(payload)
                runner = RecordingInferenceRunner(cfg)
                n_segments = len(RecordingInferenceRunner.discover_segments(payload))
                self.ui_queue.put(("segments_total", n_segments))
                self.ui_queue.put(("log", f"Recording folder: routing output to {self.output_dir_for_open}"))
                result = runner.run(control=self.control, progress_cb=self._progress_cb)
                self.ui_queue.put(("done", result))
            else:
                self.ui_queue.put(("segments_total", len(payload)))
                detector = build_detector(cfg)
                ram_segments = [(seg_id, str(video_path), str(csv_path)) for seg_id, video_path, csv_path in payload]
                results = run_ram_pipeline_segments(
                    detector, cfg, ram_segments, control=self.control, progress_cb=self._progress_cb
                )
                processed = sum(1 for r in results.values() if r["status"] == "ok")
                failures = [
                    (seg_id, str(r["error"])) for seg_id, r in results.items() if r["status"] == "error"
                ]
                cancelled = any(r["status"] == "cancelled" for r in results.values())
                self.ui_queue.put(("done", {"processed": processed, "failures": failures, "cancelled": cancelled}))
        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))

    # ---- UI thread: drain queue from the worker -------------------------

    def _drain_ui_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "segments_total":
                    self.segments_total_count = msg[1]
                elif kind == "progress":
                    _, seg_id, frames_done, frames_total = msg
                    self.seg_done[seg_id] = frames_done
                    if frames_total:
                        self.seg_totals[seg_id] = frames_total
                    self._refresh_progress()
                elif kind == "done":
                    self._on_worker_done(msg[1])
                elif kind == "error":
                    self._on_worker_error(msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._drain_ui_queue)

    def _refresh_progress(self):
        done = sum(self.seg_done.values())
        known_total = sum(v for v in self.seg_totals.values())
        # Segments whose total isn't known yet don't contribute to the
        # denominator; the bar undercounts slightly until they report in.
        total = max(done, known_total, 1)
        self.progress.configure(value=done, maximum=total)
        n_active = len(self.seg_done)
        total_segments = self.segments_total_count
        seg_text = f"{n_active}/{total_segments}" if total_segments else str(n_active)
        self.progress_label.configure(
            text=f"Progress: {done}/{total} frames finalized, {seg_text} segment(s) started"
        )

    def _on_worker_done(self, result):
        self._log(f"Done: {result}")
        self.open_output_btn.configure(state="normal" if self.output_dir_for_open else "disabled")
        if result.get("cancelled"):
            messagebox.showinfo("Cancelled", "Pipeline cancelled.")
        elif result.get("failures"):
            messagebox.showwarning(
                "Finished with failures",
                f"Processed {result.get('processed', 0)}; failed: {result['failures']}",
            )
        else:
            messagebox.showinfo("Done", f"Processed {result.get('processed', 0)} segment(s).")
        self._update_button_states(running=False)

    def _on_worker_error(self, message):
        self._log(f"ERROR: {message}")
        messagebox.showerror("Pipeline error", message)
        self._update_button_states(running=False)

    # ---- log helpers -----------------------------------------------------

    def _log(self, line):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", f"{ts}  {line}\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _clear_log(self):
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")


def main():
    root = tk.Tk()
    PipelineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
