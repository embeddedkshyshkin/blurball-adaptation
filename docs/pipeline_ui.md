# RAM pipeline and desktop UI (S2.9)

This covers `runner.use_ram_pipeline=true` and the tkinter app at
`src/ui/pipeline_app.py`. Both are opt-in and only tested against BlurBall
(`--config-name=inference_blurball`, the only model config that actually
ships in `src/configs/model/`); the default Hydra CLI path
(`runner.use_ram_pipeline=false`, the default) is unchanged.

## What it does

The historical path extracts every frame of a video to PNGs on disk, then
reads them back for inference. The RAM path keeps frames in memory: three
threads -- extract, model (GPU forward + transfer), and post (blob
detection + tracker + CSV write) -- run concurrently on overlapping windows
of `frames_in` frames, connected by two bounded queues (Queue A: tensors,
Queue B: heatmaps). Preprocessing and windowing are the same code path as
the disk-based pipeline, so output is identical (verified bit-exact against
the disk path -- see the `git log` on `s2.9-*` commits for the parity
numbers).

For a folder of segments (`runner.num_segment_workers` > 1), several
segments run in parallel, but they share **one** detector/model on the GPU
-- not one model copy per segment, which is what the old
`runner.num_parallel_segments` (removed) did via a process pool. Sharing
the model keeps VRAM use flat regardless of worker count, at the cost of
the GPU forward pass being the shared bottleneck: parallel segments overlap
their I/O/extraction and CPU-side post-processing with each other's GPU
time, but they don't get a proportional multiple of GPU throughput. See
[Measured speedup](#measured-speedup) below for real numbers.

What it does **not** do (out of scope for this slice):
- No PNG/frame dump, no annotated-video rendering, no heatmap visualization.
  `runner.mode` must be `csv_only` (or `trajectory_only`, normalized to the
  same thing) when `use_ram_pipeline=true`.
- No speed/direction HUD -- that's been moved to PongEye itself.
- Only the `blurball` detector is wired to the pipeline's
  `to_heatmaps`/`results_from_heatmaps` split (`tracknetv2`/`deepball` still
  only support the disk path).

## CLI

Single video:

```bash
python src/main.py --config-name=inference_blurball \
  detector.model_path=<path_to_blurball_weights> \
  input_vid=<path_to_video> \
  runner.use_ram_pipeline=true \
  detector.step=3
```

CSV is written next to the video, as `<stem>_traj.csv`.

A PongEye recording folder (`<recording>/segments/*.mov|*.mp4`), several
segments in parallel:

```bash
python src/main.py --config-name=inference_blurball \
  detector.model_path=<path_to_blurball_weights> \
  input_folder=<path_to_recording> \
  runner.mode=csv_only \
  runner.use_ram_pipeline=true \
  runner.num_segment_workers=3 \
  runner.queue_maxsize=128 \
  detector.step=3
```

CSVs land at `<recording>/blurBall/<segment>.csv` -- the same convention as
the existing folder mode, and what PongEye's `reconstruct_trajectory.py`
reads. Already-complete segments are skipped unless `overwrite=true`, same
as the disk path.

| Key | Default | Notes |
|---|---|---|
| `runner.use_ram_pipeline` | `false` | Opt-in; disk path unchanged when false |
| `runner.num_segment_workers` | `1` | 1-6, folder mode only; ignored by the disk path |
| `runner.queue_maxsize` | `128` | Per segment, per queue; ~1 window's tensor is ~5-7 MB at the default 288x512 input, so 128 is roughly 1 GB |
| `runner.mode` | `standard` | Must be `csv_only` when `use_ram_pipeline=true` |

## Desktop UI

```bash
python src/ui/pipeline_app.py
# or, from the repo root:
python -m src.ui.pipeline_app
```

1. **Input**: either
   - a PongEye recording folder (contains `segments/`) -- output routes to
     `<input>/blurBall/`, the Output folder field is ignored;
   - a plain folder of `.mp4`/`.mov` files (no `segments/` subfolder) -- every
     video directly inside it becomes its own segment;
   - individually picked files via "Browse files".

   For the second and third cases, set **Output folder**: one `<stem>.csv`
   is written there per video.
2. **Model weights**: a `.pth`/`.pth.tar` checkpoint. Model is fixed to
   `blurball` (see [What it does](#what-it-does) for why).
3. **Step**: `1` (slower, every window overlaps by 2 frames) or `3` (faster,
   non-overlapping windows).
4. **Workers**: segments in parallel, 1-6. Start at 2; see
   [Measured speedup](#measured-speedup) for what to expect from higher
   values.
5. **Queue size**: per-segment queue depth; lower it if you're tight on RAM.
6. **Start / Pause / Resume / Cancel**: Pause takes effect at the next
   window boundary (extraction and post both check between windows, not
   mid-window); Cancel stops every in-flight segment and writes no CSV for
   any segment that hadn't already finished.
7. **Progress / Log**: progress is frames finalized (i.e. written to a CSV
   row) across all segments that have started, not frames decoded -- it can
   sit at 0% for a moment while the first window/model call warms up.
8. **Open output folder**: opens the same directory the run wrote to.

Closing the window mid-run asks for confirmation, then cancels and waits
for the background thread to actually stop before closing -- don't force-kill
the window while a run is in progress; if you do, the process can abort
because a native CUDA call was still in flight when Python exited (this can
happen independent of the RAM pipeline itself, any long-running CUDA thread
under tkinter is subject to it -- the confirm-and-wait dialog is the fix,
not a workaround for something specific to this pipeline).

## Measured speedup

Measured on this machine (RTX 5000 Ada, 16 GB VRAM, 22 threads, 64 GB RAM)
with real BlurBall weights, on real clips (30s @ 30fps each). Detector/model
load time is excluded from every number below (timed from after the
checkpoint is already on the GPU, same for both old and new), since that's
a fixed ~seconds-long cost independent of which pipeline runs, and both the
CLI and the UI only pay it once per process.

**Single video, 450 frames:**

| Path | step=3 | step=1 |
|---|---|---|
| OLD (disk: extract every frame to PNG, then infer) | 33.8s | 50.1s |
| NEW (`use_ram_pipeline=true`, no parallelism) | 6.2s | 17.3s |
| **Speedup** | **5.4x** | **2.9x** |

**3 segments x 450 frames each (folder mode), step=3:**

| Path | Wall time | Speedup vs OLD |
|---|---|---|
| OLD (sequential, one segment at a time, disk-based) | 98.6s | 1.0x |
| NEW, `num_segment_workers=1` | 19.4s | 5.1x |
| NEW, `num_segment_workers=2` | 16.8s | 5.9x |
| NEW, `num_segment_workers=3` | 15.3s | 6.4x |

Takeaways:
- **Most of the win is the RAM architecture itself, not parallel segments.**
  Going from the old disk-bound sequential path to the RAM pipeline with
  `num_segment_workers=1` already gets ~5x, both for a single video (no
  PNG round-trip, extract/model overlap) and for a folder run (same, plus
  no per-segment PNG cleanup). That's commit 1's change; segment
  parallelism (commit 2) is what's layered on top of it.
- **Parallel segments add a real but smaller further gain**: workers=1 to
  workers=3 was 19.4s to 15.3s here, about 1.27x, not 3x. All workers share
  one GPU forward pass (that's the point of commit 2 -- one model, not one
  per segment), so more workers overlap segments' I/O/CPU-bound work with
  each other but don't multiply GPU throughput. Diminishing returns past
  2-3 workers is expected on a single GPU; the ceiling is set by total GPU
  time, which stays roughly constant regardless of worker count.
- `step=1` (sliding, 3x more windows than `step=3`) sees a smaller
  speedup (2.9x vs 5.4x) because GPU forward-pass time is a bigger share of
  total time relative to extraction, and that part doesn't change between
  old and new -- overlap only hides work that used to be sequential, it
  doesn't make the GPU faster.

## Known limitation

`runner.num_parallel_segments` (the old `ProcessPoolExecutor`-based
parallel path, one full model per worker process) has been removed. If you
were relying on it for something the shared-model path doesn't cover (e.g.
genuinely wanting separate model copies), that capability is gone --
`num_segment_workers` always shares one model.
