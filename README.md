<div align="center">

# [BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking](https://arxiv.org/abs/2509.18387)

Thomas Gossard, Filip Radovic, Andreas Ziegler, Andreas Zell  
University of Tübingen

**CVPRW 2026 – CVSports**

[[Paper](https://arxiv.org/abs/2509.18387)]
[[Project Page](https://cogsys-tuebingen.github.io/blurball/)]
[[Dataset](#dataset)]

</div>

<p align="center">
  <img src="demo.gif" width="800"/>
</p>

## Overview

BlurBall is a framework for joint table tennis ball detection and motion blur estimation in high-speed scenes.

Unlike conventional point-based detectors, BlurBall explicitly models the ball motion blur and jointly predicts:

- Ball location
- Blur orientation
- Blur extent

This repository builds upon [WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://github.com/nttcom/WASB-SBDT/) with additional training scripts, dataset support, and modifications for blur-aware ball detection.


## Contributions

- **New dataset**: 64k frames from real table tennis games, annotated with ball positions and explicit motion blur attributes (length + orientation).  
- **Blur-aware labeling**: A new annotation convention placing the ball at the blur center, improving accuracy across detectors.  
- **BlurBall model**: An HRNet-based multi-frame detector extended with attention mechanisms (Squeeze-and-Excitation) to jointly predict ball position and blur.  
- **Pretrained models**: Ready-to-use weights for BlurBall and other baselines (WASB, TrackNetV2, Monotrack, DeepBall, BallSeg).  


## Dataset

We release a **table tennis ball dataset** that provides both ball positions and motion blur attributes.  
Download here: [NextCloud](https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7).

After downloading, update the dataset config:

```yaml
# src/configs/dataset/tabletennis.yaml
root_dir: <path_to_dataset>
```

## Weights

All trained model weights for BlurBall, WASB, TrackNetv2, ResTrackNetv2, BallSeg, DeepBall, DeepBall large and Monotrack are available here: [Nextcloud](https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC)

## Installation

```bash
pip install -r requirements.txt
```

Set the root of the repository correctly in src/config/global:
```bash
WASB_ROOT=<path to root of repo>
```

Set hydra environment variable:
```bash
export HYDRA_FULL_ERROR=1
```

## Evaluation
Once the weights and dataset are downloaded, you can evaluate the different models as follows:

### BlurBall

#### 3-step inference
```
python src/main.py --config-name=eval_blurball detector.model_path=<path_to_blurball>
```

#### 1-step inference (recommended with threshold=0.7):
```
python src/main.py --config-name=eval_blurball detector.model_path=<path_to_blurball> detector.step=1 detector.postprocessor.score_threshold=0.7
```

### For the other models
```
python src/main.py --config-name=eval_<model_name> runner=eval detector.model_path=<path_to_model>
```

## Inference

BlurBall uses a multi-frame MIMO setup and is sensitive to duplicated frames (common in re-encoded online videos). During inference, the script will automatically generate a directory of unique frames.

Run inference on a video:

```
python src/main.py --config-name=inference_<model> detector.model_path=<path to corresponding model> +input_vid=<path to vid>
```

For the speed/direction visualization, use the same Hydra CLI style and additionally provide the PongEye calibration JSON:

```
python src/main.py --config-name=inference_blurball detector.model_path=<path to corresponding model> +input_vid=<path_to_video> +calibration_file=<path_to_calibration.json>
```

The calibration JSON must contain the calibrated table corners and table dimensions. An example is provided at:

```
examples/pongeye_calibration.json
```

The visualization displays the detected ball, a continuously rotating direction arrow, and the current calibrated table-plane speed in km/h in the upper-center HUD. Speed is calculated using the source video's FPS and the image-to-table homography derived from the calibration corners.

### PongEye recording folders

To process a PongEye recording containing `segments/*.mov` or `segments/*.mp4`, pass
the recording directory rather than an individual video:

```
python src/main.py --config-name=inference_blurball \
  detector.model_path=<path_to_blurball> \
  input_folder=/path/to/recording
```

This writes one CSV per segment to `blurBall/segment_XXX.csv` and, in the default
`standard` mode, annotated videos to `blurBall/segments/segment_XXX.mov`. Segments
are sorted by filename and are independently tracked. `calibration.json` is loaded
once from the recording directory when the speed/direction HUD is enabled.

For trajectory CSVs only (no annotated video and no persistent extracted PNGs):

```
python src/main.py --config-name=inference_blurball \
  detector.model_path=<path_to_blurball> \
  input_folder=/path/to/recording \
  runner.mode=csv_only
```

Existing complete outputs are skipped by default. Regenerate them with
`overwrite=true`.

### BlurBall parameters
- Step size: trade off between accuracy and speed
- Score threshold: recommended 0.7 for 1-step inference
- Speed/direction HUD: enabled by default for this adaptation
- Speed smoothing window: 4 frames
- Speed smoothing factor: 0.35

Example:
```
python src/main.py --config-name=inference_blurball detector.model_path=<path to corresponding model> detector.step=1 detector.postprocessor.score_threshold=0.7 +input_vid=<path_to_video> +calibration_file=<path_to_calibration.json>
```

## Training
Train BlurBall from scratch:

```
python src/main.py --config-name=train_blur
```

To fine-tune on your own dataset:

- Add images and GT CSV labels in the same format.  
- Update `src/configs/dataset/tabletennis.yaml` with your dataset name.

## Citation

If you use this work, please cite:

```bibtex
@article{gossard2026blurball,
  title   = {BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking},
  author  = {Thomas Gossard and Filip Radovic and Andreas Ziegler and Andreas Zell},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year    = {2026}
}
```
