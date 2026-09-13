import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import shutil
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn
import cv2

from dataloaders import build_dataloader
from detectors import build_detector
from trackers import build_tracker
from utils import mkdir_if_missing, draw_frame, draw_speed_direction_hud, gen_video, Center, Evaluator
from utils.image import get_affine_transform, affine_transform
from utils.preprocess import process_video
from utils.motion import MotionEstimator

from .base import BaseRunner


def load_speed_calibration(calibration_file):
    """Load PongEye table calibration and build image -> table homography."""
    if not calibration_file:
        return None

    path = Path(calibration_file)
    if not path.is_absolute():
        path = Path(HydraConfig.get().runtime.cwd) / path

    with path.open("r", encoding="utf-8") as f:
        calibration = json.load(f)

    data = calibration["calibration"]
    corners = data["corners"]
    length_m = float(data["tableDimensions"]["lengthMeters"])
    width_m = float(data["tableDimensions"]["widthMeters"])

    image_points = np.array(
        [
            [corners["topLeft"]["x"], corners["topLeft"]["y"]],
            [corners["topRight"]["x"], corners["topRight"]["y"]],
            [corners["bottomRight"]["x"], corners["bottomRight"]["y"]],
            [corners["bottomLeft"]["x"], corners["bottomLeft"]["y"]],
        ],
        dtype=np.float32,
    )
    table_points = np.array(
        [
            [0.0, 0.0],
            [length_m, 0.0],
            [length_m, width_m],
            [0.0, width_m],
        ],
        dtype=np.float32,
    )

    return cv2.getPerspectiveTransform(image_points, table_points)


def project_to_table(point_xy, homography):
    point = np.array([[point_xy]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, homography)[0, 0]
    return float(projected[0]), float(projected[1])


def load_trajectory(traj_path, imgs_paths, model_name):
    """Load an existing BlurBall trajectory without rerunning detection/tracking."""
    df = pd.read_csv(traj_path)
    if not {"X", "Y", "Visibility"}.issubset(df.columns):
        raise ValueError(f"Invalid trajectory file: {traj_path}")

    if len(df) > len(imgs_paths):
        raise ValueError(
            f"Trajectory has {len(df)} rows but only {len(imgs_paths)} extracted frames are available"
        )

    result_dict = {}
    for index, row in df.iterrows():
        img_path = str(imgs_paths[index])
        result = {
            "x": float(row["X"]),
            "y": float(row["Y"]),
            "visi": int(row["Visibility"]),
            "score": 0.0,
        }
        if model_name == "blurball":
            if not {"L", "Theta"}.issubset(df.columns):
                raise ValueError(f"BlurBall trajectory is missing L/Theta columns: {traj_path}")
            result["angle"] = float(row["Theta"])
            result["length"] = float(row["L"])
        result_dict[img_path] = result

    print(f"Reusing trajectory: {traj_path} ({len(result_dict)} frames)")
    return result_dict


@torch.no_grad()
def inference_video(
    detector,
    tracker,
    input_video_path,
    frame_dir,
    cfg,
    vis_frame_dir=None,
    vis_hm_dir=None,
    vis_traj_path=None,
    dist_thresh=10.0,
    existing_traj_path=None,
):
    t_start = time.time()
    num_frames = 0
    print("Starting********")

    imgs_paths = sorted(Path(frame_dir).glob("*.png"))
    if not imgs_paths:
        raise ValueError(f"No extracted PNG frames found in {frame_dir}")

    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps <= 0:
        raise ValueError("Could not determine source video FPS")

    hm_results = defaultdict(list)
    if existing_traj_path is not None:
        result_dict = load_trajectory(existing_traj_path, imgs_paths, cfg["model"]["name"])
    else:
        c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        s = max(h, w) * 1.0
        trans = np.stack(
            [get_affine_transform(c, s, 0, [cfg["model"]["inp_width"], cfg["model"]["inp_height"]], inv=1) for _ in range(3)],
            axis=0,
        )
        trans = torch.tensor(trans)[None, :]
        preprocess_frame = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((cfg["model"]["inp_height"], cfg["model"]["inp_width"])),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        step = cfg["detector"]["step"]
        det_results = defaultdict(list)
        img_paths_buffer = []
        frames_buffer = []
        for img_path in imgs_paths:
            frame = cv2.imread(str(img_path))
            frames_buffer.append(frame)
            img_paths_buffer.append(str(img_path))
            if len(frames_buffer) == cfg["model"]["frames_in"]:
                frames_processed = [preprocess_frame(f) for f in frames_buffer]
                input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0)
                batch_results, hms_vis = detector.run_tensor(input_tensor, trans)
                for ie in batch_results[0].keys():
                    path = img_paths_buffer[ie]
                    preds = batch_results[0][ie]
                    det_results[path].extend(preds)
                    hm_results[path].extend(hms_vis[0][ie])
                if step == 1:
                    frames_buffer.pop(0)
                    img_paths_buffer.pop(0)
                elif step == 3:
                    img_paths_buffer = []
                    frames_buffer = []

        tracker.refresh()
        result_dict = {}
        print("Running tracker")
        for img_path, preds in det_results.items():
            result_dict[img_path] = tracker.update(preds)
        print("Finished tracking")

    t_elapsed = time.time() - t_start

    x_fin, y_fin, vis_fin = [], [], []
    if cfg["model"]["name"] == "blurball":
        l_fin, theta_fin = [], []

    vis_cfg = cfg.get("runner", {}).get("visualization", {})
    show_speed_direction = bool(vis_cfg.get("show_speed_direction", False))
    calibration_file = cfg.get("calibration_file", None)
    speed_window = max(3, int(vis_cfg.get("speed_window_frames", 7)))
    direction_window = max(3, int(vis_cfg.get("direction_window_frames", 5)))
    smoothing_alpha = float(vis_cfg.get("speed_smoothing_alpha", 0.25))
    direction_change_threshold_deg = float(vis_cfg.get("direction_change_threshold_deg", 115.0))
    direction_min_distance_m = float(vis_cfg.get("direction_min_distance_m", 0.015))
    min_speed_kmh = float(vis_cfg.get("direction_min_speed_kmh", 2.0))
    reversal_confirm_frames = max(1, int(vis_cfg.get("reversal_confirm_frames", 3)))
    direction_smoothing_alpha = float(vis_cfg.get("direction_smoothing_alpha", 0.35))
    hud_position = vis_cfg.get("hud_position", "top_center")

    speed_homography = None
    motion_estimator = None
    if show_speed_direction:
        if not calibration_file:
            print("Calibration not provided; speed/direction calculation is disabled")
            show_speed_direction = False
        else:
            speed_homography = load_speed_calibration(calibration_file)
            print("Loaded PongEye calibration from " + str(calibration_file))
            motion_estimator = MotionEstimator(
                fps=fps,
                speed_window_frames=speed_window,
                direction_window_frames=direction_window,
                speed_smoothing_alpha=smoothing_alpha,
                direction_change_threshold_deg=direction_change_threshold_deg,
                min_displacement_m=direction_min_distance_m,
                min_speed_kmh=min_speed_kmh,
                reversal_confirm_frames=reversal_confirm_frames,
                direction_smoothing_alpha=direction_smoothing_alpha,
            )

    for cnt, img_path in enumerate(result_dict.keys()):
        x_pred = result_dict[img_path]["x"]
        y_pred = result_dict[img_path]["y"]
        visi_pred = result_dict[img_path]["visi"]
        score_pred = result_dict[img_path]["score"]
        if cfg["model"]["name"] == "blurball":
            angle_pred = result_dict[img_path]["angle"]
            length_pred = result_dict[img_path]["length"]

        x_fin.append(int(min(max(x_pred, 0), 100000)))
        y_fin.append(int(min(max(y_pred, 0), 100000)))
        vis_fin.append(int(visi_pred))
        if cfg["model"]["name"] == "blurball":
            theta_fin.append(angle_pred)
            l_fin.append(length_pred)

        current_speed_kmh = 0.0
        current_direction_rad = None
        if motion_estimator is not None and visi_pred:
            table_position = project_to_table((float(x_pred), float(y_pred)), speed_homography)
            current_speed_kmh, current_direction_rad = motion_estimator.update(cnt, table_position)
        elif motion_estimator is not None:
            motion_estimator.reset()

        if not visi_pred:
            current_speed_kmh = 0.0
            current_direction_rad = None

        if vis_frame_dir is not None:
            vis_frame_path = osp.join(vis_frame_dir, osp.basename(img_path))
            vis_pred = cv2.imread(img_path)

            color_pred = (255, 0, 0)
            if cfg["model"]["name"] == "blurball":
                vis_pred = draw_frame(
                    vis_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                    angle=angle_pred,
                    l=length_pred,
                )
            else:
                vis_pred = draw_frame(
                    vis_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                )

            if show_speed_direction:
                vis_pred = draw_speed_direction_hud(
                    vis_pred,
                    current_speed_kmh,
                    current_direction_rad,
                    position=hud_position,
                )

            cv2.imwrite(vis_frame_path, vis_pred)

            if vis_hm_dir is not None:
                hm_path = osp.join(vis_hm_dir, osp.basename(img_path))
                if img_path in hm_results and hm_results[img_path]:
                    vis_hm_pred = cv2.cvtColor(
                        (255 * hm_results[img_path][0]["hm"]).astype(np.uint8),
                        cv2.COLOR_GRAY2RGB,
                    )
                    vis_hm_pred = cv2.resize(vis_hm_pred, (1280, 720))
                    vis_hm_pred = draw_frame(
                        vis_hm_pred,
                        center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                        color=color_pred,
                        radius=3,
                        angle=angle_pred if cfg["model"]["name"] == "blurball" else None,
                        l=length_pred if cfg["model"]["name"] == "blurball" else None,
                    )
                    if show_speed_direction:
                        vis_hm_pred = draw_speed_direction_hud(
                            vis_hm_pred,
                            current_speed_kmh,
                            current_direction_rad,
                            position=hud_position,
                        )
                    cv2.imwrite(hm_path, vis_hm_pred)

    if vis_frame_dir is not None:
        video_path = "{}.mp4".format(vis_frame_dir)
        gen_video(video_path, vis_frame_dir, fps=fps)
        print("Saving video at " + video_path)

    if existing_traj_path is None:
        if cfg["model"]["name"] == "blurball":
            df = pd.DataFrame(
                {
                    "Frame": x_fin,
                    "X": x_fin,
                    "Y": y_fin,
                    "Visibility": vis_fin,
                    "L": l_fin,
                    "Theta": theta_fin,
                }
            )
        else:
            df = pd.DataFrame({"Frame": x_fin, "X": x_fin, "Y": y_fin, "Visibility": vis_fin})
        df["Frame"] = df.index
        df.to_csv(osp.join(frame_dir, "traj.csv"), index=False)
        print("Saving csv at " + osp.join(frame_dir, "traj.csv"))

    return {"t_elapsed": t_elapsed, "num_frames": num_frames}


class NewVideosInferenceRunner(BaseRunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        runner_cfg = cfg["runner"]
        self._mode = str(runner_cfg.get("mode", "standard"))
        self._keep_extracted_frames = bool(runner_cfg.get("keep_extracted_frames", True))
        self._vis_result = bool(runner_cfg.get("vis_result", True))
        self._vis_hm = bool(runner_cfg.get("vis_hm", True))
        self._vis_traj = bool(runner_cfg.get("vis_traj", False))
        self._input_vid_path = Path(cfg["input_vid"])

        if self._mode not in {"standard", "trajectory_only"}:
            raise ValueError(
                f"Unsupported runner.mode={self._mode!r}; expected 'standard' or 'trajectory_only'"
            )
        if self._mode == "trajectory_only":
            self._vis_result = False
            self._vis_hm = False
            self._vis_traj = False

    def run(self, model=None, model_dir=None):
        return self._run_model(model=model)

    def _run_model(self, model=None):
        # BlurBall requires CUDA. Select it automatically on machines with an NVIDIA GPU.
        if torch.cuda.is_available():
            self._cfg["runner"]["device"] = "cuda"
            self._cfg["runner"]["gpus"] = [0]
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._cfg["runner"]["device"] = "cuda"
            print("CUDA is not available; BlurBall requires an NVIDIA CUDA GPU")

        frame_dir = self._input_vid_path.parent / ("frames_" + self._input_vid_path.stem)
        frame_pngs = list(frame_dir.glob("*.png")) if frame_dir.is_dir() else []
        extracted_here = False
        if frame_pngs:
            print(f"Reusing extracted frames: {frame_dir} ({len(frame_pngs)} PNGs)")
        else:
            frame_dir = Path(process_video(self._input_vid_path))
            extracted_here = True
            print("Finished preprocess_video")

        traj_path = frame_dir / "traj.csv"
        reuse_traj = traj_path.is_file()
        detector = None
        tracker = None
        if reuse_traj:
            print(f"Reusing existing trajectory: {traj_path}")
        else:
            detector = build_detector(self._cfg, model=model)
            tracker = build_tracker(self._cfg)

        t_elapsed_all = 0.0
        num_frames_all = 0

        vis_frame_dir, vis_hm_dir, vis_traj_path = None, None, None
        if self._vis_result:
            vis_frame_dir = osp.join(self._input_vid_path.parent, "frames")
            mkdir_if_missing(vis_frame_dir)
        if self._vis_hm:
            vis_hm_dir = osp.join(self._input_vid_path.parent, "hm")
            mkdir_if_missing(vis_hm_dir)

        tmp = inference_video(
            detector,
            tracker,
            self._input_vid_path,
            frame_dir,
            self._cfg,
            vis_frame_dir=vis_frame_dir,
            vis_hm_dir=vis_hm_dir,
            existing_traj_path=traj_path if reuse_traj else None,
        )
        t_elapsed_all += tmp["t_elapsed"]
        num_frames_all += tmp["num_frames"]

        if self._mode == "trajectory_only" and extracted_here and not self._keep_extracted_frames:
            if frame_dir.exists():
                print(f"Removing temporary extracted frames: {frame_dir}")
                shutil.rmtree(frame_dir)

        return
