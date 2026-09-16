import os
import os.path as osp
import matplotlib.pyplot as plt
import shutil
import tempfile
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
import time
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from utils.ball_kinematics import BallKinematicsEstimator, load_calibration

from .base import BaseRunner


def load_speed_calibration(calibration_file):
    """Load calibration and return the image -> table-plane homography.

    ``load_calibration`` also retains K/R/t for the gravity-constrained solver.
    This compatibility wrapper keeps the current 2-D motion fallback unchanged.
    """
    if not calibration_file:
        return None

    path = Path(calibration_file)
    if not path.is_absolute():
        path = Path(HydraConfig.get().runtime.cwd) / path

    return load_calibration(path).H_image_to_table


def _motion_states_from_trajectory(result_dict, fps, calibration_file, vis_cfg):
    """Run the existing gravity-constrained solver over one segment trajectory."""
    if not calibration_file:
        return None
    try:
        calibration = load_calibration(calibration_file)
    except Exception as exc:
        print(f"3D kinematics unavailable: {exc}")
        return None
    if not calibration.has_pose:
        return None

    paths = list(result_dict)
    frames = np.arange(len(paths))
    uv = np.array([[result_dict[path]["x"], result_dict[path]["y"]] for path in paths], dtype=float)
    visibility = np.array([bool(result_dict[path]["visi"]) for path in paths])
    estimator = BallKinematicsEstimator(
        calibration, fps=float(fps),
        half_window=max(2, int(vis_cfg.get("speed_window_frames", 9)) // 2),
        degree=min(2, max(1, int(vis_cfg.get("kinematics_polynomial_degree", 2)))),
        min_track=max(8, int(vis_cfg.get("kinematics_min_track", 12))),
        margin=float(vis_cfg.get("kinematics_plane_margin_m", 1.0)),
        gravity_weight=float(vis_cfg.get("kinematics_gravity_weight", 3.0)),
    )
    speed, heading_deg, mode, _ = estimator.estimate(frames, uv, visibility)
    if not np.any(mode == "vertical"):
        return None
    return {
        path: (float(speed[i]), float(np.deg2rad(heading_deg[i])) if np.isfinite(heading_deg[i]) else None)
        for i, path in enumerate(paths) if visibility[i]
    }


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
