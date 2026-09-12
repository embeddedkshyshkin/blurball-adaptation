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

    homography = cv2.getPerspectiveTransform(image_points, table_points)
    return homography


def project_to_table(point_xy, homography):
    point = np.array([[point_xy]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, homography)[0, 0]
    return float(projected[0]), float(projected[1])


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
):
    frames_in = detector.frames_in
    frames_out = detector.frames_out
    t_start = time.time()

    det_results = []
    hm_results = []
    num_frames = 0
    print("Starting********")

    imgs_paths = sorted(Path(frame_dir).glob("*.png"))
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps <= 0:
        raise ValueError("Could not determine source video FPS")

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
    hm_results = defaultdict(list)
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
    cm_pred = plt.get_cmap("Reds", len(result_dict))

    x_fin, y_fin, vis_fin = [], [], []
    if cfg["model"]["name"] == "blurball":
        l_fin, theta_fin = [], []

    vis_cfg = cfg.get("runner", {}).get("visualization", {})
    show_speed_direction = bool(vis_cfg.get("show_speed_direction", False))
    calibration_file = vis_cfg.get("calibration_file", None)
    speed_window = max(1, int(vis_cfg.get("speed_window_frames", 4)))
    smoothing_alpha = float(vis_cfg.get("speed_smoothing_alpha", 0.35))
    hud_position = vis_cfg.get("hud_position", "top_center")

    speed_homography = None
    if show_speed_direction and calibration_file:
        speed_homography = load_speed_calibration(calibration_file)
        print("Loaded PongEye calibration from " + str(calibration_file))

    recent_positions = []
    smoothed_speed_kmh = None

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

        current_speed_kmh = None
        current_direction_rad = None
        if show_speed_direction and visi_pred:
            current_position = (float(x_pred), float(y_pred))
            if speed_homography is not None:
                table_position = project_to_table(current_position, speed_homography)
            else:
                table_position = current_position

            recent_positions.append((cnt, current_position, table_position))
            if len(recent_positions) > speed_window + 1:
                recent_positions.pop(0)

            if len(recent_positions) >= 2:
                first_frame, first_image_pos, first_table_pos = recent_positions[0]
                dt = (cnt - first_frame) / fps
                if dt > 0:
                    dx_table = table_position[0] - first_table_pos[0]
                    dy_table = table_position[1] - first_table_pos[1]
                    distance_m = float(np.hypot(dx_table, dy_table))
                    if distance_m > 0:
                        raw_speed_kmh = distance_m / dt * 3.6
                        if smoothed_speed_kmh is None:
                            smoothed_speed_kmh = raw_speed_kmh
                        else:
                            smoothed_speed_kmh = (
                                smoothing_alpha * raw_speed_kmh
                                + (1.0 - smoothing_alpha) * smoothed_speed_kmh
                            )
                        current_speed_kmh = smoothed_speed_kmh

                    # The arrow is rendered in image coordinates so it points
                    # along the visible ball trajectory in the video.
                    dx_image = current_position[0] - first_image_pos[0]
                    dy_image = current_position[1] - first_image_pos[1]
                    if np.hypot(dx_image, dy_image) > 0:
                        current_direction_rad = float(np.arctan2(dy_image, dx_image))

        if vis_frame_dir is not None:
            vis_frame_path = osp.join(vis_frame_dir, osp.basename(img_path))
            hm_path = osp.join(vis_hm_dir, osp.basename(img_path))
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
                vis_hm_pred = cv2.cvtColor((255 * hm_results[img_path][0]["hm"]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                vis_hm_pred = cv2.resize(vis_hm_pred, (1280, 720))
                vis_hm_pred = draw_frame(
                    vis_hm_pred,
                    center=Center(is_visible=visi_pred, x=x_pred, y=y_pred),
                    color=color_pred,
                    radius=3,
                    angle=angle_pred,
                    l=length_pred,
                )
            else:
                vis_pred = draw_frame(vis_pred, center=Center(is_visible=visi_pred, x=x_pred, y=y_pred), color=color_pred, radius=3)
                vis_hm_pred = cv2.cvtColor((255 * hm_results[img_path][0]["hm"]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                vis_hm_pred = draw_frame(vis_hm_pred, center=Center(is_visible=visi_pred, x=x_pred, y=y_pred), color=color_pred, radius=3)

            if current_speed_kmh is not None and current_direction_rad is not None:
                vis_pred = draw_speed_direction_hud(
                    vis_pred,
                    current_speed_kmh,
                    current_direction_rad,
                    position=hud_position,
                )

            cv2.imwrite(vis_frame_path, vis_pred)
            cv2.imwrite(hm_path, vis_hm_pred)

    if vis_frame_dir is not None:
        video_path = "{}.mp4".format(vis_frame_dir)
        gen_video(video_path, vis_frame_dir, fps=fps)
        print("Saving video at " + video_path)

    if cfg["model"]["name"] == "blurball":
        df = pd.DataFrame({"Frame": x_fin, "X": x_fin, "Y": y_fin, "Visibility": vis_fin, "L": l_fin, "Theta": theta_fin})
    else:
        df = pd.DataFrame({"Frame": x_fin, "X": x_fin, "Y": y_fin, "Visibility": vis_fin})
    df["Frame"] = df.index
    df.to_csv(osp.join(frame_dir, "traj.csv"), index=False)
    print("Saving csv at " + osp.join(frame_dir, "traj.csv"))
    return {"t_elapsed": t_elapsed, "num_frames": num_frames}


class NewVideosInferenceRunner(BaseRunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._vis_result = cfg["runner"]["vis_result"]
        self._vis_hm = cfg["runner"]["vis_hm"]
        self._vis_traj = cfg["runner"]["vis_traj"]
        self._input_vid_path = Path(cfg["input_vid"])

    def run(self, model=None, model_dir=None):
        return self._run_model(model=model)

    def _run_model(self, model=None):
        detector = build_detector(self._cfg, model=model)
        tracker = build_tracker(self._cfg)
        frame_dir = process_video(self._input_vid_path)
        print("Finished preprocess_video")
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
        )
        t_elapsed_all += tmp["t_elapsed"]
        num_frames_all += tmp["num_frames"]
        return
