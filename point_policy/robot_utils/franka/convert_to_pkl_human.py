import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))    # point_policy/ (point_utils, cfgs, …)
sys.path.insert(0, str(_here))               # franka/utils.py takes priority over point_policy/utils.py

import re
import yaml
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import torch
import numpy as np
from pandas import read_csv
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from point_utils.points_class import PointsClass
from point_utils import task_pkl_io
from utils import (
    camera2pixelkey,
    pixel2d_to_3d_torch,
    triangulate_points,
)

# Create the parser
parser = argparse.ArgumentParser(
    description="Convert processed human data into a pkl file"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument("--calib_path", type=str, help="Path to the calibration file")
parser.add_argument("--task_names", nargs="+", type=str, help="List of task names")
parser.add_argument(
    "--num_demos", type=int, default=None, help="Number of demonstrations to process"
)
parser.add_argument("--process_points", action="store_true", help="Process key points")
parser.add_argument(
    "--use_gt_depth", action="store_true", help="Use ground truth depth"
)


args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_names = args.task_names
NUM_DEMOS = args.num_demos
process_points = args.process_points
use_gt_depth = args.use_gt_depth

camera_indices = [1, 2]
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = None
object_labels = [
    "human_hand",
    "objects",
]

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data"
SAVE_DATA_PATH = Path(DATA_DIR) / "processed_data_pkl"

if save_img_size is None:
    save_img_size = (
        int(original_img_size[0] * (crop_w[1] - crop_w[0])),
        int(original_img_size[1] * (crop_h[1] - crop_h[0])),
    )

assert len(task_names) == 1, "Only one task name is supported for now"

if task_names is None:
    task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Calibration data
calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

episode_list = {}
for cam_idx in camera_indices:
    pixel_key = f"pixels{cam_idx}"
    episode_list[pixel_key] = []


# Load p3po config for tracking
if process_points:
    _cfg_path = Path(__file__).resolve().parents[2] / "cfgs" / "suite" / "points_cfg.yaml"
    with open(_cfg_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        root_dir, dift_path, cotracker_checkpoint = (
            cfg["root_dir"],
            cfg["dift_path"],
            cfg["cotracker_checkpoint"],
        )
        cfg["dift_path"] = f"{root_dir}/{dift_path}"
        cfg["cotracker_checkpoint"] = f"{root_dir}/{cotracker_checkpoint}"
        cfg["tapir_checkpoint"] = f"{root_dir}/{cfg['tapir_checkpoint']}"
        cfg["task_name"] = task_names[0]
        cfg["pixel_keys"] = [
            camera2pixelkey[f"cam_{cam_idx}"] for cam_idx in camera_indices
        ]
        cfg["object_labels"] = object_labels
        cfg["use_gt_depth"] = use_gt_depth

    points_class = PointsClass(**cfg)


# 2D track smoothing applied before triangulation.
# Filtering in 2D before triangulation is important: a single-camera flicker
# in one frame produces a bad ray, which contaminates the entire triangulated 3D point.
TRACK_MEDIAN_WINDOW = 5    # frames: removes isolated spike detections
TRACK_SAVGOL_WINDOW = 11   # frames (odd): smooths continuous jitter
TRACK_SAVGOL_ORDER  = 3


def smooth_2d_tracks(tracks):
    """Apply median + Savitzky-Golay to (T, N, 2) 2D track array."""
    T = tracks.shape[0]
    tracks = tracks.astype(np.float64)
    win_m = min(TRACK_MEDIAN_WINDOW, T)
    tracks = median_filter(tracks, size=(win_m, 1, 1))
    win_s = min(TRACK_SAVGOL_WINDOW, T)
    win_s = win_s if win_s % 2 == 1 else win_s - 1
    if win_s >= TRACK_SAVGOL_ORDER + 1:
        tracks = savgol_filter(tracks, window_length=win_s,
                               polyorder=TRACK_SAVGOL_ORDER, axis=0)
    return tracks


def extract_number(s):
    s = s.strip()
    # Remove any leading/trailing brackets or whitespace
    s = s.strip("[]")
    # Match 'np.float32(number)' or 'np.float64(number)'
    match = re.match(r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    else:
        # Match plain numbers, including negatives and decimals
        match = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"Cannot extract number from '{s}'")


for TASK_NAME in task_names:
    DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")
    if use_gt_depth:
        TASK_NAME = f"{TASK_NAME}_gt_depth"

    # Task pkl is a directory of per-demo files (see point_utils/task_pkl_io.py)
    # rather than one big pickle -- avoids holding every demo's decoded video
    # frames in memory simultaneously (OOM risk with enough demos).
    task_pkl_dir = SAVE_DATA_PATH / TASK_NAME

    if task_pkl_dir.exists() and not process_points:
        print(f"Data for {TASK_NAME} already exists. Appending to it...")
        input("Press Enter to continue...")
        meta = task_pkl_io.read_meta(task_pkl_dir)
        max_cartesian = meta["max_cartesian"]
        min_cartesian = meta["min_cartesian"]
        max_gripper = meta["max_gripper"]
        min_gripper = meta["min_gripper"]
        # Raw (non-tracking) pass only needs to add demos that aren't already
        # on disk -- existing demo files are never re-read or rewritten.
        existing_demo_ids = set(task_pkl_io.iter_demo_ids(task_pkl_dir))
    else:
        max_cartesian, min_cartesian = None, None
        max_gripper, min_gripper = None, None
        existing_demo_ids = set()

    dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
    # Sort numerically by the trailing demonstration_N id, not lexicographically
    # as strings (which would order demonstration_10 before demonstration_2).
    dirs = sorted(dirs, key=lambda p: int(str(p).split("_")[-1]))

    for i, data_point in enumerate(dirs):
        print(f"Processing data point {i+1}/{len(dirs)}")

        demo_num = int(str(data_point).split("_")[-1])

        if NUM_DEMOS is not None and demo_num >= NUM_DEMOS:
            print(f"Skipping data point {data_point}")
            continue

        if not process_points and demo_num in existing_demo_ids:
            print(f"Skipping data point {data_point} (already processed)")
            continue

        observation = {}
        image_dir = data_point / "videos"
        if not image_dir.exists():
            print(f"Data point {data_point} is incomplete")
            continue

        for save_idx, idx in enumerate(camera_indices):
            video_path = image_dir / f"camera{idx}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Video {video_path} could not be opened")
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # crop the image
                h, w, _ = frame.shape
                frame = frame[
                    int(h * crop_h[0]) : int(h * crop_h[1]),
                    int(w * crop_w[0]) : int(w * crop_w[1]),
                ]

                frame = cv2.resize(frame, save_img_size)
                frames.append(frame)

            observation[f"pixels{idx}"] = np.array(frames)

        # Process depth
        if use_gt_depth:
            depth_dir = data_point / "depth"
            if not depth_dir.exists():
                print(f"Data point {data_point} is incomplete (no depth)")
                continue

            for idx in camera_indices:
                depth_file = depth_dir / f"depth{idx}.pkl"
                with open(depth_file, "rb") as f:
                    depth = pkl.load(f)  # depth in meters
                camera_name = f"cam_{idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"depth_{pixel_key}"] = depth

        state_csv_path = data_point / "states.csv"
        state = read_csv(state_csv_path)

        # Parsing cartesian pose data
        cartesian_states = state["pose_aa"].values
        cartesian_states = np.array(
            [
                np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                for pose in cartesian_states
            ],
            dtype=np.float32,
        )

        gripper_states = state["gripper_state"].values.astype(np.float32)
        observation["cartesian_states"] = cartesian_states.astype(np.float32)
        observation["gripper_states"] = gripper_states.astype(np.float32)

        if process_points:
            # Human hand tracks
            mark_every = 8
            save = True
            for cam_idx in camera_indices:
                if save == False:
                    break
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]

                frames = observation[pixel_key]
                # CV2 reads in BGR format, so we need to convert to RGB
                frames = [frame[..., ::-1] for frame in frames]
                points_class.add_to_image_list(frames[0], pixel_key)
                for object_label in object_labels:
                    points_class.find_semantic_similar_points(pixel_key, object_label)
                try:
                    points_class.track_points(
                        pixel_key, last_n_frames=mark_every, is_first_step=True
                    )
                except Exception:
                    import traceback

                    print(f"Error in tracking hand points for {pixel_key} -- skipping this demo:")
                    traceback.print_exc()
                    points_class.reset_episode()
                    save = False
                    continue

                points_class.track_points(
                    pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1)
                )

                points_list = []
                points = points_class.get_points_on_image(pixel_key)
                points_list.append(points[0])

                if use_gt_depth:
                    points_3d_list = []
                    if use_gt_depth:
                        depth = observation[f"depth_{pixel_key}"][0]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )
                    else:
                        points_class.get_depth()
                    points_with_depth = points_class.get_points(pixel_key)
                    depths = points_with_depth[:, :, -1]

                    P = calibration_data[camera_name]["ext"]  # 4x4
                    K = calibration_data[camera_name]["int"]  # 3x3
                    points3d = pixel2d_to_3d_torch(points[0], depths[0], K, P)
                    points_3d_list.append(points3d)

                for idx, image in enumerate(frames[1:]):
                    print(f"Traj: {i}, Frame: {idx}, Image: {pixel_key}")
                    points_class.add_to_image_list(image, pixel_key)

                    if use_gt_depth:
                        depth = observation[f"depth_{pixel_key}"][idx]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )

                    if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                        to_add = mark_every - (idx + 1) % mark_every
                        if to_add < mark_every:
                            for j in range(to_add):
                                points_class.add_to_image_list(image, pixel_key)
                        else:
                            to_add = 0

                        points_class.track_points(
                            pixel_key,
                            last_n_frames=mark_every,
                            one_frame=(mark_every == 1),
                        )

                        points = points_class.get_points_on_image(
                            pixel_key, last_n_frames=mark_every
                        )
                        for j in range(mark_every - to_add):
                            points_list.append(points[j])

                        if use_gt_depth:
                            if not use_gt_depth:
                                points_class.get_depth(last_n_frames=mark_every)

                            points_with_depth = points_class.get_points(
                                pixel_key, last_n_frames=mark_every
                            )
                            for j in range(mark_every - to_add):
                                depth = points_with_depth[j, :, -1]
                                points3d = pixel2d_to_3d_torch(points[j], depth, K, P)
                                points_3d_list.append(points3d)

                observation[f"human_tracks_{pixel_key}"] = torch.stack(
                    points_list
                ).numpy()
                if use_gt_depth:
                    observation[f"human_tracks_3d_{pixel_key}"] = torch.stack(
                        points_3d_list
                    ).numpy()
                points_class.reset_episode()

            if save == False:
                continue

        # Update max and min cartesian values for normalization
        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))

        # Update max and min gripper values for normalization
        if max_gripper is None:
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

        if process_points and not use_gt_depth:
            """
            Triangulate 3D points from 2D points when gt_depth is not available
            """
            # Smooth 2D tracks per camera before triangulation so that a
            # single-camera flicker doesn't corrupt the triangulated 3D point.
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                raw = observation[f"human_tracks_{pixel_key}"]  # (T, N, 2or3)
                smoothed = smooth_2d_tracks(raw[:, :, :2])       # filter x,y only
                if raw.shape[2] > 2:                             # preserve depth col if present
                    smoothed = np.concatenate([smoothed, raw[:, :, 2:]], axis=2)
                observation[f"human_tracks_{pixel_key}"] = smoothed

            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"human_tracks_3d_{pixel_key}"] = []
            for t_idx in range(
                len(observation[f"human_tracks_{pixel_key}"])
            ):  # for each frame
                P, pts = [], []
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    extr = calibration_data[camera_name]["ext"]
                    intr = calibration_data[camera_name]["int"]
                    intr = np.concatenate([intr, np.zeros((3, 1))], axis=1)
                    P.append(intr @ extr)

                    pt2d = observation[f"human_tracks_{pixel_key}"][t_idx]

                    # compute point_h in original image
                    point_h = pt2d[:, 1]
                    h_orig, h_curr = original_img_size[1], save_img_size[1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = (
                        point_h / h_curr
                    ) * h_orig_cropped + h_orig * crop_h[0]
                    point_h_orig = point_h_orig.astype(np.int32)

                    # compute point_w in original image
                    point_w = pt2d[:, 0]
                    w_orig, w_curr = original_img_size[0], save_img_size[0]
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = (
                        point_w / w_curr
                    ) * w_orig_cropped + w_orig * crop_w[0]
                    point_w_orig = point_w_orig.astype(np.int32)

                    pt2d = np.column_stack((point_w_orig, point_h_orig))

                    # Undistort before triangulation: the projection matrix P = K @ E
                    # assumes undistorted pixel coordinates. Passing raw distorted pixels
                    # introduces error proportional to lens distortion (especially near
                    # image edges). cv2.undistortPoints with P=K returns undistorted
                    # pixel coordinates in the same pixel space.
                    K_orig = calibration_data[camera_name]["int"]
                    D_orig = calibration_data[camera_name]["dist_coeff"]
                    pt2d = cv2.undistortPoints(
                        pt2d.reshape(-1, 1, 2).astype(np.float32),
                        K_orig, D_orig, P=K_orig
                    ).reshape(-1, 2)

                    pts.append(pt2d)

                pts3d = triangulate_points(P, pts)
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]
                    observation[f"human_tracks_3d_{pixel_key}"].append(pts3d[:, :3])
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"human_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"human_tracks_3d_{pixel_key}"]
                )

        task_pkl_io.write_demo(task_pkl_dir, demo_num, observation)

    task_pkl_io.write_meta(task_pkl_dir, {
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
    })

print("Processing complete.")
