"""
Standalone HaMeR hand-keypoint detector -- this is Phase A of
convert_to_pkl_human_hamer.py, split into its own script so it can be run
once and reused across multiple Step 2 (object-tracking) runs -- e.g. adding
a new --object_labels group later -- without re-running HaMeR, which is the
GPU-heavy, slow part of Step 2.

Output: processed_data_pkl/{task_name}[_gt_depth]_hand_tracks/demo_NNNN.pkl,
one file per demo -- {"hand_tracks": {pixel_key: (T,9,2)}, "cartesian_states",
"gripper_states"}, the same structure convert_to_pkl_human_hamer.py's own
Phase A used to keep only in memory (phase_a_cache).

STATUS: this script writes the cache, but nothing reads it yet.
convert_to_pkl_human_hamer.py still runs its own inline Phase A and does not
consult this directory -- wiring that up is part of the planned 3-way split of
that script.

If your task has ALREADY been through Step 2, you do not need this script at
all: retrack_objects.py re-runs only the DIFT+TAPIR object tracking and lifts
the hand track straight out of the existing task pkl (its first 9 points),
which is the same HaMeR output this cache would hold. Use this script only for
tasks that have never been tracked.

Usage:
    cd point_policy/robot_utils/franka
    python detect_hand_tracks_hamer.py \
        --data_dir /path/to/data --task_names peg_hole_15

    # Re-detect even for demos already cached (e.g. after changing --hand_side)
    python detect_hand_tracks_hamer.py \
        --data_dir /path/to/data --task_names peg_hole_15 \
        --hand_side left --force
"""

import argparse
import gc
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from pandas import read_csv

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))  # point_policy/ (point_utils)
sys.path.insert(0, str(_here))             # franka/utils.py (camera2pixelkey)

from point_utils import task_pkl_io  # noqa: E402
from utils import camera2pixelkey  # noqa: E402

# ---------------------------------------------------------------------------
# HaMeR subprocess worker -- identical to convert_to_pkl_human_hamer.py's copy.
# Duplicated rather than imported: that script is straight-line, side-effecting
# module code (parses argparse and runs its whole pipeline at import time), so
# importing from it would trigger its entire Step 2 run.
# ---------------------------------------------------------------------------
REPO_ROOT = _here.parents[2]  # .../Point-Policy
HAMER_DIR = REPO_ROOT / "hamer"
HAMER_WORKER_SCRIPT = HAMER_DIR / "hamer_hand_worker.py"
HAMER_PYTHON = os.environ.get("HAMER_PYTHON", str(
    HAMER_DIR / ".hamer_venv" / "bin" / "python"))

MIN_FREE_GPU_MIB = 3000
GPU_WAIT_TIMEOUT_S = 60
GPU_POLL_INTERVAL_S = 2

camera_indices = [1, 2]
NUM_HAND_POINTS = 9


def _free_gpu_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_free_gpu_memory(min_free_mib=MIN_FREE_GPU_MIB, timeout_s=GPU_WAIT_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while True:
        free_mib = _free_gpu_mib()
        if free_mib is None or free_mib >= min_free_mib:
            return
        if time.monotonic() >= deadline:
            print(f"Warning: only {free_mib}MiB GPU free after waiting {timeout_s}s "
                  f"(wanted {min_free_mib}MiB) -- proceeding anyway.")
            return
        time.sleep(GPU_POLL_INTERVAL_S)


def release_main_process_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def detect_hand_tracks_hamer(frames_rgb, hand_side=None, vitpose_on_gpu=False, body_detector=None):
    release_main_process_gpu_memory()
    wait_for_free_gpu_memory()
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.npy"
        out_path = Path(tmp_dir) / "hand_tracks.npy"
        np.save(frames_path, np.asarray(frames_rgb, dtype=np.uint8))

        cmd = [HAMER_PYTHON, "-I", str(HAMER_WORKER_SCRIPT),
               "--frames_npy", str(frames_path), "--out_npy", str(out_path)]
        if hand_side is not None:
            cmd += ["--hand", hand_side]
        if vitpose_on_gpu:
            cmd += ["--vitpose_on_gpu"]
        if body_detector is not None:
            cmd += ["--body_detector", body_detector]
        subprocess.run(cmd, cwd=str(HAMER_DIR), check=True)
        return np.load(out_path)


# ---------------------------------------------------------------------------
# Per-demo I/O helpers -- same as convert_to_pkl_human_hamer.py's copies,
# minus anything object-tracking/calibration related (not needed here).
# ---------------------------------------------------------------------------
def load_demo_frames(data_point, crop_h, crop_w, save_img_size):
    image_dir = data_point / "videos"
    if not image_dir.exists():
        return None

    frames_by_key = {}
    for idx in camera_indices:
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
            h, w, _ = frame.shape
            frame = frame[
                int(h * crop_h[0]): int(h * crop_h[1]),
                int(w * crop_w[0]): int(w * crop_w[1]),
            ]
            frame = cv2.resize(frame, save_img_size)
            frames.append(frame)

        camera_name = f"cam_{idx}"
        pixel_key = camera2pixelkey[camera_name]
        frames_by_key[pixel_key] = np.array(frames)
    return frames_by_key


def demo_depth_complete(data_point):
    depth_dir = data_point / "depth"
    if not depth_dir.exists():
        return False
    return all((depth_dir / f"depth{idx}.pkl").exists() for idx in camera_indices)


def extract_number(s):
    s = s.strip().strip("[]")
    match = re.match(
        r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    match = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
    if match:
        return float(match.group(0))
    raise ValueError(f"Cannot extract number from '{s}'")


def load_demo_state(data_point):
    state_csv_path = data_point / "states.csv"
    state = read_csv(state_csv_path)
    cartesian_states = state["pose_aa"].values
    cartesian_states = np.array(
        [np.array([extract_number(x) for x in pose.strip("[]").split(",")])
         for pose in cartesian_states],
        dtype=np.float32,
    )
    gripper_states = state["gripper_state"].values.astype(np.float32)
    return cartesian_states, gripper_states


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone HaMeR hand-keypoint detection, cached for reuse "
                    "across multiple object-tracking runs of convert_to_pkl_human_hamer.py."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--task_names", nargs="+", type=str, required=True, help="List of task names")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of demonstrations to process")
    parser.add_argument("--use_gt_depth", action="store_true",
                        help="Only affects which demos are considered complete (gates on "
                             "depth availability, matching Step 2's own gating) -- HaMeR "
                             "itself never uses depth.")
    parser.add_argument("--hand_side", type=str, default=None, choices=["left", "right"],
                        help="Force HaMeR to track this hand instead of auto-picking.")
    parser.add_argument("--vitpose_on_gpu", action="store_true",
                        help="Run ViTPose+-Huge on GPU instead of CPU.")
    parser.add_argument("--body_detector", type=str, default=None, choices=["vitdet", "regnety"],
                        help="Person detector backbone for HaMeR's hand-bbox localization step.")
    parser.add_argument("--force", action="store_true",
                        help="Re-detect even for demos already cached (e.g. after changing "
                             "--hand_side/--body_detector).")
    return parser.parse_args()


def main():
    args = parse_args()
    DATA_DIR = Path(args.data_dir)
    PROCESSED_DATA_PATH = DATA_DIR / "processed_data"
    SAVE_DATA_PATH = DATA_DIR / "processed_data_pkl"
    original_img_size = (640, 480)
    crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
    save_img_size = (
        int(original_img_size[0] * (crop_w[1] - crop_w[0])),
        int(original_img_size[1] * (crop_h[1] - crop_h[0])),
    )

    for TASK_NAME in args.task_names:
        DATASET_PATH = PROCESSED_DATA_PATH / TASK_NAME
        task_name = f"{TASK_NAME}_gt_depth" if args.use_gt_depth else TASK_NAME
        cache_dir = SAVE_DATA_PATH / f"{task_name}_hand_tracks"

        existing_ids = set() if args.force else set(task_pkl_io.iter_demo_ids(cache_dir))
        if existing_ids:
            print(f"[{task_name}] {len(existing_ids)} demo(s) already cached in "
                  f"{cache_dir} -- skipping those (use --force to redo).")

        dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
        # Sort numerically by the trailing demonstration_N id, matching
        # convert_to_pkl_human_hamer.py's own ordering.
        dirs = sorted(dirs, key=lambda p: int(str(p).split("_")[-1]))

        for i, data_point in enumerate(dirs):
            demo_num = int(str(data_point).split("_")[-1])
            if args.num_demos is not None and demo_num >= args.num_demos:
                continue
            if demo_num in existing_ids:
                continue

            print(f"[{task_name}] Processing demo {demo_num} ({i + 1}/{len(dirs)})")
            frames_by_key = load_demo_frames(data_point, crop_h, crop_w, save_img_size)
            if frames_by_key is None:
                print(f"  {data_point} is incomplete (no videos) -- skipping")
                continue
            if args.use_gt_depth and not demo_depth_complete(data_point):
                print(f"  {data_point} is incomplete (no depth) -- skipping")
                continue

            cartesian_states, gripper_states = load_demo_state(data_point)

            hand_tracks_by_key = {}
            ok = True
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                frames_rgb = frames_by_key[pixel_key][..., ::-1]  # BGR -> RGB
                try:
                    hand_tracks_by_key[pixel_key] = detect_hand_tracks_hamer(
                        np.array(frames_rgb, dtype=np.uint8), hand_side=args.hand_side,
                        vitpose_on_gpu=args.vitpose_on_gpu, body_detector=args.body_detector,
                    )
                except Exception:
                    print(f"  Error in HaMeR hand tracking for {pixel_key} -- skipping this demo:")
                    traceback.print_exc()
                    ok = False
                    break
            if not ok:
                continue

            task_pkl_io.write_demo(cache_dir, demo_num, {
                "hand_tracks": hand_tracks_by_key,
                "cartesian_states": cartesian_states.astype(np.float32),
                "gripper_states": gripper_states.astype(np.float32),
            })

        n_cached = len(task_pkl_io.iter_demo_ids(cache_dir))
        print(f"[{task_name}] Done -- {n_cached} demo(s) cached in {cache_dir}")


if __name__ == "__main__":
    main()
