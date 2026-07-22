"""
HaMeR-based variant of convert_to_pkl_human.py.

The only difference from convert_to_pkl_human.py is how the human hand is
tracked: instead of MediaPipe (inside PointsClass), this script shells out
to hamer_hand_worker.py running in its own venv (../../../hamer/.hamer_venv
by default -- override with the HAMER_PYTHON env var), since HaMeR's
dependencies (old torch/mmcv/detectron2 pins) are incompatible with this
project's environment. Object-point tracking (DIFT/TAPIR) is untouched --
PointsClass is instantiated with object_labels=["objects"] only, so it never
imports MediaPipe.

The resulting human_tracks_{pixel_key} array keeps the exact same layout as
the MediaPipe version: 9 hand points (wrist, index MCP->TIP, thumb CMC->TIP)
followed by the object points, in pixel coordinates -- so nothing
downstream (convert_pkl_human_to_robot.py etc.) needs to change.

GPU memory note: all HaMeR subprocess passes for a demo (one per camera) are
now run up front, before any object tracking (DIFT/CoTracker) touches the
GPU in the main process. This avoids a VRAM-fragmentation crash that could
occur when HaMeR and the main process's tracking models were interleaved
per-camera: PyTorch's caching allocator does not return freed VRAM to the
driver, so nvidia-smi could still report the previous camera's tracking
activations as "used" even after they were logically freed, causing
wait_for_free_gpu_memory() to time out and the next HaMeR worker to OOM.
Running all HaMeR passes first, then all tracking passes, means the two
GPU consumers are no longer interleaved within a demo. gc.collect() +
torch.cuda.empty_cache() calls are also added at natural boundaries as a
second line of defense.

This is a temporary/experimental script -- see ur5e_pipeline/run_pipeline.sh
for how it's wired in in place of convert_to_pkl_human.py.
"""

from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from pandas import read_csv
import numpy as np
import torch
import cv2
import tempfile
import subprocess
import pickle as pkl
import argparse
import yaml
import time
import re
import os
import gc
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
# point_policy/ (point_utils, cfgs, …)
sys.path.insert(0, str(_here.parents[1]))
# franka/utils.py takes priority over point_policy/utils.py
sys.path.insert(0, str(_here))

from utils import (
    camera2pixelkey,
    pixel2d_to_3d_torch,
    triangulate_points,
)
from point_utils import task_pkl_io
from point_utils.points_class import PointsClass


# ---------------------------------------------------------------------------
# HaMeR subprocess worker (runs in its own venv -- see hamer/hamer_hand_worker.py)
# ---------------------------------------------------------------------------
REPO_ROOT = _here.parents[2]  # .../Point-Policy
HAMER_DIR = REPO_ROOT / "hamer"
HAMER_WORKER_SCRIPT = HAMER_DIR / "hamer_hand_worker.py"
HAMER_PYTHON = os.environ.get("HAMER_PYTHON", str(
    HAMER_DIR / ".hamer_venv" / "bin" / "python"))

# GPU memory the worker needs resident at once (HaMeR + detectron2 + cuDNN/
# cuBLAS workspace) is a few GB on an 8GB card. When one camera's worker
# subprocess exits, the OS confirms the process is dead immediately, but the
# NVIDIA driver's teardown of that large a CUDA context can lag behind by a
# few seconds under memory pressure -- so launching the next camera's worker
# right away can hit a transient OOM even though the previous process is
# already gone. Poll actual free VRAM instead of assuming it's available.
MIN_FREE_GPU_MIB = 3000
GPU_WAIT_TIMEOUT_S = 60
GPU_POLL_INTERVAL_S = 2


def _free_gpu_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None  # no GPU / nvidia-smi unavailable -- don't block on it


def wait_for_free_gpu_memory(min_free_mib=MIN_FREE_GPU_MIB, timeout_s=GPU_WAIT_TIMEOUT_S):
    """Block until at least min_free_mib of GPU memory is free, or timeout."""
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
    """
    Return the main process's cached-but-unused CUDA memory to the driver.

    PyTorch's caching allocator holds on to freed blocks so it can reuse them
    without going back to the driver -- great for repeated allocations of the
    same size, bad right before we hand the GPU to a subprocess (HaMeR) that
    needs the memory to actually show up as free in nvidia-smi. gc.collect()
    first, since dead tensor references (e.g. from a caught exception's
    traceback frame) can otherwise keep blocks from being freed at all.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def detect_hand_tracks_hamer(frames_rgb, hand_side=None, vitpose_on_gpu=False, body_detector=None):
    """
    Run the HaMeR hand-pose worker over every frame of one camera's episode.
    Returns a (T, 9, 2) pixel-coordinate array in the same wrist/index/thumb
    layout MediaPipe's PointsClass.track_points_hand produces.

    hand_side: forwarded to hamer_hand_worker.py's --hand flag -- None
    auto-picks whichever side is more confident, or pass "left"/"right" to
    force it (see --hand_side on this script's own CLI).

    vitpose_on_gpu: forwarded to hamer_hand_worker.py's --vitpose_on_gpu
    flag. False (default) runs ViTPose+-Huge on CPU (the bottleneck, but
    safe on an 8GB card); True runs it on GPU (much faster, but needs the
    VRAM headroom -- see --vitpose_on_gpu on this script's own CLI).

    body_detector: forwarded to hamer_hand_worker.py's --body_detector flag
    -- None uses the worker's own default ("regnety", the lite/lighter
    detector); pass "vitdet" for the heavier, more accurate ViTDet-H
    detector (see --body_detector on this script's own CLI).
    """
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


def sample_gt_depth(depth_frame, points_xy, original_image_size, current_image_size, crop_ratios):
    """
    Look up per-point depth the same way PointsClass.get_points does: map
    each (x, y) from the current (possibly cropped/resized) image space back
    into the original depth map's pixel space.
    """
    crop_h, crop_w = crop_ratios
    w_orig, h_orig = original_image_size
    w_curr, h_curr = current_image_size

    depths = np.zeros(len(points_xy), dtype=np.float32)
    for i, (x, y) in enumerate(points_xy):
        h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
        point_h_orig = int((y / h_curr) * h_orig_cropped + h_orig * crop_h[0])
        w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
        point_w_orig = int((x / w_curr) * w_orig_cropped + w_orig * crop_w[0])
        depths[i] = depth_frame[point_h_orig, point_w_orig]
    return depths


# Create the parser
parser = argparse.ArgumentParser(
    description="Convert processed human data into a pkl file (HaMeR hand tracking)"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument("--calib_path", type=str,
                    help="Path to the calibration file")
parser.add_argument("--task_names", nargs="+",
                    type=str, help="List of task names")
parser.add_argument(
    "--num_demos", type=int, default=None, help="Number of demonstrations to process"
)
parser.add_argument("--process_points", action="store_true",
                    help="Process key points")
parser.add_argument(
    "--use_gt_depth", action="store_true", help="Use ground truth depth"
)
parser.add_argument(
    "--hand_side", type=str, default=None, choices=["left", "right"],
    help="Force HaMeR to track this hand instead of auto-picking whichever "
         "side ViTPose is more confident about. Useful when ViTPose "
         "confuses left/right on a close-up single-hand crop.",
)
parser.add_argument(
    "--vitpose_on_gpu", action="store_true",
    help="Run ViTPose+-Huge (hand bbox localization) on GPU instead of CPU. "
         "It's the per-frame compute bottleneck, so this is much faster, but "
         "it needs a few extra GB of VRAM -- only enable it if the GPU has "
         "the headroom (8GB is tight alongside HaMeR and this process's own "
         "CoTracker/DIFT models).",
)
parser.add_argument(
    "--body_detector", type=str, default=None, choices=["vitdet", "regnety"],
    help="Person detector backbone for HaMeR's hand-bbox localization step. "
         "Default (unset) is the worker's own default, 'regnety' -- far "
         "lighter on RAM/VRAM. Pass 'vitdet' for HaMeR's original, heavier "
         "and more accurate ViTDet-H detector (2.77GB checkpoint; needs more "
         "RAM/VRAM headroom).",
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
# "human_hand" is intentionally absent here -- PointsClass only ever sees
# "objects", so it never imports MediaPipe. Hand tracking is done separately
# via detect_hand_tracks_hamer() below and spliced into human_tracks_* in
# the same 9-points-then-objects layout MediaPipe produced.
object_labels = [
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


# Load p3po config for tracking.
#
# PointsClass is NOT constructed here. Its DIFT (SD 2.1) + TAPIR models load
# straight onto the GPU in __init__ (~1.3GB), and this script's whole point
# is to keep that memory OFF the GPU while the HaMeR worker subprocess is
# running (pass 1), so the worker gets the full VRAM budget -- important on
# an 8GB card, and essential if ViTPose is also on the GPU (--vitpose_on_gpu).
# So construction is deferred to the first pass-2 use via get_points_class().
_points_cfg = None
_points_class = None

if process_points:
    _cfg_path = Path(__file__).resolve(
    ).parents[2] / "cfgs" / "suite" / "points_cfg.yaml"
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

    _points_cfg = cfg


def get_points_class():
    """
    Build PointsClass on first use and cache it, so its GPU-resident DIFT/
    TAPIR models don't sit on the GPU during the first demo's HaMeR worker
    calls (pass 1). Once built it's reused for every subsequent demo.

    Note: this only keeps the GPU clear for the FIRST demo's pass 1 -- after
    that the models stay resident and are present during later demos' pass-1
    HaMeR calls too. That's fine with ViTPose on CPU (the default), but if
    you enable --vitpose_on_gpu for a multi-demo run and hit an OOM on the
    2nd+ demo, that residual ~1.3GB is why (single-demo runs are unaffected).
    """
    global _points_class
    if _points_class is None:
        _points_class = PointsClass(**_points_cfg)
    return _points_class


# 2D track smoothing applied before triangulation.
# Filtering in 2D before triangulation is important: a single-camera flicker
# in one frame produces a bad ray, which contaminates the entire triangulated 3D point.
TRACK_MEDIAN_WINDOW = 5    # frames: removes isolated spike detections
TRACK_SAVGOL_WINDOW = 11   # frames (odd): smooths continuous jitter
TRACK_SAVGOL_ORDER = 3


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
    match = re.match(
        r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    else:
        # Match plain numbers, including negatives and decimals
        match = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"Cannot extract number from '{s}'")


def check_annotations_exist(root_dir, task_name, pixel_keys, object_labels):
    """
    Raise immediately if any annotation file PointsClass will need is
    missing, before Phase A does any HaMeR work. Without this, a missing
    annotation would previously surface only after burning through every
    demo's HaMeR pass (Phase A now runs for the whole task before Phase B
    ever constructs PointsClass) -- same files run_pipeline.sh's own
    annotations_missing() bash check and PointsClass.__init__ both require.
    """
    missing = []
    coords_root = Path(root_dir) / "coordinates" / task_name
    for pk in pixel_keys:
        img_path = coords_root / "images" / f"{pk}.png"
        if not img_path.exists():
            missing.append(str(img_path))
        for label in object_labels:
            coords_path = coords_root / "coords" / f"{pk}_{label}.pkl"
            if not coords_path.exists():
                missing.append(str(coords_path))
    if missing:
        raise FileNotFoundError(
            "Missing annotation file(s) required for --process_points:\n  "
            + "\n  ".join(missing)
            + "\nRun label_points.py to create them first (see "
              "ur5e_pipeline/run_pipeline.sh's annotation workflow)."
        )


# ---------------------------------------------------------------------------
# Per-demo I/O helpers -- shared by the raw pass and both Phase A/B loops
# ---------------------------------------------------------------------------
def load_demo_frames(data_point):
    """Decode both cameras' videos for one demo. Returns {pixel_key: (T,H,W,3)
    BGR array} or None if the demo's videos/ directory is missing entirely."""
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
    """Whether every camera's depth pkl exists for one demo (existence check
    only -- Phase A never needs the actual depth values, only HaMeR-unrelated
    completeness gating; see load_demo_depth for the real load)."""
    depth_dir = data_point / "depth"
    if not depth_dir.exists():
        return False
    return all((depth_dir / f"depth{idx}.pkl").exists() for idx in camera_indices)


def load_demo_depth(data_point):
    """Load per-camera raw depth arrays (meters) for one demo."""
    depth_dir = data_point / "depth"
    depth_by_key = {}
    for idx in camera_indices:
        depth_file = depth_dir / f"depth{idx}.pkl"
        with open(depth_file, "rb") as f:
            depth = pkl.load(f)
        camera_name = f"cam_{idx}"
        pixel_key = camera2pixelkey[camera_name]
        depth_by_key[pixel_key] = depth
    return depth_by_key


def load_demo_state(data_point):
    """Parse states.csv for one demo -> (cartesian_states (T,6), gripper_states (T,))."""
    state_csv_path = data_point / "states.csv"
    state = read_csv(state_csv_path)

    cartesian_states = state["pose_aa"].values
    cartesian_states = np.array(
        [
            np.array([extract_number(x) for x in pose.strip("[]").split(",")])
            for pose in cartesian_states
        ],
        dtype=np.float32,
    )
    gripper_states = state["gripper_state"].values.astype(np.float32)
    return cartesian_states, gripper_states


def update_cartesian_gripper_bounds(bounds, cartesian_states, gripper_states):
    """Fold one demo's cartesian_states/gripper_states into the running
    (max_cartesian, min_cartesian, max_gripper, min_gripper) normalization
    bounds. Only ever called for demos that make it all the way to being
    written -- a demo that fails HaMeR or tracking must NOT contribute to
    these bounds, matching the original single-pass script's behavior
    (the update happened right before task_pkl_io.write_demo, after every
    earlier `continue` on failure)."""
    max_cartesian, min_cartesian, max_gripper, min_gripper = bounds
    if max_cartesian is None:
        max_cartesian = np.max(cartesian_states, axis=0)
        min_cartesian = np.min(cartesian_states, axis=0)
    else:
        max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
        min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))

    if max_gripper is None:
        max_gripper = np.max(gripper_states)
        min_gripper = np.min(gripper_states)
    else:
        max_gripper = np.maximum(max_gripper, np.max(gripper_states))
        min_gripper = np.minimum(min_gripper, np.min(gripper_states))
    return max_cartesian, min_cartesian, max_gripper, min_gripper


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

    if process_points:
        # Pre-flight: fail immediately if annotations are missing, before
        # Phase A burns any GPU time at all (see check_annotations_exist).
        check_annotations_exist(
            _points_cfg["root_dir"], _points_cfg["task_name"],
            _points_cfg["pixel_keys"], object_labels,
        )

        mark_every = 8

        # -----------------------------------------------------------------
        # Phase A: HaMeR for every demo in the task, before ANY object
        # tracking touches the GPU -- not just the first demo's. The
        # previous design only deferred PointsClass construction to the
        # first Phase-2 use, which kept the GPU clear for demo 0's HaMeR
        # call but not later demos' (PointsClass stays resident once
        # built), causing exactly the multi-demo OOM this restructure
        # fixes: finishing every demo's HaMeR pass first, then building
        # PointsClass once and running every demo's tracking pass second,
        # guarantees the two GPU consumers are never concurrent for the
        # whole run, not just demo 0.
        # -----------------------------------------------------------------
        phase_a_cache = {}  # demo_num -> {"hand_tracks", "cartesian_states", "gripper_states"}
        valid_demo_nums = []

        for i, data_point in enumerate(dirs):
            print(f"[Phase A: HaMeR] Processing data point {i+1}/{len(dirs)}")
            demo_num = int(str(data_point).split("_")[-1])

            if NUM_DEMOS is not None and demo_num >= NUM_DEMOS:
                print(f"Skipping data point {data_point}")
                continue

            frames_by_key = load_demo_frames(data_point)
            if frames_by_key is None:
                print(f"Data point {data_point} is incomplete")
                continue

            if use_gt_depth and not demo_depth_complete(data_point):
                print(f"Data point {data_point} is incomplete (no depth)")
                continue

            cartesian_states, gripper_states = load_demo_state(data_point)

            hand_tracks_by_key = {}
            save = True
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                # CV2 reads in BGR format, so we need to convert to RGB.
                frames_rgb = frames_by_key[pixel_key][..., ::-1]

                try:
                    hand_tracks_by_key[pixel_key] = detect_hand_tracks_hamer(
                        np.array(frames_rgb, dtype=np.uint8), hand_side=args.hand_side,
                        vitpose_on_gpu=args.vitpose_on_gpu, body_detector=args.body_detector,
                    )
                except Exception:
                    import traceback

                    print(
                        f"Error in HaMeR hand tracking for {pixel_key} -- skipping this demo:")
                    traceback.print_exc()
                    save = False
                    break

            if not save:
                continue

            phase_a_cache[demo_num] = {
                "hand_tracks": hand_tracks_by_key,
                "cartesian_states": cartesian_states,
                "gripper_states": gripper_states,
            }
            valid_demo_nums.append(demo_num)

        # Hand the GPU back cleanly before Phase B ever constructs
        # PointsClass -- this is now the ONLY point in the whole run where
        # HaMeR's GPU usage and PointsClass's GPU usage could ever be
        # adjacent in time, and they still never overlap.
        release_main_process_gpu_memory()

        # -----------------------------------------------------------------
        # Phase B: object tracking (DIFT/CoTracker via PointsClass) for
        # every demo that survived Phase A. PointsClass is constructed
        # exactly once here (get_points_class), so its GPU-resident models
        # are never present during any of Phase A's HaMeR subprocess calls.
        # -----------------------------------------------------------------
        points_class = get_points_class()
        for i, demo_num in enumerate(valid_demo_nums):
            print(f"[Phase B: tracking] Processing data point {i+1}/{len(valid_demo_nums)}")
            data_point = DATASET_PATH / f"demonstration_{demo_num}"
            cached = phase_a_cache[demo_num]
            hand_tracks_by_key = cached["hand_tracks"]
            cartesian_states = cached["cartesian_states"]
            gripper_states = cached["gripper_states"]

            # Re-decode frames (and depth) fresh here rather than holding
            # them from Phase A -- avoids holding every demo's
            # full-resolution frames in memory at once across both phases
            # (the same reason task_pkl_io stores one file per demo instead
            # of one big pickle; hand tracks are tiny and fine to cache,
            # frames are not).
            observation = {}
            frames_by_key = load_demo_frames(data_point)
            observation.update(frames_by_key)
            observation["cartesian_states"] = cartesian_states.astype(np.float32)
            observation["gripper_states"] = gripper_states.astype(np.float32)

            if use_gt_depth:
                depth_by_key = load_demo_depth(data_point)
                for pixel_key, depth in depth_by_key.items():
                    observation[f"depth_{pixel_key}"] = depth

            save = True
            for cam_idx in camera_indices:
                if save == False:
                    break
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]

                frames = observation[pixel_key][..., ::-1]
                hand_tracks_2d = hand_tracks_by_key[pixel_key]

                points_class.add_to_image_list(frames[0], pixel_key)
                for object_label in object_labels:
                    points_class.find_semantic_similar_points(
                        pixel_key, object_label)
                try:
                    points_class.track_points(
                        pixel_key, last_n_frames=mark_every, is_first_step=True
                    )
                except Exception:
                    import traceback

                    print(
                        f"Error in tracking object points for {pixel_key} -- skipping this demo:")
                    traceback.print_exc()
                    points_class.reset_episode()
                    save = False
                    continue

                points_class.track_points(
                    pixel_key, last_n_frames=mark_every, one_frame=(
                        mark_every == 1)
                )

                points_list = []
                points = points_class.get_points_on_image(pixel_key)
                points_list.append(points[0])

                if use_gt_depth:
                    points_3d_list = []
                    depth = observation[f"depth_{pixel_key}"][0]
                    points_class.set_depth(
                        depth,
                        pixel_key,
                        original_img_size,
                        save_img_size,
                        (crop_h, crop_w),
                    )
                    points_with_depth = points_class.get_points(pixel_key)
                    depths = points_with_depth[:, :, -1]

                    P = calibration_data[camera_name]["ext"]  # 4x4
                    K = calibration_data[camera_name]["int"]  # 3x3
                    points3d = pixel2d_to_3d_torch(points[0], depths[0], K, P)
                    points_3d_list.append(points3d)

                for idx, image in enumerate(frames[1:]):
                    print(f"Traj: {demo_num}, Frame: {idx}, Image: {pixel_key}")
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
                                points_class.add_to_image_list(
                                    image, pixel_key)
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
                            points_with_depth = points_class.get_points(
                                pixel_key, last_n_frames=mark_every
                            )
                            for j in range(mark_every - to_add):
                                depth = points_with_depth[j, :, -1]
                                points3d = pixel2d_to_3d_torch(
                                    points[j], depth, K, P)
                                points_3d_list.append(points3d)

                # points_list holds object-only 2D tracks (PointsClass never
                # saw "human_hand"), so splice the HaMeR hand track in front
                # to reproduce the same [hand(9), objects(N)] layout the
                # MediaPipe-based script produces.
                obj_tracks_2d = torch.stack(points_list).numpy()
                observation[f"human_tracks_{pixel_key}"] = np.concatenate(
                    [hand_tracks_2d, obj_tracks_2d], axis=1
                )
                if use_gt_depth:
                    obj_tracks_3d = torch.stack(points_3d_list).numpy()
                    hand_tracks_3d = np.stack(
                        [
                            pixel2d_to_3d_torch(
                                torch.tensor(
                                    hand_tracks_2d[t], dtype=torch.float32),
                                torch.tensor(
                                    sample_gt_depth(
                                        observation[f"depth_{pixel_key}"][t],
                                        hand_tracks_2d[t],
                                        original_img_size,
                                        save_img_size,
                                        (crop_h, crop_w),
                                    ),
                                    dtype=torch.float32,
                                ),
                                K,
                                P,
                            ).numpy()
                            for t in range(len(hand_tracks_2d))
                        ]
                    )
                    observation[f"human_tracks_3d_{pixel_key}"] = np.concatenate(
                        [hand_tracks_3d, obj_tracks_3d], axis=1
                    )
                points_class.reset_episode()

            if save == False:
                continue

            # Update max/min cartesian & gripper bounds -- only for demos
            # that make it all the way through both phases (matching the
            # original script: this update happened after every earlier
            # `continue` on failure, so a demo that failed HaMeR or
            # tracking never contributed to these bounds).
            max_cartesian, min_cartesian, max_gripper, min_gripper = update_cartesian_gripper_bounds(
                (max_cartesian, min_cartesian, max_gripper, min_gripper),
                cartesian_states, gripper_states,
            )

            if not use_gt_depth:
                """
                Triangulate 3D points from 2D points when gt_depth is not available
                """
                # Smooth 2D tracks per camera before triangulation so that a
                # single-camera flicker doesn't corrupt the triangulated 3D point.
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]
                    raw = observation[f"human_tracks_{pixel_key}"]  # (T, N, 2or3)
                    smoothed = smooth_2d_tracks(
                        raw[:, :, :2])       # filter x,y only
                    # preserve depth col if present
                    if raw.shape[2] > 2:
                        smoothed = np.concatenate(
                            [smoothed, raw[:, :, 2:]], axis=2)
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
                        observation[f"human_tracks_3d_{pixel_key}"].append(
                            pts3d[:, :3])
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]
                    observation[f"human_tracks_3d_{pixel_key}"] = np.array(
                        observation[f"human_tracks_3d_{pixel_key}"]
                    )

            task_pkl_io.write_demo(task_pkl_dir, demo_num, observation)

    else:
        # Raw pass (no tracking): unaffected by the HaMeR/PointsClass GPU
        # contention this two-phase restructure fixes, so it stays a single
        # loop, unchanged in behavior from before.
        for i, data_point in enumerate(dirs):
            print(f"Processing data point {i+1}/{len(dirs)}")
            demo_num = int(str(data_point).split("_")[-1])

            if NUM_DEMOS is not None and demo_num >= NUM_DEMOS:
                print(f"Skipping data point {data_point}")
                continue

            if demo_num in existing_demo_ids:
                print(f"Skipping data point {data_point} (already processed)")
                continue

            observation = {}
            frames_by_key = load_demo_frames(data_point)
            if frames_by_key is None:
                print(f"Data point {data_point} is incomplete")
                continue
            observation.update(frames_by_key)

            if use_gt_depth:
                if not demo_depth_complete(data_point):
                    print(f"Data point {data_point} is incomplete (no depth)")
                    continue
                depth_by_key = load_demo_depth(data_point)
                for pixel_key, depth in depth_by_key.items():
                    observation[f"depth_{pixel_key}"] = depth

            cartesian_states, gripper_states = load_demo_state(data_point)
            observation["cartesian_states"] = cartesian_states.astype(np.float32)
            observation["gripper_states"] = gripper_states.astype(np.float32)

            max_cartesian, min_cartesian, max_gripper, min_gripper = update_cartesian_gripper_bounds(
                (max_cartesian, min_cartesian, max_gripper, min_gripper),
                cartesian_states, gripper_states,
            )

            task_pkl_io.write_demo(task_pkl_dir, demo_num, observation)

    task_pkl_io.write_meta(task_pkl_dir, {
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
    })

print("Processing complete.")
