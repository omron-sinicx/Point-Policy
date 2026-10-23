"""
DIFT + TAPIR object-keypoint tracking, as an importable module.

This is the logic that lives inline as "Phase B" of
convert_to_pkl_human_hamer.py (lines 632-890), lifted out so it can run
without HaMeR -- in particular, on already-processed task pkls, which is what
retrack_objects.py does.

Unlike the convert_to_pkl_* scripts (straight-line module code that runs
argparse and a whole pipeline at import time, and therefore cannot be imported
from), this module has no side effects on import.

How the two models divide the work:
  - DIFT runs on FRAME 0 ONLY, once per (demo, camera, label). It transfers the
    keypoints you annotated on coordinates/<task>/images/<pixel_key>.png onto
    this demo's first frame -- solving "the object starts somewhere slightly
    different in every demo".
  - TAPIR is seeded with those frame-0 points and tracks them through every
    remaining frame.
The human hand is not involved at all: PointsClass.find_semantic_similar_points
early-returns on "human_hand", and the hand track comes from HaMeR separately.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))  # point_policy/ (point_utils, cfgs)
sys.path.insert(0, str(_here))             # franka/utils.py

from utils import camera2pixelkey, triangulate_points  # noqa: E402

# HaMeR always emits exactly 9 hand points (wrist, index MCP->TIP, thumb
# CMC->TIP). This is a fixed property of the detector's output layout, NOT
# something to derive from a track array's width -- see split_hand_and_objects.
NUM_HAND_POINTS = 9

# Frame geometry. Must match what the pkl was originally built with, since the
# triangulation below maps points back into original-image coordinates using
# these. Same values as convert_to_pkl_human_hamer.py.
CAMERA_INDICES = [1, 2]
ORIGINAL_IMG_SIZE = (640, 480)
CROP_H, CROP_W = (0.0, 1.0), (0.0, 1.0)
SAVE_IMG_SIZE = (
    int(ORIGINAL_IMG_SIZE[0] * (CROP_W[1] - CROP_W[0])),
    int(ORIGINAL_IMG_SIZE[1] * (CROP_H[1] - CROP_H[0])),
)

# 2D track smoothing applied before triangulation: a single-camera flicker in
# one frame produces a bad ray, which contaminates the triangulated 3D point.
TRACK_MEDIAN_WINDOW = 5
TRACK_SAVGOL_WINDOW = 11
TRACK_SAVGOL_ORDER = 3

MARK_EVERY = 8


def smooth_2d_tracks(tracks):
    """Median + Savitzky-Golay on a (T, N, 2) 2D track array."""
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


def split_hand_and_objects(tracks, num_hand_points=NUM_HAND_POINTS):
    """Split a (T, 9+N, C) combined track array into (hand, objects).

    Always slices at the fixed hand-point count. Deriving the split from the
    *current* object count instead would be wrong exactly when the label set
    changes -- which is the whole point of re-tracking: an 11-point pkl (9
    hand + 2 objects) re-tracked to 8 object points would infer 11-8=3 hand
    points and silently corrupt everything downstream.
    """
    if tracks.shape[-2] < num_hand_points:
        raise ValueError(
            f"track array has only {tracks.shape[-2]} points, fewer than the "
            f"{num_hand_points} hand points it must start with"
        )
    return tracks[:, :num_hand_points], tracks[:, num_hand_points:]


def check_annotations_exist(root_dir, task_name, pixel_keys, object_labels):
    """Raise before any GPU work if an annotation PointsClass needs is missing."""
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
            "Missing annotation file(s):\n  " + "\n  ".join(missing)
            + f"\nCreate them with:  python label_points.py --task_name {task_name} "
              f"--label <label>"
        )


def build_points_cfg(task_name, pixel_keys, object_labels, use_gt_depth=False,
                     cfg_path=None):
    """Build the PointsClass kwargs dict from cfgs/suite/points_cfg.yaml.

    task_name must be the UN-suffixed task name: annotations live under
    coordinates/<task>/ without the _gt_depth suffix, even when the pkl
    directory carries it.

    use_gt_depth is forced to False for PointsClass regardless of the caller's
    own gt-depth mode. The key is absent from the YAML and PointsClass defaults
    it to True, which constructs Depth("/path/to/Depth-Anything-V2/", device)
    -- a placeholder path that crashes. That model is only used by get_depth(),
    which this tracking path never calls (it uses set_depth() with ground-truth
    depth instead), so disabling it is safe in both modes.
    """
    if cfg_path is None:
        cfg_path = _here.parents[1] / "cfgs" / "suite" / "points_cfg.yaml"
    with open(cfg_path) as stream:
        cfg = yaml.safe_load(stream)

    root_dir = cfg["root_dir"]
    cfg["dift_path"] = f"{root_dir}/{cfg['dift_path']}"
    cfg["cotracker_checkpoint"] = f"{root_dir}/{cfg['cotracker_checkpoint']}"
    cfg["tapir_checkpoint"] = f"{root_dir}/{cfg['tapir_checkpoint']}"
    cfg["task_name"] = task_name
    cfg["pixel_keys"] = list(pixel_keys)
    # Copy: PointsClass.__init__ does self.object_labels.remove("human_hand")
    # in place, which would mutate the caller's list.
    cfg["object_labels"] = list(object_labels)
    cfg["use_gt_depth"] = False  # see docstring -- deliberately not `use_gt_depth`
    return cfg


def track_objects_for_camera(points_class, frames_bgr, pixel_key, object_labels,
                             mark_every=MARK_EVERY, demo_label="", verbose=True):
    """Run DIFT (frame 0) + TAPIR (all frames) for one camera of one demo.

    frames_bgr: (T, H, W, 3) uint8 BGR, exactly as stored in the task pkl --
    already cropped/resized, so do NOT re-crop or re-resize.

    Returns (T, N_obj, 2) float32 object tracks in that frame's pixel space.
    Caller is responsible for points_class.reset_episode() afterwards.
    """
    # BGR -> RGB. This is a negative-stride view, which is fine here because
    # add_to_image_list does .astype() (a copy) internally -- but it would
    # raise if handed straight to torch.from_numpy.
    frames = frames_bgr[..., ::-1]

    points_class.add_to_image_list(frames[0], pixel_key)
    for object_label in object_labels:
        points_class.find_semantic_similar_points(pixel_key, object_label)  # <- DIFT
    points_class.track_points(pixel_key, last_n_frames=mark_every, is_first_step=True)
    # Second call is required: it converts self.tracks from the raw (N,3)
    # semantic tensor into the [1,T,N,2] layout get_points_on_image indexes.
    points_class.track_points(pixel_key, last_n_frames=mark_every,
                              one_frame=(mark_every == 1))

    points_list = []
    points = points_class.get_points_on_image(pixel_key)
    points_list.append(points[0])

    # In a terminal, overwrite one status line; when redirected to a file,
    # \r doesn't overwrite anything, so throttle to occasional real lines
    # instead of one per frame.
    is_tty = sys.stdout.isatty()
    n_frames = len(frames) - 1

    for idx, image in enumerate(frames[1:]):
        if verbose and (is_tty or (idx + 1) % 50 == 0 or idx == n_frames - 1):
            print(f"  {demo_label} frame {idx + 1}/{n_frames} [{pixel_key}]",
                  end="\r" if is_tty else "\n", flush=True)
        points_class.add_to_image_list(image, pixel_key)

        if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
            # Pad the final partial chunk out to mark_every by re-adding the
            # last real frame; those duplicates are dropped from points_list
            # below, so len(points_list) == len(frames) exactly.
            to_add = mark_every - (idx + 1) % mark_every
            if to_add < mark_every:
                for _ in range(to_add):
                    points_class.add_to_image_list(image, pixel_key)
            else:
                to_add = 0

            points_class.track_points(pixel_key, last_n_frames=mark_every,
                                      one_frame=(mark_every == 1))
            points = points_class.get_points_on_image(pixel_key,
                                                      last_n_frames=mark_every)
            for j in range(mark_every - to_add):
                points_list.append(points[j])

    if verbose and is_tty:
        print()
    obj_tracks_2d = torch.stack(points_list).numpy()
    if len(obj_tracks_2d) != len(frames):
        raise RuntimeError(
            f"tracked {len(obj_tracks_2d)} frames but the demo has {len(frames)} "
            f"-- object/hand concatenation would be misaligned"
        )
    return obj_tracks_2d


def triangulate_tracks(observation, calibration_data, camera_indices=CAMERA_INDICES,
                       smooth_from=None):
    """Triangulate human_tracks_{pixel_key} across cameras into 3D, in place.

    Writes human_tracks_3d_{pixel_key} for every camera (the same shared
    world-frame point cloud stored under each key, matching how the pipeline
    has always done it).

    smooth_from: if given, only points from this index onward are smoothed.
    Use it to avoid double-smoothing hand points that were already smoothed
    when the pkl was first written -- re-applying median(5)+savgol(11,3) would
    subtly alter the trajectory that Step 3's rigid_transform_3D orientation
    propagation and TCP position depend on. Pass None to smooth everything
    (the original behavior, correct for freshly-detected tracks).
    """
    for cam_idx in camera_indices:
        pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
        raw = observation[f"human_tracks_{pixel_key}"]  # (T, N, 2 or 3)
        xy = raw[:, :, :2]
        if smooth_from is None:
            smoothed = smooth_2d_tracks(xy)
        elif smooth_from >= xy.shape[1]:
            smoothed = xy  # nothing new to smooth
        else:
            smoothed = xy.copy()
            smoothed[:, smooth_from:] = smooth_2d_tracks(xy[:, smooth_from:])
        if raw.shape[2] > 2:  # preserve depth column if present
            smoothed = np.concatenate([smoothed, raw[:, :, 2:]], axis=2)
        observation[f"human_tracks_{pixel_key}"] = smoothed

    for cam_idx in camera_indices:
        pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
        observation[f"human_tracks_3d_{pixel_key}"] = []

    n_frames = len(observation[f"human_tracks_{camera2pixelkey[f'cam_{camera_indices[0]}']}"])
    for t_idx in range(n_frames):
        P, pts = [], []
        for cam_idx in camera_indices:
            camera_name = f"cam_{cam_idx}"
            pixel_key = camera2pixelkey[camera_name]

            extr = calibration_data[camera_name]["ext"]
            intr = calibration_data[camera_name]["int"]
            intr_h = np.concatenate([intr, np.zeros((3, 1))], axis=1)
            P.append(intr_h @ extr)

            pt2d = observation[f"human_tracks_{pixel_key}"][t_idx]

            # Map back into original-image pixel space (the calibration's space).
            point_h = pt2d[:, 1]
            h_orig, h_curr = ORIGINAL_IMG_SIZE[1], SAVE_IMG_SIZE[1]
            h_orig_cropped = h_orig * (CROP_H[1] - CROP_H[0])
            point_h_orig = ((point_h / h_curr) * h_orig_cropped
                            + h_orig * CROP_H[0]).astype(np.int32)

            point_w = pt2d[:, 0]
            w_orig, w_curr = ORIGINAL_IMG_SIZE[0], SAVE_IMG_SIZE[0]
            w_orig_cropped = w_orig * (CROP_W[1] - CROP_W[0])
            point_w_orig = ((point_w / w_curr) * w_orig_cropped
                            + w_orig * CROP_W[0]).astype(np.int32)

            pt2d = np.column_stack((point_w_orig, point_h_orig))

            # Undistort before triangulation: P = K @ E assumes undistorted
            # pixels, so passing raw distorted ones adds error proportional to
            # lens distortion (worst near the image edges).
            K_orig = calibration_data[camera_name]["int"]
            D_orig = calibration_data[camera_name]["dist_coeff"]
            pt2d = cv2.undistortPoints(
                pt2d.reshape(-1, 1, 2).astype(np.float32),
                K_orig, D_orig, P=K_orig,
            ).reshape(-1, 2)

            pts.append(pt2d)

        pts3d = triangulate_points(P, pts)
        for cam_idx in camera_indices:
            pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
            observation[f"human_tracks_3d_{pixel_key}"].append(pts3d[:, :3])

    for cam_idx in camera_indices:
        pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
        observation[f"human_tracks_3d_{pixel_key}"] = np.array(
            observation[f"human_tracks_3d_{pixel_key}"]
        )
    return observation
