"""
Re-triangulate the 3D tracks of an already-processed task from its *saved* 2D
tracks and a (possibly updated) calibration file -- WITHOUT re-running the
expensive 2D keypoint extraction (HaMeR hand tracking + CoTracker/DIFT object
tracking).

Why this exists
---------------
Triangulation is not a separate stage in the main pipeline: it runs inline at
the end of `convert_to_pkl_human_hamer.py --process_points`, right after the
2D extraction (the `if not use_gt_depth:` block). There is no flag to run it
alone. But every demo pkl already persists the per-camera 2D tracks
(`human_tracks_{pixel_key}`, already smoothed) next to the 3D output, and the
triangulation math consumes only those 2D tracks + the calibration dict. So
after you update calib.npy you can regenerate just the 3D in place, skipping
the slow model passes.

This script reproduces the triangulation block of
convert_to_pkl_human_hamer.py:817-881 verbatim, with one deliberate omission:
it does NOT re-smooth the 2D tracks (the saved tracks are already smoothed --
re-smoothing would double-filter them).

Usage
-----
    cd .../Point-Policy
    python3 ur5e_pipeline/retriangulate.py \
        --data_dir /root/osx-ur/data \
        --calib_path calib/calib.npy \
        --task_names <TASK>

Then refresh the downstream steps (robot conversion, viz, lerobot export) from
the new 3D without re-extracting:
    SKIP_TRACKING=1 bash ur5e_pipeline/run_pipeline.sh <TASK>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

# --- import the same building blocks the main pipeline uses ---------------
_here = Path(__file__).resolve().parent
REPO_ROOT = _here.parent                       # .../Point-Policy
_point_policy = REPO_ROOT / "point_policy"
_franka = _point_policy / "robot_utils" / "franka"
# franka/utils.py must take priority over point_policy/utils.py -> insert last
sys.path.insert(0, str(_point_policy))         # for `point_utils`
sys.path.insert(0, str(_franka))               # for `utils` (franka's)

from utils import camera2pixelkey, triangulate_points  # noqa: E402
from point_utils import task_pkl_io                     # noqa: E402


# --- constants: must match convert_to_pkl_human_hamer.py exactly ----------
CAMERA_INDICES = [1, 2]
ORIGINAL_IMG_SIZE = (640, 480)
CROP_H, CROP_W = (0.0, 1.0), (0.0, 1.0)
SAVE_IMG_SIZE = (
    int(ORIGINAL_IMG_SIZE[0] * (CROP_W[1] - CROP_W[0])),
    int(ORIGINAL_IMG_SIZE[1] * (CROP_H[1] - CROP_H[0])),
)


def retriangulate_demo(obs, calibration_data):
    """Overwrite obs["human_tracks_3d_{pixel_key}"] in place from the saved 2D
    tracks + calibration. Returns True if the demo was re-triangulated, False
    if it was skipped (missing 2D tracks, e.g. a gt-depth demo).

    Reproduces convert_to_pkl_human_hamer.py:817-881 -- but NOT the preceding
    smoothing pass (lines 803-815): the saved 2D tracks are already smoothed.
    """
    # Guard: every camera's 2D tracks must be present to triangulate.
    for cam_idx in CAMERA_INDICES:
        pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
        if f"human_tracks_{pixel_key}" not in obs:
            return False

    for cam_idx in CAMERA_INDICES:
        pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
        obs[f"human_tracks_3d_{pixel_key}"] = []

    # number of frames (any camera has the same length)
    ref_key = camera2pixelkey[f"cam_{CAMERA_INDICES[0]}"]
    n_frames = len(obs[f"human_tracks_{ref_key}"])

    for t_idx in range(n_frames):
        P, pts = [], []
        for cam_idx in CAMERA_INDICES:
            camera_name = f"cam_{cam_idx}"
            pixel_key = camera2pixelkey[camera_name]

            extr = calibration_data[camera_name]["ext"]
            intr = calibration_data[camera_name]["int"]
            intr = np.concatenate([intr, np.zeros((3, 1))], axis=1)
            P.append(intr @ extr)

            pt2d = obs[f"human_tracks_{pixel_key}"][t_idx][:, :2]

            # crop-space -> original-image pixel coordinates
            point_h = pt2d[:, 1]
            h_orig, h_curr = ORIGINAL_IMG_SIZE[1], SAVE_IMG_SIZE[1]
            h_orig_cropped = h_orig * (CROP_H[1] - CROP_H[0])
            point_h_orig = (point_h / h_curr) * h_orig_cropped + h_orig * CROP_H[0]
            point_h_orig = point_h_orig.astype(np.int32)

            point_w = pt2d[:, 0]
            w_orig, w_curr = ORIGINAL_IMG_SIZE[0], SAVE_IMG_SIZE[0]
            w_orig_cropped = w_orig * (CROP_W[1] - CROP_W[0])
            point_w_orig = (point_w / w_curr) * w_orig_cropped + w_orig * CROP_W[0]
            point_w_orig = point_w_orig.astype(np.int32)

            pt2d = np.column_stack((point_w_orig, point_h_orig))

            # undistort before triangulation: P = K @ E assumes undistorted px
            K_orig = calibration_data[camera_name]["int"]
            D_orig = calibration_data[camera_name]["dist_coeff"]
            pt2d = cv2.undistortPoints(
                pt2d.reshape(-1, 1, 2).astype(np.float32),
                K_orig, D_orig, P=K_orig,
            ).reshape(-1, 2)

            pts.append(pt2d)

        pts3d = triangulate_points(P, pts)
        for cam_idx in CAMERA_INDICES:
            pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
            obs[f"human_tracks_3d_{pixel_key}"].append(pts3d[:, :3])

    for cam_idx in CAMERA_INDICES:
        pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
        obs[f"human_tracks_3d_{pixel_key}"] = np.array(
            obs[f"human_tracks_3d_{pixel_key}"]
        )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data root (contains processed_data_pkl/)")
    parser.add_argument("--calib_path", type=str, required=True,
                        help="Path to the (updated) calib.npy")
    parser.add_argument("--task_names", nargs="+", type=str, required=True,
                        help="Task name(s) under processed_data_pkl/")
    args = parser.parse_args()

    assert len(args.task_names) == 1, "Only one task name is supported for now"
    task_name = args.task_names[0]

    calibration_data = np.load(args.calib_path, allow_pickle=True).item()
    task_dir = Path(args.data_dir) / "processed_data_pkl" / task_name

    demo_ids = task_pkl_io.iter_demo_ids(task_dir)
    if not demo_ids:
        raise SystemExit(f"No demos found in {task_dir}")

    print(f"Re-triangulating {len(demo_ids)} demo(s) in {task_dir}")
    print(f"Using calibration: {args.calib_path}")

    n_done, n_skipped = 0, 0
    for demo_id in demo_ids:
        obs = task_pkl_io.read_demo(task_dir, demo_id)
        if retriangulate_demo(obs, calibration_data):
            task_pkl_io.write_demo(task_dir, demo_id, obs)
            n_done += 1
            print(f"  demo_{demo_id:04d}: re-triangulated")
        else:
            n_skipped += 1
            print(f"  demo_{demo_id:04d}: SKIPPED (no 2D tracks -- gt-depth demo?)")

    print(f"Done. Re-triangulated {n_done} demo(s), skipped {n_skipped}.")


if __name__ == "__main__":
    main()
