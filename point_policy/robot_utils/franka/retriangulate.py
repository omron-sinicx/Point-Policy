"""
Re-triangulate an ALREADY-PROCESSED task pkl against a different calibration,
without re-running HaMeR or DIFT/TAPIR.

Use this after re-calibrating the cameras. The 2D tracks stored in the pkl
depend on the images, the intrinsics and the distortion coefficients -- none
of which a pure extrinsics re-calibration changes -- so the only thing that
needs redoing is the triangulation that turns them into human_tracks_3d_*.
Check that assumption before trusting the output: if `int` or `dist_coeff`
also moved, the 2D->original-pixel undistortion inside triangulate_tracks
changes too, and while this script still applies it correctly, any DIFT/TAPIR
track that was computed against the OLD intrinsics is a separate question --
re-run retrack_objects.py in that case.

Usage:
    cd point_policy/robot_utils/franka

    # try it on one demo first, writing to a separate task dir
    python retriangulate.py \
        --data_dir /root/osx-ur/data \
        --calib_path ../../../calib/calib_0903.npy \
        --task_name bottle_open_0903_07 \
        --demo_ids 0 --out_task_name bottle_open_0903_07_recalib

    # then commit to it in place
    python retriangulate.py ... --task_name bottle_open_0903_07

Re-run Step 3 (convert_pkl_human_to_robot.py) afterwards -- the robot pkl in
expert_demos/ is derived from human_tracks_3d_* and is stale the moment this
script rewrites them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))  # point_policy/ (point_utils, cfgs)
sys.path.insert(0, str(_here))             # franka/utils.py, object_tracking.py

from object_tracking import (  # noqa: E402
    CAMERA_INDICES,
    triangulate_tracks,
)
from utils import camera2pixelkey  # noqa: E402
from point_utils import task_pkl_io  # noqa: E402

REPO_ROOT = _here.parents[2]  # .../Point-Policy
DEFAULT_DATA_ROOT = REPO_ROOT.parents[1] / "data"

# triangulate_tracks() smooths the 2D tracks it is handed and writes the
# smoothed version back. Every track in an already-processed pkl was smoothed
# on the way in, so ANY smoothing here is a second pass -- median(5)+savgol
# applied twice measurably shifts the trajectory that Step 3's TCP position
# and rigid_transform_3D orientation propagation depend on. Its documented
# escape hatch is `smooth_from >= n_points -> smoothed = xy` (no-op), so a
# sentinel past any real point count disables smoothing outright.
NO_SMOOTHING = 1 << 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-triangulate an already-processed task pkl against a new "
                    "calibration, reusing its existing 2D tracks."
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_ROOT),
                        help=f"Root data directory (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--calib_path", type=str, required=True,
                        help="Path to the NEW calib .npy to triangulate against.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name, WITHOUT any _gt_depth suffix.")
    parser.add_argument("--out_task_name", type=str, default=None,
                        help="Write results to this task directory instead of "
                             "overwriting the input (default: in-place).")
    parser.add_argument("--demo_ids", nargs="+", type=int, default=None,
                        help="Only process these demo ids (default: all).")
    parser.add_argument("--use_gt_depth", action="store_true",
                        help="Input task used ground-truth depth. Rejected: that "
                             "path derives 3D from the depth image, not from "
                             "cross-camera triangulation, so there is nothing "
                             "here for a new calibration to change.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_gt_depth:
        raise SystemExit(
            "--use_gt_depth tasks do not triangulate: human_tracks_3d_* comes from "
            "the depth image via sample_gt_depth(), so re-triangulating is a no-op. "
            "Re-run Step 2 if the calibration those depths were unprojected with "
            "has changed."
        )

    data_pkl_root = Path(args.data_dir) / "processed_data_pkl"
    in_dir = data_pkl_root / args.task_name
    out_dir = data_pkl_root / args.out_task_name if args.out_task_name else in_dir

    if not in_dir.exists():
        raise FileNotFoundError(f"No processed task pkl at {in_dir}")

    demo_ids = task_pkl_io.iter_demo_ids(in_dir)
    if not demo_ids:
        raise RuntimeError(f"No demos found in {in_dir}")
    if args.demo_ids is not None:
        unknown = sorted(set(args.demo_ids) - set(demo_ids))
        if unknown:
            raise ValueError(f"demo id(s) {unknown} not present in {in_dir}")
        demo_ids = list(args.demo_ids)

    calibration_data = np.load(args.calib_path, allow_pickle=True).item()

    print(f"Re-triangulating {len(demo_ids)} demo(s) from {in_dir}")
    print(f"  calib  : {args.calib_path}")
    print(f"  output : {out_dir}" + ("  (in-place)" if out_dir == in_dir else ""))

    pixel_keys = [camera2pixelkey[f"cam_{i}"] for i in CAMERA_INDICES]

    n_ok, skipped = 0, []
    for i, demo_id in enumerate(demo_ids):
        observation = task_pkl_io.read_demo(in_dir, demo_id)

        # Step 2 skips (rather than aborts on) demos whose HaMeR or object
        # tracking raised, leaving only the raw pass's keys. There are no 2D
        # tracks to re-triangulate for those.
        missing = [pk for pk in pixel_keys if f"human_tracks_{pk}" not in observation]
        if missing:
            print(f"[{i + 1}/{len(demo_ids)}] demo {demo_id}: no {missing} -- "
                  f"never tracked, skipping")
            skipped.append(demo_id)
            continue

        before = np.asarray(observation[f"human_tracks_3d_{pixel_keys[0]}"]).copy()
        triangulate_tracks(observation, calibration_data, smooth_from=NO_SMOOTHING)
        after = np.asarray(observation[f"human_tracks_3d_{pixel_keys[0]}"])

        shift = np.linalg.norm(after - before, axis=-1)
        print(f"[{i + 1}/{len(demo_ids)}] demo {demo_id}: {before.shape[0]} frames, "
              f"3D shift median {np.median(shift) * 1000:.1f}mm "
              f"max {shift.max() * 1000:.1f}mm")

        task_pkl_io.write_demo(out_dir, demo_id, observation)
        n_ok += 1

    if out_dir != in_dir:
        try:
            task_pkl_io.write_meta(out_dir, task_pkl_io.read_meta(in_dir))
        except FileNotFoundError:
            print(f"  Warning: {in_dir}/meta.pkl not found -- output task will have "
                  f"no normalization bounds.")

    print(f"\nDone. {n_ok} demo(s) re-triangulated"
          + (f", {len(skipped)} skipped (untracked): {skipped}" if skipped else "")
          + f" -> {out_dir}")
    if n_ok:
        print("  Re-run Step 3 (convert_pkl_human_to_robot.py) -- the robot pkl in "
              "expert_demos/ is now stale.")


if __name__ == "__main__":
    main()
