"""
One-time reference builder for the "object_keypoints" orientation method.

Reads one demo's frame-0 "gripper_orientation"-labeled object keypoints
(annotated via label_points.py --label gripper_orientation, tracked the same
way as the "objects" label) plus a trusted frame-0 gripper orientation
(seeded from one of the hand-based methods, or given explicitly), and saves
both as the reference that every other demo's object_keypoints orientation
will be Procrustes-aligned against.

Usage:
    cd point_policy/robot_utils/ur5e
    python build_orientation_reference.py \
        --data_dir /path/to/data \
        --task_name peg_hole_15 \
        --ref_demo_idx 0

Run this once per task, after Step 2 (tracking, with "gripper_orientation" in
--object_labels) and before selecting --orientation_method object_keypoints
in convert_pkl_human_to_robot.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))          # point_policy/ (point_utils)
sys.path.insert(0, str(_here.parent / "franka"))  # franka/utils.py
sys.path.insert(0, str(_here))                     # ur5e/ itself

from gripper_points import robot_base_orientation  # noqa: E402
from utils import compute_pinch_orientation as compute_pinch_orientation_fingertip  # noqa: E402
from plane_fit_orientation import compute_pinch_orientation as compute_pinch_orientation_plane_fit  # noqa: E402
from object_keypoint_orientation import (  # noqa: E402
    save_reference,
    warn_if_degenerate,
    load_orientation_inputs,
)

# point_policy/robot_utils/ur5e -> repo root, matching REPO_ROOT in run_pipeline.sh
REPO_ROOT = _here.parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT.parents[1] / "data"

NUM_HAND_POINTS = 9
ORIENTATION_LABEL = "gripper_orientation"

SEED_METHODS = {
    "fingertip": compute_pinch_orientation_fingertip,
    "plane_fit": compute_pinch_orientation_plane_fit,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the object-keypoint orientation reference for one task."
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_ROOT),
                        help=f"Root data directory (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name -- matches PKL_TASK_NAME (includes the "
                             "_gt_depth suffix if that's in use).")
    parser.add_argument("--pixel_key", type=str, default="pixels1",
                        help="Which camera's triangulated 3D track to read "
                             "(default: pixels1 -- both cameras' 3D tracks "
                             "are identical in the standard triangulated path).")
    parser.add_argument("--ref_demo_idx", type=int, default=0,
                        help="Which demo (by id) to build the reference from (default: 0).")
    parser.add_argument("--seed_method", type=str, default="plane_fit",
                        choices=list(SEED_METHODS.keys()),
                        help="Hand-based method used to seed the reference orientation, "
                             "when --ref_rotvec isn't given (default: plane_fit).")
    parser.add_argument("--ref_rotvec", type=float, nargs=3, default=None,
                        metavar=("RX", "RY", "RZ"),
                        help="Manually specify the reference gripper orientation as a "
                             "rotation vector, instead of seeding it from the hand.")
    parser.add_argument("--adjust_rotvec", type=float, nargs=3, default=None,
                        metavar=("RX", "RY", "RZ"),
                        help="Small correction rotation (radians, rotation-vector form) "
                             "applied on top of the orientation from --seed_method/"
                             "--ref_rotvec, in the gripper's OWN frame (right-multiplied: "
                             "ref_ori @ R.from_rotvec(adjust_rotvec)) -- e.g. [0,0,0.2] "
                             "rolls the gripper about its own reach axis without changing "
                             "the reach direction. Use this to iterate: run once with no "
                             "adjustment, inspect the printed rotvec (or visualize_3d.py's "
                             "output), then re-run with a small correction instead of "
                             "guessing the whole absolute rotation via --ref_rotvec.")
    return parser.parse_args()


def main():
    args = parse_args()

    hand_point, ref_pts = load_orientation_inputs(
        args.data_dir, args.task_name, args.pixel_key, args.ref_demo_idx,
        num_hand_points=NUM_HAND_POINTS, orientation_label=ORIENTATION_LABEL,
    )

    if args.ref_rotvec is not None:
        ref_ori = R.from_rotvec(args.ref_rotvec).as_matrix()
    else:
        seed_fn = SEED_METHODS[args.seed_method]
        ref_ori = seed_fn(hand_point) @ robot_base_orientation

    pre_adjust_rotvec = R.from_matrix(ref_ori).as_rotvec()
    if args.adjust_rotvec is not None:
        ref_ori = ref_ori @ R.from_rotvec(args.adjust_rotvec).as_matrix()

    degeneracy_ratio = warn_if_degenerate(ref_pts)

    coords_path = REPO_ROOT / "coordinates" / args.task_name / "orientation_reference.npz"
    save_reference(coords_path, ref_pts, ref_ori)

    print(f"Reference built from demo {args.ref_demo_idx}, {ref_pts.shape[0]} "
          f"'{ORIENTATION_LABEL}' keypoint(s).")
    print(f"  Degeneracy ratio (smallest/largest singular value): {degeneracy_ratio:.4f}")
    if args.adjust_rotvec is not None:
        print(f"  Orientation before --adjust_rotvec: {pre_adjust_rotvec}")
        print(f"  Applied adjustment (local frame):   {args.adjust_rotvec}")
    print(f"  Reference orientation (rotvec): {R.from_matrix(ref_ori).as_rotvec()}")
    print(f"  Saved to {coords_path}")


if __name__ == "__main__":
    main()
