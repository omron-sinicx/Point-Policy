"""
Re-run DIFT + TAPIR object tracking on an ALREADY-PROCESSED task pkl, without
re-running HaMeR.

Use this to add or change object-keypoint label groups (e.g. adding
"gripper_orientation" for convert_pkl_human_to_robot.py's object_keypoints
orientation method) on tasks that have already been through Step 2. HaMeR's
output does not depend on --object_labels at all, so re-detecting the hand
would be pure waste -- and on a 50-demo task it is by far the slowest part of
the pipeline.

This works because the hand track is only ever used for the final
concatenation: it is never passed to PointsClass, and never influences DIFT
correspondence or TAPIR queries. So the existing pkl's first 9 points are
lifted out as-is and re-spliced onto freshly-tracked object points.

Usage:
    cd point_policy/robot_utils/franka

    # add a second label group to an already-tracked task
    python retrack_objects.py \
        --data_dir /root/osx-ur/data \
        --calib_path /root/osx-ur/dependencies/Point-Policy/calib/calib.npy \
        --task_name peg_hole_dir1_01 \
        --object_labels objects gripper_orientation

    # try it on one demo first, writing to a separate task dir
    python retrack_objects.py ... --demo_ids 0 --out_task_name peg_hole_test

Prerequisite: every label in --object_labels must already be annotated, i.e.
coordinates/<task>/coords/<pixel_key>_<label>.pkl exists. Annotate a new one
with:
    python label_points.py --task_name <task> --label gripper_orientation
(label_points.py reads frame 0 straight from the task pkl, so no video needed.)
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))  # point_policy/ (point_utils, cfgs)
sys.path.insert(0, str(_here))             # franka/utils.py, object_tracking.py

from object_tracking import (  # noqa: E402
    CAMERA_INDICES,
    NUM_HAND_POINTS,
    build_points_cfg,
    check_annotations_exist,
    split_hand_and_objects,
    track_objects_for_camera,
    triangulate_tracks,
)
from utils import camera2pixelkey  # noqa: E402
from point_utils import task_pkl_io  # noqa: E402
from point_utils.points_class import PointsClass  # noqa: E402

REPO_ROOT = _here.parents[2]  # .../Point-Policy
DEFAULT_DATA_ROOT = REPO_ROOT.parents[1] / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-run DIFT+TAPIR object tracking on an already-processed "
                    "task pkl, reusing its existing HaMeR hand tracks."
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_ROOT),
                        help=f"Root data directory (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--calib_path", type=str, required=True,
                        help="Path to calib.npy (needed for triangulation).")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name, WITHOUT any _gt_depth suffix (that's added "
                             "by --use_gt_depth).")
    parser.add_argument("--object_labels", nargs="+", type=str, default=["objects"],
                        help="Label group(s) to track, in concatenation order "
                             "(default: objects). Each needs its own "
                             "coordinates/<task>/coords/<pixel_key>_<label>.pkl.")
    parser.add_argument("--out_task_name", type=str, default=None,
                        help="Write results to this task directory instead of "
                             "overwriting the input (default: in-place).")
    parser.add_argument("--demo_ids", nargs="+", type=int, default=None,
                        help="Only process these demo ids (default: all). Useful to "
                             "validate on one demo before committing to a 50-demo task.")
    parser.add_argument("--use_gt_depth", action="store_true",
                        help="Input task used ground-truth depth (reads/writes the "
                             "_gt_depth-suffixed pkl directory).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_gt_depth:
        # Note the asymmetry: the pkl dir carries the suffix, but annotations
        # live under the plain task name.
        pkl_task_name = f"{args.task_name}_gt_depth"
    else:
        pkl_task_name = args.task_name

    data_pkl_root = Path(args.data_dir) / "processed_data_pkl"
    in_dir = data_pkl_root / pkl_task_name
    out_dir = data_pkl_root / args.out_task_name if args.out_task_name else in_dir

    if not in_dir.exists():
        raise FileNotFoundError(f"No processed task pkl at {in_dir}")

    pixel_keys = [camera2pixelkey[f"cam_{i}"] for i in CAMERA_INDICES]
    check_annotations_exist(str(REPO_ROOT), args.task_name, pixel_keys,
                            args.object_labels)

    demo_ids = task_pkl_io.iter_demo_ids(in_dir)
    if not demo_ids:
        raise RuntimeError(f"No demos found in {in_dir}")
    if args.demo_ids is not None:
        unknown = sorted(set(args.demo_ids) - set(demo_ids))
        if unknown:
            raise ValueError(f"demo id(s) {unknown} not present in {in_dir}")
        demo_ids = list(args.demo_ids)

    calibration_data = np.load(args.calib_path, allow_pickle=True).item()

    print(f"Re-tracking {len(demo_ids)} demo(s) from {in_dir}")
    print(f"  labels : {args.object_labels}")
    print(f"  output : {out_dir}" + ("  (in-place)" if out_dir == in_dir else ""))

    cfg = build_points_cfg(args.task_name, pixel_keys, args.object_labels)
    points_class = PointsClass(**cfg)
    n_obj_points = points_class.num_points
    print(f"  tracking {n_obj_points} object point(s) per camera")

    label_sizes = [
        points_class.initial_coords[f"{pixel_keys[0]}_{label}"].shape[0]
        for label in args.object_labels
    ]

    n_ok, n_failed = 0, 0
    for i, demo_id in enumerate(demo_ids):
        print(f"\n[{i + 1}/{len(demo_ids)}] demo {demo_id}")
        observation = task_pkl_io.read_demo(in_dir, demo_id)

        ok = True
        hand_tracks = {}
        for cam_idx in CAMERA_INDICES:
            pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
            existing = observation.get(f"human_tracks_{pixel_key}")
            if existing is None:
                raise RuntimeError(
                    f"demo {demo_id} has no human_tracks_{pixel_key} -- it was never "
                    f"tracked, so there is no HaMeR hand track to reuse. Run Step 2 "
                    f"(convert_to_pkl_human_hamer.py --process_points) on it first."
                )
            hand_2d, old_obj_2d = split_hand_and_objects(existing[:, :, :2])
            hand_tracks[pixel_key] = hand_2d
            if i == 0 and cam_idx == CAMERA_INDICES[0]:
                print(f"  reusing {hand_2d.shape[1]} hand points, replacing "
                      f"{old_obj_2d.shape[1]} object point(s) with {n_obj_points}")

        for cam_idx in CAMERA_INDICES:
            pixel_key = camera2pixelkey[f"cam_{cam_idx}"]
            frames_bgr = observation[pixel_key]
            try:
                obj_2d = track_objects_for_camera(
                    points_class, frames_bgr, pixel_key, args.object_labels,
                    demo_label=f"demo {demo_id}",
                )
            except Exception:
                print(f"  Error tracking {pixel_key} -- skipping demo {demo_id}:")
                traceback.print_exc()
                ok = False
                points_class.reset_episode()
                break
            observation[f"human_tracks_{pixel_key}"] = np.concatenate(
                [hand_tracks[pixel_key], obj_2d], axis=1
            )
            # reset_episode() clears every pixel_key at once, so cameras must be
            # processed strictly sequentially -- resetting mid-way through the
            # other camera would destroy its in-flight TAPIR causal state.
            points_class.reset_episode()

        if not ok:
            n_failed += 1
            continue

        # Hand points were already smoothed when this pkl was first written
        # (the original smoothing pass covered all points, not just objects),
        # so re-smoothing them would apply median+savgol twice and shift the
        # trajectory Step 3's orientation propagation depends on. Smooth only
        # the freshly-tracked object columns.
        triangulate_tracks(observation, calibration_data,
                           smooth_from=NUM_HAND_POINTS)
        task_pkl_io.write_demo(out_dir, demo_id, observation)
        n_ok += 1

    # Carry the input task's meta over first when writing somewhere new --
    # otherwise the output task would have no normalization bounds at all
    # (update_meta would find nothing to merge into, and the bounds are still
    # valid here since cartesian/gripper states are untouched). In-place runs
    # already have them, and update_meta preserves whatever is there.
    if out_dir != in_dir:
        try:
            task_pkl_io.write_meta(out_dir, task_pkl_io.read_meta(in_dir))
        except FileNotFoundError:
            print(f"  Warning: {in_dir}/meta.pkl not found -- output task will have "
                  f"no normalization bounds.")
    task_pkl_io.update_meta(out_dir, {
        "object_labels": list(args.object_labels),
        "object_label_sizes": label_sizes,
    })

    print(f"\nDone. {n_ok} demo(s) re-tracked"
          + (f", {n_failed} failed" if n_failed else "")
          + f" -> {out_dir}")
    print(f"  object_labels={args.object_labels} object_label_sizes={label_sizes}")


if __name__ == "__main__":
    main()
