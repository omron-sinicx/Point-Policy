"""
Interactive 2D overlay tool for tuning the object_keypoints orientation
reference. Projects the UR5e gripper's 9-point geometry onto the actual
camera frame -- the same look as visualize_tracks.py's --mode robot output
(e.g. robot_action.mp4) -- and updates it live as you drag Roll/Pitch/Yaw
sliders. Click "Save Reference" to write
coordinates/{task_name}/orientation_reference.npz directly, with no need to
guess --adjust_rotvec values via build_orientation_reference.py's CLI and
re-run.

Both camera views are shown side by side by default, each projecting the SAME
3D pose through its own calibration. That makes them a real cross-check: a
single view can't constrain depth along its own optical axis, so an
orientation can look perfectly aligned in one camera while being visibly
wrong in the other.

Usage:
    cd point_policy/robot_utils/ur5e
    python interactive_orientation_adjust.py \
        --data_dir /path/to/data --calib_path /path/to/calib/calib.npy \
        --task_name peg_hole_15 --ref_demo_idx 0

    # single view only
    python interactive_orientation_adjust.py ... --cameras cam_1

Requires a display (matplotlib TkAgg -- same GUI stack label_points.py uses).

Run this once per task, after Step 2 (tracking, with "gripper_orientation" in
--object_labels) and before selecting --orientation_method object_keypoints
in convert_pkl_human_to_robot.py.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Slider, Button  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))                     # point_policy/ (point_utils)
sys.path.insert(0, str(_here.parent / "franka"))              # franka/utils.py
sys.path.insert(0, str(_here.parents[2] / "ur5e_pipeline"))  # visualize_tracks.py draw helpers
sys.path.insert(0, str(_here))                                 # ur5e/ itself

from gripper_points import extrapoints, Tshift, robot_base_orientation  # noqa: E402
from utils import compute_pinch_orientation as compute_pinch_orientation_fingertip  # noqa: E402
from plane_fit_orientation import compute_pinch_orientation as compute_pinch_orientation_plane_fit  # noqa: E402
from object_keypoint_orientation import (  # noqa: E402
    save_reference,
    load_reference,
    warn_if_degenerate,
    load_orientation_inputs,
    compute_orientation,
)
from orientation_overrides import save_override, load_overrides  # noqa: E402
from visualize_tracks import draw_points, draw_object_points  # noqa: E402
from point_utils import task_pkl_io  # noqa: E402

# point_policy/robot_utils/ur5e -> repo root, matching REPO_ROOT in run_pipeline.sh
REPO_ROOT = _here.parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT.parents[1] / "data"

NUM_HAND_POINTS = 9
ORIENTATION_LABEL = "gripper_orientation"
_CAM2KEY = {"cam_1": "pixels1", "cam_2": "pixels2"}

SEED_METHODS = {
    "fingertip": compute_pinch_orientation_fingertip,
    "plane_fit": compute_pinch_orientation_plane_fit,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively tune the object_keypoints orientation reference, "
                    "with a live overlay on the camera frame."
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_ROOT),
                        help=f"Root data directory (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--calib_path", type=str, required=True,
                        help="Path to calib.npy.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name -- matches PKL_TASK_NAME (includes the "
                             "_gt_depth suffix if that's in use).")
    parser.add_argument("--cameras", nargs="+", type=str, default=["cam_1", "cam_2"],
                        choices=["cam_1", "cam_2"],
                        help="Which camera view(s) to overlay on, side by side "
                             "(default: both). Both is recommended -- an orientation "
                             "can look correct in one view and be visibly wrong in "
                             "the other, since a single view can't show depth along "
                             "its own optical axis.")
    parser.add_argument("--ref_demo_idx", type=int, default=0,
                        help="Which demo (by id) to build the reference from (default: 0).")
    parser.add_argument("--seed_method", type=str, default="plane_fit",
                        choices=list(SEED_METHODS.keys()),
                        help="Hand-based method used to seed the starting orientation, "
                             "when --ref_rotvec isn't given (default: plane_fit).")
    parser.add_argument("--ref_rotvec", type=float, nargs=3, default=None,
                        metavar=("RX", "RY", "RZ"),
                        help="Manually specify the starting gripper orientation as a "
                             "rotation vector, instead of seeding it from the hand.")
    parser.add_argument("--demo_idx", type=int, default=None,
                        help="PER-DEMO MODE. Tune the frame-0 orientation of THIS demo "
                             "alone and save it to coordinates/<task>/"
                             "orientation_overrides.json, leaving the task-wide "
                             "reference and every other demo untouched. The sliders "
                             "start from the orientation Step 3 currently derives for "
                             "this demo, so they read as a correction to exactly what "
                             "you see in its visualization. Use this for the handful of "
                             "demos whose keypoints were mislocalized; use the default "
                             "task-wide mode when the whole task is off.")
    parser.add_argument("--from_existing", action="store_true",
                        help="Start from the orientation already saved in "
                             "coordinates/<task>/orientation_reference.npz instead of "
                             "re-seeding from the hand -- i.e. adjust the reference you "
                             "already have, rather than starting over. The sliders then "
                             "read as a correction on top of it.")
    return parser.parse_args()


def gripper_points_3d(ori, anchor_pos):
    """9-point UR5e gripper geometry (TCP center + 8 extrapoints) at the
    given orientation, anchored at anchor_pos -- same construction as
    convert_pkl_human_to_robot.py's per-frame robot_points_list."""
    T_g_world = np.eye(4)
    T_g_world[:3, :3] = ori
    T_g_world[:3, 3] = anchor_pos
    T_g_world = T_g_world @ Tshift
    points3d = [T_g_world[:3, 3]]
    for Tp in extrapoints:
        points3d.append((T_g_world @ Tp)[:3, 3])
    return np.array(points3d)  # (9, 3)


def project(pts3d, K, D, r_vec, t_vec):
    pts2d = cv2.projectPoints(pts3d.astype(np.float32), r_vec, t_vec, K, D)[0].squeeze()
    if pts2d.ndim == 1:
        pts2d = pts2d[np.newaxis]
    return pts2d


def build_views(args, calibration_data, observation, ref_pts):
    """One render context per requested camera: its frame 0, projection
    parameters, and the reference keypoints already projected (they're fixed,
    so project them once rather than per slider move)."""
    views = []
    for camera_name in args.cameras:
        pixel_key = _CAM2KEY[camera_name]
        cam_calib = calibration_data[camera_name]
        K, D, P = cam_calib["int"], cam_calib["dist_coeff"], cam_calib["ext"]
        r_vec, _ = cv2.Rodrigues(P[:3, :3])
        t_vec = P[:3, 3]
        views.append({
            "camera_name": camera_name,
            "frame_bgr": observation[pixel_key][0],  # (H, W, 3) BGR, frame 0
            "K": K, "D": D, "r_vec": r_vec, "t_vec": t_vec,
            "ref_pts_2d": project(ref_pts, K, D, r_vec, t_vec),
        })
    return views


def main():
    args = parse_args()

    # The 3D tracks are identical across cameras in the standard triangulated
    # path (one shared world-frame point cloud stored under every pixel key),
    # so the keypoints/hand read here don't depend on which camera we render.
    per_demo = args.demo_idx is not None
    # In per-demo mode everything is read from THAT demo -- its frame-0
    # orientation keypoints, its frame, its currently-derived pose -- so the
    # overlay and the sliders describe the demo being corrected, not demo 0.
    load_demo_idx = args.demo_idx if per_demo else args.ref_demo_idx

    hand_point, ref_pts = load_orientation_inputs(
        args.data_dir, args.task_name, _CAM2KEY[args.cameras[0]], load_demo_idx,
        num_hand_points=NUM_HAND_POINTS, orientation_label=ORIENTATION_LABEL,
    )
    anchor_pos = ref_pts.mean(axis=0)

    ref_path = REPO_ROOT / "coordinates" / args.task_name / "orientation_reference.npz"
    if per_demo:
        # Start from what Step 3 produces for this demo TODAY, so the sliders are
        # a correction to the pose visible in robot_action_demo<N>.mp4 rather than
        # to some freshly seeded guess the user has never seen.
        saved_ov = load_overrides(args.task_name).get(args.demo_idx, {})
        # Honour a per-demo method override, so the sliders sit on top of the
        # estimator this demo actually uses rather than the task default.
        demo_method = saved_ov.get("method") or "object_keypoints"
        if demo_method == "object_keypoints":
            task_ref_pts, task_ref_ori = load_reference(ref_path)
            base_ori = compute_orientation(ref_pts, task_ref_pts, task_ref_ori)
        else:
            base_ori = SEED_METHODS[demo_method](hand_point) @ robot_base_orientation
            print(f"Demo {args.demo_idx} uses method={demo_method} (per-demo override).")
        existing = saved_ov.get("rotvec")
        if existing is not None:
            # Re-tuning a demo: fold its saved correction in, so the sliders start
            # where the last session left off instead of silently discarding it.
            base_ori = base_ori @ R.from_rotvec(existing).as_matrix()
            print(f"Demo {args.demo_idx} already has an override "
                  f"{np.round(existing, 5)} -- starting from it.")
        print(f"PER-DEMO mode: tuning demo {args.demo_idx} only; the task-wide "
              f"reference at {ref_path} will not be modified.")
    elif args.from_existing:
        # load_reference raises with an actionable message if it's missing.
        _, base_ori = load_reference(ref_path)
        print(f"Starting from saved reference: {ref_path}")
    elif args.ref_rotvec is not None:
        base_ori = R.from_rotvec(args.ref_rotvec).as_matrix()
    else:
        base_ori = SEED_METHODS[args.seed_method](hand_point) @ robot_base_orientation

    warn_if_degenerate(ref_pts)

    calibration_data = np.load(args.calib_path, allow_pickle=True).item()
    human_pkl_dir = Path(args.data_dir) / "processed_data_pkl" / args.task_name
    observation = task_pkl_io.read_demo(human_pkl_dir, load_demo_idx)
    views = build_views(args, calibration_data, observation, ref_pts)

    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5))
    axes = np.atleast_1d(axes)
    plt.subplots_adjust(bottom=0.28, top=0.90, wspace=0.05)
    for view, ax in zip(views, axes):
        view["ax"] = ax
        view["im"] = ax.imshow(view["frame_bgr"][:, :, ::-1])  # BGR -> RGB
        ax.set_title(view["camera_name"])
        ax.axis("off")

    current = {"R": base_ori}

    def render(roll, pitch, yaw):
        adjustment = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        ori = base_ori @ adjustment  # local (gripper-frame) correction, matches
                                      # build_orientation_reference.py's --adjust_rotvec
        # One shared 3D pose, projected independently into each camera -- that's
        # what makes the two views a genuine cross-check rather than two
        # renderings of the same 2D guess.
        gripper_pts3d = gripper_points_3d(ori, anchor_pos)
        for view in views:
            gripper_pts2d = project(
                gripper_pts3d, view["K"], view["D"], view["r_vec"], view["t_vec"])
            frame = view["frame_bgr"].copy()
            frame = draw_object_points(frame, view["ref_pts_2d"])  # magenta: annotated ref kps
            frame = draw_points(frame, gripper_pts2d)               # colored: gripper at this ori
            view["im"].set_data(frame[:, :, ::-1])
        return ori

    saved = {"flag": False}

    def refresh_title():
        rv = R.from_matrix(current["R"]).as_rotvec()
        suffix = "   [SAVED]" if saved["flag"] else ""
        mode = f"demo {args.demo_idx} OVERRIDE" if per_demo else f"demo {args.ref_demo_idx} reference"
        fig.suptitle(
            f"{args.task_name}  {mode}   "
            f"rotvec=[{rv[0]:.3f}, {rv[1]:.3f}, {rv[2]:.3f}]{suffix}"
        )

    def on_change(_):
        current["R"] = render(s_roll.val, s_pitch.val, s_yaw.val)
        saved["flag"] = False  # any further adjustment invalidates the saved marker
        refresh_title()
        fig.canvas.draw_idle()

    ax_roll = plt.axes([0.15, 0.17, 0.7, 0.03])
    ax_pitch = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_yaw = plt.axes([0.15, 0.07, 0.7, 0.03])
    s_roll = Slider(ax_roll, "Roll", -180, 180, valinit=0)
    s_pitch = Slider(ax_pitch, "Pitch", -180, 180, valinit=0)
    s_yaw = Slider(ax_yaw, "Yaw", -180, 180, valinit=0)
    s_roll.on_changed(on_change)
    s_pitch.on_changed(on_change)
    s_yaw.on_changed(on_change)

    ax_reset = plt.axes([0.15, 0.015, 0.15, 0.04])
    b_reset = Button(ax_reset, "Reset")

    def on_reset(_):
        s_roll.reset()
        s_pitch.reset()
        s_yaw.reset()

    b_reset.on_clicked(on_reset)

    ax_save = plt.axes([0.7, 0.015, 0.15, 0.04])
    b_save = Button(ax_save, "Save Demo Override" if per_demo else "Save Reference")

    def on_save(_):
        if per_demo:
            # Store the DELTA from this demo's derived pose, not the absolute
            # orientation: Step 3 re-derives the base every run (it moves if the
            # reference or the tracking changes), and a correction expressed
            # relative to it stays meaningful, where an absolute pose would
            # silently go stale. base_ori.T @ R is the local-frame residual,
            # matching the right-multiplied convention used everywhere else.
            delta = R.from_matrix(base_ori.T @ current["R"]).as_rotvec()
            path = save_override(args.task_name, args.demo_idx, delta)
            print(f"Saved demo {args.demo_idx} override {np.round(delta, 5)} -> {path}")
        else:
            save_reference(ref_path, ref_pts, current["R"])
            print(f"Saved reference orientation to {ref_path}")
        saved["flag"] = True
        refresh_title()
        fig.canvas.draw_idle()

    b_save.on_clicked(on_save)

    on_change(None)
    plt.show()


if __name__ == "__main__":
    main()
