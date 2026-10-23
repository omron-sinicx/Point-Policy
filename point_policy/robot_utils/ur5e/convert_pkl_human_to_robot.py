"""
Convert human hand keypoints in a pkl file to UR5e gripper keypoints.

This is the UR5e analogue of:
    point_policy/robot_utils/franka/convert_pkl_human_to_robot.py

The only differences are:
  - Imports Tshift, extrapoints, robot_base_orientation from ur5e/gripper_points.py
  - Can write output to a configurable env directory (default: franka_env so that
    existing training configs work without changes)

Input:
    processed_data_pkl/{task_name}.pkl
    (produced by convert_to_pkl_human.py — contains human_tracks_3d_pixels*)

Output:
    expert_demos/{env_name}/{task_name}.pkl
    (contains robot_tracks_3d_pixels*, robot_tracks_pixels*, object_tracks_*, gripper_states)

Usage:
    cd point_policy/robot_utils/ur5e
    python convert_pkl_human_to_robot.py \
        --data_dir /path/to/data \
        --calib_path /path/to/calib/calib.npy \
        --task_name pick_cup

    # If you used --use_gt_depth in convert_to_pkl_human.py:
    python convert_pkl_human_to_robot.py ... --use_gt_depth

Before running this script, open gripper_points.py and set:
  - Tshift: flange-to-TCP offset for your custom gripper
  - extrapoints: 8 body point offsets matching your gripper geometry
  - robot_base_orientation: rotation aligning hand frame to UR5e base frame
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[1]))          # point_policy/ (point_utils) -- inserted first, lowest priority
sys.path.insert(0, str(_here.parent / "franka"))  # franka/utils.py (rigid_transform_3D etc.)
sys.path.insert(0, str(_here))                     # ur5e/gripper_points.py takes priority over franka's

import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom, median_filter
from scipy.signal import savgol_filter

from gripper_points import extrapoints, Tshift, robot_base_orientation
from utils import camera2pixelkey, rigid_transform_3D
from utils import compute_pinch_orientation as compute_pinch_orientation_fingertip
from plane_fit_orientation import compute_pinch_orientation as compute_pinch_orientation_plane_fit
from orientation_overrides import load_overrides  # noqa: E402
from object_keypoint_orientation import (
    compute_orientation as compute_object_keypoint_orientation,
    load_reference as load_orientation_reference,
    split_tracks_by_label,
)
from point_utils import task_pkl_io

# point_policy/robot_utils/ur5e -> repo root, matching REPO_ROOT in run_pipeline.sh
REPO_ROOT = _here.parents[2]
ORIENTATION_LABEL = "gripper_orientation"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Convert human hand keypoints to UR5e gripper keypoints"
)
parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
parser.add_argument("--calib_path", type=str, required=True, help="Path to calib.npy")
parser.add_argument("--task_name", type=str, required=True, help="Task name")
parser.add_argument("--use_gt_depth", action="store_true", help="Data was collected with gt depth")
parser.add_argument("--num_demos", type=int, default=None,
                    help="Only convert demo ids < this value (matches Step 2's --num_demos "
                         "convention). Default: all demos found in the human pkl dir.")
parser.add_argument("--env_name", type=str, default="franka_env",
                    help="Subfolder under expert_demos/ (default: franka_env, keeps training configs intact)")
parser.add_argument("--save_image_size", nargs=2, type=int, default=[256, 256],
                    metavar=("W", "H"), help="Resize images to this size for training (default: 256 256)")
parser.add_argument("--orientation_method", type=str, default="fingertip",
                    choices=["fingertip", "plane_fit", "object_keypoints"],
                    help="Frame-0 orientation estimator: 'fingertip' (default) uses "
                         "single index/thumb-tip vectors; 'plane_fit' fits a plane + "
                         "level-paired correspondence over all 8 finger keypoints, which "
                         "stays well-conditioned during a sustained pinch (e.g. peg-in-hole); "
                         "'object_keypoints' Procrustes-aligns N>=3 tracked "
                         f"'{ORIENTATION_LABEL}' object keypoints against a saved reference "
                         "pose (see build_orientation_reference.py) -- reads the grasped "
                         "object's own pose instead of the hand, for tasks where the hand "
                         "geometry itself is unreliable throughout the whole demo.")

args = parser.parse_args()

compute_pinch_orientation = (
    compute_pinch_orientation_plane_fit
    if args.orientation_method == "plane_fit"
    else compute_pinch_orientation_fingertip
)

DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_name = args.task_name
use_gt_depth = args.use_gt_depth
save_image_size = tuple(args.save_image_size)  # (W, H) for cv2.resize

camera_indices = [1, 2]
num_hand_points = 9

# MediaPipe hand landmarks used for gripper center and open/closed detection:
#   index finger = landmarks 3, 4  (proximal, tip)
#   thumb        = landmarks 7, 8  (ip, tip)
index_finger_indices = [3, 4]
thumb_indices = [7, 8]

# Pairs of (index, thumb) landmarks to compute pinch distance
index_finger_thumb_pairs = [
    (idx1, idx2) for idx1 in index_finger_indices for idx2 in thumb_indices
]
PINCH_CLOSE_THRESHOLD = 0.07  # meters: fingers closer than this → gripper closed

# Smoothing / filtering parameters — tune these to taste
MEDIAN_WINDOW = 5        # frames: knocks out single-frame MediaPipe spikes
SAVGOL_WINDOW = 15       # frames (odd): Savitzky-Golay window — preserves motion peaks
SAVGOL_ORDER  = 3        # polynomial order for Savitzky-Golay
GRIPPER_HOLD  = 8        # minimum consecutive frames before gripper state switches

if use_gt_depth:
    task_name += "_gt_depth"

DATA_DIR_PKL = DATA_DIR / "processed_data_pkl"
SAVE_DIR = DATA_DIR_PKL / "expert_demos" / args.env_name

# Loaded once, up front, rather than per demo -- the reference pose is fixed
# for the whole task. Fail fast with an actionable message rather than partway
# through a long run.
# Per-demo frame-0 corrections, for the handful of demos the task-wide reference
# gets wrong (see orientation_overrides.py). A demo may override the rotation,
# the estimator, or both; the layer is inert when the file is absent.
orientation_overrides = load_overrides(task_name)
if orientation_overrides:
    print(f"Loaded {len(orientation_overrides)} per-demo orientation override(s): "
          f"{sorted(orientation_overrides)}")

# Load the reference if the RUN needs it or if any single demo overrides its way
# into needing it -- otherwise a task run with --orientation_method fingertip and
# one "object_keypoints" demo would reach the Procrustes call with ref_pts=None.
ref_pts = ref_ori = None
_needs_reference = args.orientation_method == "object_keypoints" or any(
    ov.get("method") == "object_keypoints" for ov in orientation_overrides.values()
)
if _needs_reference:
    ref_path = REPO_ROOT / "coordinates" / task_name / "orientation_reference.npz"
    ref_pts, ref_ori = load_orientation_reference(ref_path)

calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

# Task pkls are directories of per-demo files (point_utils/task_pkl_io.py),
# not one big pickle -- avoids holding every demo's frames in memory at once.
human_pkl_dir = DATA_DIR_PKL / task_name
robot_pkl_dir = SAVE_DIR / task_name
meta = task_pkl_io.read_meta(human_pkl_dir)

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------
def smooth_hand_points(pts):
    """Median + Savitzky-Golay filter on (T, N, 3) hand keypoint sequence."""
    T = pts.shape[0]
    # Step 1 — median filter: removes single-frame spikes from MediaPipe failures
    win_m = min(MEDIAN_WINDOW, T)
    pts = median_filter(pts.astype(np.float64), size=(win_m, 1, 1))
    # Step 2 — Savitzky-Golay: smooths continuous jitter while preserving peaks
    win_s = min(SAVGOL_WINDOW, T)
    win_s = win_s if win_s % 2 == 1 else win_s - 1  # must be odd
    if win_s >= SAVGOL_ORDER + 1:
        pts = savgol_filter(pts, window_length=win_s, polyorder=SAVGOL_ORDER, axis=0)
    return pts


def debounce_gripper(states):
    """Require GRIPPER_HOLD consecutive frames in new state before switching."""
    out = list(states)
    current = states[0]
    candidate = states[0]
    run_len = 0
    for i, s in enumerate(states):
        if s == candidate:
            run_len += 1
        else:
            candidate = s
            run_len = 1
        if run_len >= GRIPPER_HOLD:
            current = candidate
        out[i] = current
    return out


def smooth_rotvec(rotvecs):
    """Median + Savitzky-Golay filter on a (T,3) rotation-vector sequence.

    rigid_transform_3D can produce noisy per-frame rotation estimates (see
    smooth_hand_points' docstring above) -- but unlike position, this was
    never filtered before, even though robot_tcp_poses' orientation (what
    replay_pkl.py actually commands the real arm with) uses it directly.
    Filtered via quaternions with sign continuity enforced first: two
    rotvecs can represent the same rotation with opposite quaternion sign,
    which would otherwise look like noise to a naive per-component filter.
    """
    T = rotvecs.shape[0]
    quats = R.from_rotvec(rotvecs).as_quat()  # (T,4) [x,y,z,w]
    for i in range(1, T):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    win_m = min(MEDIAN_WINDOW, T)
    quats = median_filter(quats.astype(np.float64), size=(win_m, 1))
    win_s = min(SAVGOL_WINDOW, T)
    win_s = win_s if win_s % 2 == 1 else win_s - 1
    if win_s >= SAVGOL_ORDER + 1:
        quats = savgol_filter(quats, window_length=win_s, polyorder=SAVGOL_ORDER, axis=0)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # re-normalize after filtering
    return R.from_quat(quats).as_rotvec()


# ---------------------------------------------------------------------------
# Depth resize helper
# ---------------------------------------------------------------------------
def resize_depth_image(depth_image, new_size):
    zoom_factors = (new_size[1] / depth_image.shape[0],
                    new_size[0] / depth_image.shape[1])
    return zoom(depth_image, zoom_factors, order=1)


# ---------------------------------------------------------------------------
# Process each demonstration
# ---------------------------------------------------------------------------
human_demo_ids = task_pkl_io.iter_demo_ids(human_pkl_dir)
if args.num_demos is not None:
    # Matches Step 2's `demo_num >= NUM_DEMOS` cutoff -- skips ids the tracked
    # pass never touched, rather than crashing on their untracked raw pkl.
    human_demo_ids = [d for d in human_demo_ids if d < args.num_demos]

untracked_demo_ids = []

for obs_idx, demo_id in enumerate(human_demo_ids):
    observation = task_pkl_io.read_demo(human_pkl_dir, demo_id)
    print(f"Processing observation {obs_idx + 1}/{len(human_demo_ids)}")

    # Step 2 skips (rather than aborts on) any demo whose HaMeR or object
    # tracking raised -- see convert_to_pkl_human_hamer.py's Phase A/B
    # `save = False; continue` paths. Those demos keep only the raw pass's
    # keys (pixels*/cartesian_states/gripper_states), so they have no
    # human_tracks_3d_* to convert. Skip them with a loud warning instead of
    # crashing the whole run on the first one, the same way --num_demos
    # already avoids ids the tracked pass never touched.
    missing = [
        camera2pixelkey[f"cam_{cam_idx}"]
        for cam_idx in camera_indices
        if f"human_tracks_3d_{camera2pixelkey[f'cam_{cam_idx}']}" not in observation
    ]
    if missing:
        print(
            f"  WARNING: demo {demo_id} has no human_tracks_3d for {missing} -- "
            f"Step 2 skipped it (HaMeR or object tracking failed). Not converting."
        )
        untracked_demo_ids.append(demo_id)
        continue

    for cam_idx in camera_indices:
        camera_name = f"cam_{cam_idx}"
        pixel_key = camera2pixelkey[camera_name]

        # Resize RGB images — capture original dimensions first so we can
        # scale 2D pixel coordinates to match the resized frames later.
        pixels = observation[pixel_key]
        orig_H, orig_W = pixels[0].shape[:2]    # e.g. 480, 640
        save_W, save_H = save_image_size        # e.g. 256, 256
        scale_xy = np.array([save_W / orig_W, save_H / orig_H], dtype=np.float64)
        pixels = np.array([cv2.resize(p, save_image_size) for p in pixels])
        observation[pixel_key] = pixels

        if use_gt_depth:
            depth = observation.get(f"depth_{pixel_key}")
            if depth is not None:
                depth = np.array([resize_depth_image(d, save_image_size) for d in depth])
                observation[f"depth_{pixel_key}"] = depth

        # Human hand 3D tracks: shape (T, num_hand_points + num_obj_points, 3)
        human_tracks_3d = observation[f"human_tracks_3d_{pixel_key}"]
        hand_points_3d = human_tracks_3d[:, :num_hand_points]    # (T, 9, 3)
        hand_points_3d = smooth_hand_points(hand_points_3d)      # filter noise

        object_labels_meta = meta.get("object_labels")
        object_label_sizes_meta = meta.get("object_label_sizes")
        if object_label_sizes_meta is None:
            # Backward compat: pkls written before per-label sizes were
            # recorded -- every remaining point is "objects" (today's exact
            # behavior), no orientation keypoints.
            object_labels_meta = ["objects"]
            object_label_sizes_meta = [human_tracks_3d.shape[1] - num_hand_points]

        object_points_3d, orient_points_3d = split_tracks_by_label(
            human_tracks_3d, num_hand_points, object_labels_meta, object_label_sizes_meta,
            orientation_label=ORIENTATION_LABEL,
        )

        if args.orientation_method == "object_keypoints" and orient_points_3d.shape[1] < 3:
            raise RuntimeError(
                f"--orientation_method object_keypoints needs >=3 tracked "
                f"'{ORIENTATION_LABEL}' keypoints, found {orient_points_3d.shape[1]} for "
                f"task '{task_name}'. Annotate them (label_points.py --label "
                f"{ORIENTATION_LABEL}) and re-run Step 2 with --object_labels objects "
                f"{ORIENTATION_LABEL} first."
            )

        gripper_states_list = []
        human_poses_list = []

        # ------------------------------------------------------------------
        # Pick ONE index/thumb pair to anchor the TCP for the whole episode.
        #
        # The TCP anchor is the midpoint of a selected landmark pair, so ANY
        # change of selection teleports it -- the landmarks are far apart
        # (index 3 vs 4 ~32mm, thumb 7 vs 8 similar), and half that gap lands
        # straight in the anchor. A per-frame argmin re-decides independently
        # every frame, so during a sustained pinch (peg-in-hole) the two
        # smallest distances stay nearly tied and the choice flip-flops:
        # measured 102 jumps of 15-17mm across peg_hole_dir1_03's 50 demos,
        # crossing on margins as small as 0.08mm. Savitzky-Golay below then
        # smears each step into a smooth ~20-frame excursion, so it reads as
        # deliberate motion rather than an obvious spike.
        #
        # Per-frame hysteresis was tried first and is NOT sufficient: a
        # decisive switch (margin > threshold) still teleports the anchor, and
        # real data produced 22-29mm jumps at margins of 5-10mm. The anchor
        # has to be a single consistent physical point instead. A constant
        # offset from the "ideal" pair is harmless -- the policy absorbs it --
        # whereas a mid-episode teleport is spurious motion it would learn.
        #
        # Chosen by smallest MEDIAN distance over the episode (robust to a few
        # bad frames), i.e. the pair that is most consistently the pinch.
        # ------------------------------------------------------------------
        pair_dists = np.array([
            [np.linalg.norm(hp[i1] - hp[i2]) for i1, i2 in index_finger_thumb_pairs]
            for hp in hand_points_3d
        ])                                        # (T, 4)
        locked_pair_idx = int(np.argmin(np.median(pair_dists, axis=0)))
        i_idx, th_idx = index_finger_thumb_pairs[locked_pair_idx]

        # Diagnostic: if the per-frame argmin disagrees often, the grasp really
        # does move during the demo, so the locked anchor carries a varying
        # offset. Still preferable to teleporting, but worth surfacing.
        disagree = np.mean(np.argmin(pair_dists, axis=1) != locked_pair_idx)
        if disagree > 0.25:
            print(
                f"  demo {demo_id} {pixel_key}: anchor pair locked to "
                f"{index_finger_thumb_pairs[locked_pair_idx]}, but the per-frame "
                f"closest pair differs on {disagree * 100:.0f}% of frames -- "
                f"grasp point may shift during this demo."
            )

        for t_idx, hand_point in enumerate(hand_points_3d):
            # Gripper open/closed still uses the per-frame closest pair over
            # ALL pairs -- that's a question about whether the fingers are
            # together, independent of which pair anchors the TCP.
            min_dist = float(pair_dists[t_idx].min())

            # TCP center = midpoint between chosen finger and thumb
            robot_pos = (hand_point[i_idx] + hand_point[th_idx]) / 2.0

            # Orientation: rigid transform relative to the first frame
            if t_idx == 0:
                _ov = orientation_overrides.get(demo_id, {})
                # A demo may pick a different estimator than the run: object
                # keypoints only anchor the gripper when the object is actually
                # held at frame 0, so a demo that opens mid-reach is better
                # served by the hand-based methods.
                _method = _ov.get("method") or args.orientation_method
                if _method == "object_keypoints":
                    # Already final -- do NOT compose with robot_base_orientation
                    # again (see object_keypoint_orientation.py's docstring).
                    robot_ori_0 = compute_object_keypoint_orientation(
                        orient_points_3d[0], ref_pts, ref_ori
                    )
                else:
                    _fn = (compute_pinch_orientation_plane_fit
                           if _method == "plane_fit" else compute_pinch_orientation_fingertip)
                    robot_ori_0 = _fn(hand_point) @ robot_base_orientation

                # Per-demo correction, right-multiplied so it is a rotation in
                # the gripper's OWN frame -- same convention as the picker's
                # sliders and build_orientation_reference.py's --adjust_rotvec.
                # Applied to robot_ori_0 only: every later frame is the hand's
                # rigid motion composed onto it, so correcting frame 0 corrects
                # the whole episode.
                _delta = _ov.get("rotvec")
                if _delta is not None:
                    robot_ori_0 = robot_ori_0 @ R.from_rotvec(_delta).as_matrix()
                if _ov and cam_idx == camera_indices[0]:
                    _bits = []
                    if _ov.get("method"):
                        _bits.append(f"method={_ov['method']}")
                    if _delta is not None:
                        _bits.append(f"rotvec={np.round(_delta, 5)}")
                    print(f"  demo {demo_id}: orientation override ({', '.join(_bits)})")
                robot_ori = robot_ori_0
                base_hand_points = hand_point.copy()
            else:
                rot, _ = rigid_transform_3D(base_hand_points, hand_point.copy())
                robot_ori = rot @ robot_ori_0

            # Human pose: [x, y, z, rx, ry, rz] in world frame
            human_poses_list.append(
                np.concatenate([robot_pos, R.from_matrix(robot_ori).as_rotvec()])
            )
            gripper_states_list.append(-1 if min_dist >= PINCH_CLOSE_THRESHOLD else 1)

        # Smooth orientation across the whole episode -- rigid_transform_3D's
        # raw per-frame estimate was never filtered before, even though it
        # feeds robot_tcp_poses' orientation directly (what replay_pkl.py
        # commands the real arm with). This is the actual source of jitter
        # that's most visible at the gripper: a rotation error, applied over
        # the ~15.5cm Tshift lever arm, becomes an amplified position error
        # at the fingertips (see smooth_hand_points' call below, which only
        # ever smoothed that downstream symptom, not the rotation itself).
        human_poses_arr = np.array(human_poses_list)  # (T, 6)
        human_poses_arr[:, 3:] = smooth_rotvec(human_poses_arr[:, 3:])
        gripper_states_debounced = debounce_gripper(gripper_states_list)

        # Rebuild the 9-point gripper geometry from the smoothed orientation
        # and debounced gripper state (previously built inline per-frame from
        # the raw, unsmoothed rotation and the raw, undebounced pinch check --
        # so the visual fingertip narrowing could flicker independently of
        # the reported gripper_states value).
        robot_points_list = []
        for t_idx in range(len(human_poses_arr)):
            robot_pos = human_poses_arr[t_idx, :3]
            robot_ori = R.from_rotvec(human_poses_arr[t_idx, 3:]).as_matrix()

            T_g_world = np.eye(4)
            T_g_world[:3, :3] = robot_ori
            T_g_world[:3, 3] = robot_pos
            T_g_world = T_g_world @ Tshift  # flange -> TCP offset

            is_closed = gripper_states_debounced[t_idx] == 1
            points3d = [T_g_world[:3, 3]]  # center point
            for ep_idx, Tp in enumerate(extrapoints):
                Tp_local = Tp.copy()
                if is_closed and ep_idx in (0, 1):
                    # Close fingers: narrow the fingertip Y spread
                    Tp_local[1, 3] = 0.015 if ep_idx == 0 else -0.015
                pt = T_g_world @ Tp_local
                points3d.append(pt[:3, 3])
            robot_points_list.append(np.array(points3d))  # (9, 3)

        # Smooth robot 3D positions on top of the now-smoothed rotation, to
        # kill any remaining position-only noise (e.g. from the pinch-point
        # anchor itself).
        robot_points_smoothed = smooth_hand_points(np.array(robot_points_list))

        observation[f"robot_tracks_3d_{pixel_key}"] = robot_points_smoothed
        observation[f"object_tracks_3d_{pixel_key}"] = object_points_3d
        observation["gripper_states"] = np.array(gripper_states_debounced)
        observation["human_poses"] = human_poses_arr

        # TCP (end-effector) pose: same as human_poses but with the position
        # shifted by Tshift (flange -> TCP) and smoothed, i.e. exactly the
        # position already stored as robot_tracks_3d_*[:, 0, :] ("wrist_center").
        # Orientation is unchanged since Tshift is a pure translation. This is
        # the pose to actually command the robot with -- human_poses' position
        # is the pre-Tshift pinch-point anchor, not a valid end-effector target.
        observation["robot_tcp_poses"] = np.concatenate(
            [robot_points_smoothed[:, 0, :], human_poses_arr[:, 3:]],
            axis=1,
        )

        # Scale human 2D tracks: they were produced by CoTracker in the original
        # image space (orig_W × orig_H) but the frames are now save_image_size.
        h_2d_key = f"human_tracks_{pixel_key}"
        if h_2d_key in observation:
            observation[h_2d_key] = observation[h_2d_key] * scale_xy

        # Project smoothed 3D robot tracks → 2D pixel coordinates, then scale.
        P = calibration_data[camera_name]["ext"]       # 4×4 extrinsic
        K = calibration_data[camera_name]["int"]       # 3×3 intrinsic (orig resolution)
        D = calibration_data[camera_name]["dist_coeff"]
        r, t = P[:3, :3], P[:3, 3]
        r_vec, _ = cv2.Rodrigues(r)

        robot_tracks_2d = []
        for pts3d in robot_points_smoothed:
            pts2d = cv2.projectPoints(pts3d.astype(np.float32), r_vec, t, K, D)[0].squeeze()
            robot_tracks_2d.append(pts2d * scale_xy)
        observation[f"robot_tracks_{pixel_key}"] = np.array(robot_tracks_2d)

        object_tracks_2d = []
        for pts3d in object_points_3d:
            if pts3d.shape[0] == 0:
                object_tracks_2d.append(np.empty((0, 2)))
                continue
            pts2d = cv2.projectPoints(pts3d.astype(np.float32), r_vec, t, K, D)[0].squeeze()
            if pts2d.ndim == 1:
                pts2d = pts2d[np.newaxis]
            object_tracks_2d.append(pts2d * scale_xy)
        observation[f"object_tracks_{pixel_key}"] = np.array(object_tracks_2d)

    task_pkl_io.write_demo(robot_pkl_dir, demo_id, observation)

# max/min cartesian & gripper are passed through unchanged from the human pkl
# -- this script never modifies them.
task_pkl_io.write_meta(robot_pkl_dir, meta)

num_saved = len(human_demo_ids) - len(untracked_demo_ids)
print(f"\nDone. Saved {num_saved} demonstrations to {robot_pkl_dir}")
if untracked_demo_ids:
    print(
        f"WARNING: skipped {len(untracked_demo_ids)} untracked demo(s): "
        f"{untracked_demo_ids}\n"
        f"  Step 2 failed on these. Re-run Step 2 (SKIP_TRACKING=0) and read its "
        f"'Error in ...' tracebacks to recover them."
    )
