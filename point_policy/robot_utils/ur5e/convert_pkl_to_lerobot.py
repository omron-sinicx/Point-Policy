"""
Export a ROBOT_PKL (produced by convert_pkl_human_to_robot.py) to a
LeRobotDataset for a lightweight, non-visual imitation policy:

    observation.state             -- current EE cartesian pose (pos + ortho6d) + gripper (10,)
    observation.environment_state -- tracked object keypoints, world-frame 3D (N*3,)
    action                        -- the *next* EE cartesian pose + gripper (10,)

Orientation is encoded as ortho6d (Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks": the first two columns of the
rotation matrix, flattened) rather than axis-angle -- axis-angle has
regression-unfriendly discontinuities (angle wraparound at pi, axis-sign
ambiguity for small rotations) that ortho6d avoids. rotvec_to_ortho6() below
is a scipy-only reimplementation (no ur_control/ROS dependency here) verified
to produce bit-identical output to ur_control.math_utils.ortho6_from_quaternion
for the same rotation; the ROS-side consumer (replay_pkl_episode.py) decodes
it back via ur_control.transformations.quaternion_from_ortho6, which
round-trips to the same physical rotation.

The 6dof pose used for observation.state/action is picked per-task:
  - cartesian_states -- the real recorded robot pose, IF this task was
    collected via real teleoperation (data_collection.py).
  - robot_tcp_poses -- the hand-tracking-derived TCP target (Tshift-adjusted
    wrist position + orientation), used instead whenever cartesian_states is
    all-zero -- which is always the case for tasks collected via the
    image-only human-demo pipeline (collect_image_data.py), since there's no
    real robot state to record there; cartesian_states is just the
    DummyState zero placeholder in that path. robot_tcp_poses is what
    convert_pkl_human_to_robot.py actually computes as "where the robot
    should go" from the human demonstration, so it's the correct 6dof signal
    for these datasets. Detected automatically per-task (not per-demo --
    all demos in one task must agree, or this raises an error).

N (keypoint count) is read from the pkl itself, not hardcoded, since it
depends on how many object points were annotated per-task in label_points.py.

Since action[t] = state[t+1] by construction, each demo of length T yields
T-1 training frames (the last raw frame has no "next state" to serve as its
action, so it's dropped). object_tracks_3d_pixels1/2 are identical (3D world
points don't depend on camera), so only pixels1 is used.

Input:
    processed_data_pkl/expert_demos/{env_name}/{task_name}/  (directory of per-demo
    pkls, see point_utils/task_pkl_io.py; produced by convert_pkl_human_to_robot.py)

Output:
    {data_dir}/{repo_id}/  (default repo_id: {task_name}_lerobot)

Usage:
    cd point_policy/robot_utils/ur5e
    python convert_pkl_to_lerobot.py --data_dir /path/to/data --task_name pick_cup
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # point_policy/ (point_utils)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from point_utils import task_pkl_io


def rotvec_to_ortho6(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector -> continuous 6D rotation representation
    (first two columns of the rotation matrix, flattened). Matches
    ur_control.math_utils.ortho6_from_quaternion's convention exactly
    (verified bit-identical for the same rotation)."""
    rot_matrix = R.from_rotvec(rotvec).as_matrix()
    return rot_matrix[:, :2].T.flatten().astype(np.float32)


parser = argparse.ArgumentParser(
    description="Export a ROBOT_PKL to a LeRobotDataset (state + tracked points -> next state)"
)
parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
parser.add_argument("--task_name", type=str, required=True, help="Task name")
parser.add_argument("--use_gt_depth", action="store_true", help="Data was collected with gt depth")
parser.add_argument("--env_name", type=str, default="franka_env",
                    help="Subfolder under expert_demos/ (default: franka_env)")
parser.add_argument("--repo_id", type=str, default=None,
                    help="LeRobotDataset repo_id (default: {task_name}_lerobot)")
parser.add_argument("--fps", type=int, default=30,
                    help="Capture fps (not stored in the pkl itself)")
parser.add_argument("--overwrite", action="store_true",
                    help="Delete an existing dataset at the output root before writing")
parser.add_argument("--exclude_demos", nargs="+", type=int, default=[],
                    help="Demo id(s) to leave out of the exported dataset (e.g. a bad demo)")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
task_name = args.task_name
if args.use_gt_depth:
    task_name += "_gt_depth"

robot_pkl_dir = DATA_DIR / "processed_data_pkl" / "expert_demos" / args.env_name / task_name
repo_id = args.repo_id or f"{task_name}_lerobot"
dataset_root = DATA_DIR / repo_id

# Task pkls are directories of per-demo files (point_utils/task_pkl_io.py),
# not one big pickle -- avoids holding every demo's data in memory at once.
exclude_demos = set(args.exclude_demos)
demo_ids = [d for d in task_pkl_io.iter_demo_ids(robot_pkl_dir) if d not in exclude_demos]
if not demo_ids:
    raise ValueError(f"No demos found in {robot_pkl_dir} (after excluding {sorted(exclude_demos)})")
if exclude_demos:
    print(f"Excluding demo(s): {sorted(exclude_demos)}")

# cartesian_states is all-zero (DummyState placeholder) for tasks collected
# via the image-only human-demo pipeline -- fall back to robot_tcp_poses,
# the actual hand-tracking-derived robot target, in that case.
def _pose_source(demo):
    return "robot_tcp_poses" if np.allclose(demo["cartesian_states"], 0) else "cartesian_states"

# Peek the first demo to establish N (keypoint count, depends on how many
# object points were annotated for this task -- must be read from the data,
# not assumed) and the pose source; every other demo is checked against these
# while streaming below.
first_demo = task_pkl_io.read_demo(robot_pkl_dir, demo_ids[0])
n_points = first_demo["object_tracks_3d_pixels1"].shape[1]
pose_key = _pose_source(first_demo)
print(
    f"Using '{pose_key}' as the 6dof pose source "
    + ("(no real robot state recorded; using hand-tracking-derived TCP target)"
       if pose_key == "robot_tcp_poses" else "(real recorded robot state)")
)

features = {
    "observation.state": {"dtype": "float32", "shape": (10,), "names": None},
    "observation.environment_state": {"dtype": "float32", "shape": (n_points * 3,), "names": None},
    "action": {"dtype": "float32", "shape": (10,), "names": None},
}

if dataset_root.exists():
    if not args.overwrite:
        raise FileExistsError(
            f"{dataset_root} already exists. Pass --overwrite to replace it."
        )
    shutil.rmtree(dataset_root)

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=args.fps,
    features=features,
    root=dataset_root,
    robot_type="ur5e",
    use_videos=False,
)

n_saved = 0
for demo_id in demo_ids:
    demo = task_pkl_io.read_demo(robot_pkl_dir, demo_id)
    n = demo["object_tracks_3d_pixels1"].shape[1]
    if n != n_points:
        raise ValueError(
            f"Demo {demo_id} has {n} object keypoints, expected {n_points} (from demo "
            f"{demo_ids[0]}). All demos for a task must share the same annotated "
            "keypoint count to be written into one LeRobotDataset feature."
        )
    k = _pose_source(demo)
    if k != pose_key:
        raise ValueError(
            f"Demo {demo_id} would use '{k}' but demo {demo_ids[0]} uses '{pose_key}' "
            "-- all demos for a task must agree on whether cartesian_states is real "
            "or a zero placeholder."
        )

    pose = demo[pose_key].astype(np.float32)                            # (T, 6) = [pos(3), rotvec(3)]
    gripper_states = demo["gripper_states"].astype(np.float32)          # (T,)
    points = demo["object_tracks_3d_pixels1"].reshape(-1, n_points * 3).astype(np.float32)  # (T, N*3)

    T = pose.shape[0]
    if T < 2:
        print(f"Demo {demo_id}: only {T} frame(s), skipping (need >=2 for a state->next-state pair)")
        continue

    ortho6 = np.stack([rotvec_to_ortho6(pose[t, 3:6]) for t in range(T)])  # (T, 6)
    state = np.concatenate([pose[:, :3], ortho6, gripper_states[:, None]], axis=1)  # (T, 10)

    for t in range(T - 1):
        dataset.add_frame({
            "observation.state": state[t],
            "observation.environment_state": points[t],
            "action": state[t + 1],
            "task": task_name,
        })
    dataset.save_episode()
    n_saved += 1
    print(f"Demo {demo_id}: saved episode with {T - 1} frames")

dataset.finalize()
print(f"Done. {n_saved} episode(s) saved to {dataset_root}")
