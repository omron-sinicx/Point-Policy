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
parser.add_argument("--point_jitter_m", type=float, default=0.0,
                    help="Std-dev (METRES) of i.i.d. Gaussian noise added per frame to "
                         "observation.environment_state. 0 (default) writes the exact "
                         "triangulated values. The dataset stores exact keypoints, but at "
                         "rollout tracking_node.py re-triangulates them live and after a "
                         "camera move + recalibration they will not reproduce the recorded "
                         "values exactly, so a policy trained on exact points relies on "
                         "precision it will not get. Scale this to the real uncertainty: on "
                         "this rig the static keypoint's across-demo std is ~0.5 mm, live "
                         "frame-to-frame jitter ~0.1-0.2 mm, and post-recalibration "
                         "reproducibility measured ~0.3 mm -- so 0.001 is already generous. "
                         "Beware large values: the tracked keypoints are the most "
                         "phase-informative input the policy has, and drowning them in noise "
                         "makes it fall back on predicting the dominant motion branch.")
parser.add_argument("--jitter_replicas", type=int, default=0,
                    help="Write this many EXTRA copies of every episode, each with an "
                         "independent noise draw (total = (1 + N) x episodes). Needs "
                         "--point_jitter_m. Because the noise is baked in, one copy gives each "
                         "frame a single fixed perturbation that the model can memorize over "
                         "many epochs; replicas restore some of the variety that per-epoch "
                         "resampling would give, and multiply a small demo set at the same "
                         "time. 0 (default) jitters in place, leaving the episode count alone.")
parser.add_argument("--jitter_seed", type=int, default=0,
                    help="RNG seed for the jitter, so an export is reproducible (default: 0)")
parser.add_argument("--drop_keypoints", nargs="+", type=int, default=[],
                    help="Keypoint INDICES to leave out of observation.environment_state "
                         "(0-based, in the order convert_pkl_human_to_robot.py wrote them). "
                         "Use this to drop a keypoint that carries no task information -- e.g. a "
                         "static fixture point, which is constant within a session and so acts "
                         "only as a session/calibration fingerprint the policy can latch onto. "
                         "Measured on peg_hole_dir1_03: the static keypoint alone predicts the "
                         "motion branch at 56.7%% (chance is 57.4%%) while the moving keypoint "
                         "alone gets 96.2%%. NOTE this changes the feature dimension, so such a "
                         "dataset needs a fresh policy AND a matching --drop_env_keypoints on "
                         "evaluate_policy_points.py, which otherwise aborts on the "
                         "policy-dim-vs-tracking_node-dim check.")
args = parser.parse_args()

if args.jitter_replicas < 0:
    raise SystemExit(f"--jitter_replicas must be >= 0, got {args.jitter_replicas}")
if args.jitter_replicas and args.point_jitter_m <= 0:
    raise SystemExit("--jitter_replicas needs --point_jitter_m > 0 (replicas of identical "
                     "un-jittered episodes would just duplicate the data)")
if args.point_jitter_m < 0:
    raise SystemExit(f"--point_jitter_m must be >= 0, got {args.point_jitter_m}")
if len(set(args.drop_keypoints)) != len(args.drop_keypoints):
    raise SystemExit(f"--drop_keypoints has duplicates: {args.drop_keypoints}")
if any(i < 0 for i in args.drop_keypoints):
    raise SystemExit(f"--drop_keypoints indices must be >= 0, got {args.drop_keypoints}")

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

bad = [i for i in args.drop_keypoints if i >= n_points]
if bad:
    raise SystemExit(f"--drop_keypoints {bad} out of range: this task has {n_points} keypoint(s) "
                     f"(valid indices 0..{n_points - 1})")
keep_idx = np.array([i for i in range(n_points) if i not in set(args.drop_keypoints)], dtype=int)
if keep_idx.size == 0:
    raise SystemExit("--drop_keypoints would remove every keypoint, leaving "
                     "observation.environment_state empty")
n_points_kept = int(keep_idx.size)
if args.drop_keypoints:
    print(f"\nDROPPING keypoint(s) {sorted(set(args.drop_keypoints))} of {n_points}: "
          f"observation.environment_state is {n_points_kept * 3} dims, not {n_points * 3}")
    print(f"  keeping indices {keep_idx.tolist()}")
    print(f"  -> needs a fresh policy, and evaluate_policy_points.py "
          f"--drop_env_keypoints {' '.join(str(i) for i in sorted(set(args.drop_keypoints)))} at rollout\n")
pose_key = _pose_source(first_demo)
print(
    f"Using '{pose_key}' as the 6dof pose source "
    + ("(no real robot state recorded; using hand-tracking-derived TCP target)"
       if pose_key == "robot_tcp_poses" else "(real recorded robot state)")
)

features = {
    "observation.state": {"dtype": "float32", "shape": (10,), "names": None},
    "observation.environment_state": {"dtype": "float32", "shape": (n_points_kept * 3,), "names": None},
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

rng = np.random.default_rng(args.jitter_seed)
if args.point_jitter_m > 0:
    print(f"\nPOINT JITTER: sigma = {args.point_jitter_m * 1000:.2f} mm per frame on "
          f"observation.environment_state (seed {args.jitter_seed})")
    if args.jitter_replicas:
        print(f"  writing {1 + args.jitter_replicas} independently-jittered copies of each episode")
    print(f"  -> baked into {repo_id}; keep an un-jittered export around to A/B against\n")

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
    # Select keypoints while the (T, N, 3) axis is still intact, so --drop_keypoints
    # indices mean keypoints rather than flattened components.
    points = demo["object_tracks_3d_pixels1"][:, keep_idx, :].reshape(
        -1, n_points_kept * 3).astype(np.float32)  # (T, N_kept*3)

    T = pose.shape[0]
    if T < 2:
        print(f"Demo {demo_id}: only {T} frame(s), skipping (need >=2 for a state->next-state pair)")
        continue

    ortho6 = np.stack([rotvec_to_ortho6(pose[t, 3:6]) for t in range(T)])  # (T, 6)
    state = np.concatenate([pose[:, :3], ortho6, gripper_states[:, None]], axis=1)  # (T, 10)

    # Only observation.environment_state is jittered. observation.state / action
    # come from the robot's own pose, not from vision, so tracker error does not
    # apply to them -- and perturbing the action would change what "correct" means
    # rather than making the policy robust.
    for rep in range(1 + args.jitter_replicas):
        pts = points
        if args.point_jitter_m > 0:
            pts = points + rng.normal(
                0.0, args.point_jitter_m, size=points.shape).astype(np.float32)

        for t in range(T - 1):
            dataset.add_frame({
                "observation.state": state[t],
                "observation.environment_state": pts[t],
                "action": state[t + 1],
                "task": task_name,
            })
        dataset.save_episode()
        n_saved += 1
        tag = "" if not args.jitter_replicas else f" [replica {rep + 1}/{1 + args.jitter_replicas}]"
        print(f"Demo {demo_id}: saved episode with {T - 1} frames{tag}")

dataset.finalize()
print(f"Done. {n_saved} episode(s) saved to {dataset_root}")
