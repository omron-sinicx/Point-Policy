"""
Derive a time-REVERSED LeRobotDataset from an existing one.

Unscrewing a cap and screwing it on are the same motion in opposite time
directions, so a "close" dataset can be built by playing every "open" episode
backwards -- no new recording, no HaMeR/DIFT re-run.

This is well-formed on this stack because `action` is an ABSOLUTE base-frame EE
pose, not a delta: replay_pkl_episode.py:102-116 and
evaluate_policy_points.py:2158-2175 both decode it as
[x, y, z, ortho6d(6), gripper] and feed it through a delta-clipped compliance
controller. Nothing needs a sign flip, and the per-step delta magnitudes the
safety clipper sees are identical in reverse. evaluate_policy_points.py:955-965
(rewind_through_poses) already drives the arm backwards through recorded poses
on a safety trip.

THE THING THAT MUST NOT BE GOT WRONG
------------------------------------
convert_pkl_to_lerobot.py:269-274 writes `action[t] = state[t+1]`, so in the
source `action[t] == state[t+1]` EXACTLY. Reversing the action column alongside
the others yields `action[t] = state[t-1]` -- off by two frames, and entirely
plausible-looking in the data. Actions are therefore RE-DERIVED from the
reversed state sequence here, never reversed.

Each reversed episode is one frame shorter than its source (N -> N-1). That is
deliberate: it keeps observation.state and observation.environment_state exactly
aligned. The source's final `action` holds a pose whose keypoint frame was
dropped at export time (convert_pkl_to_lerobot.py loops `range(T - 1)`), so
reusing it would pair a state with the wrong keypoints.

CAVEATS -- read before training on the output
---------------------------------------------
1. Gripper debounce is causal and does NOT reverse. debounce_gripper()
   (convert_pkl_human_to_robot.py:174-189) requires GRIPPER_HOLD = 8 consecutive
   frames before switching, so every forward transition is up to 8 frames LATE.
   Reversed, that becomes up to 8 frames EARLY -- the cap is released ~0.27 s
   (at 30 fps) before it should be. Benign for screw-on, since the cap is
   seated by then, but real. Pass --redebounce_gripper to re-apply the hold in
   the reversed time direction instead.
2. Reversed episodes start PRE-GRASPED: the gripper channel runs
   +1 (closed) -> -1 (open). A rollout must have the object already in the
   gripper -- test_task.yaml's `start_gripper: closed` plus the manual-insert
   prompt (replay_pkl_episode.py:529-541) supports exactly this.
3. observation.environment_state is tracked OBJECT keypoints. Reversed, they
   show a cap re-seating itself. The numbers are valid and self-consistent for
   behaviour cloning, but they are not causally producible by the robot alone.

Usage:
    python3 ur5e_pipeline/reverse_lerobot_dataset.py \
        --data_dir /root/osx-ur/data \
        --src bottle_open_0903_08_lerobot \
        --out bottle_close_0903_08_lerobot

Episode exclusions need no flag -- whatever was left out when the source was
exported is already absent, and the reversed dataset inherits that.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Matches convert_pkl_human_to_robot.py's GRIPPER_HOLD -- the number of
# consecutive frames a new gripper state must persist before it is accepted.
GRIPPER_HOLD = 8

# Index of the gripper channel within the 10-dim state/action vector
# ([0:3] position, [3:9] ortho6d orientation, [9] gripper).
GRIPPER_IDX = 9


def parse_args():
    p = argparse.ArgumentParser(
        description="Write a time-reversed copy of a LeRobotDataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory holding the source dataset and the output.")
    p.add_argument("--src", type=str, required=True,
                   help="Source dataset directory name under --data_dir, "
                        "e.g. bottle_open_0903_08_lerobot")
    p.add_argument("--out", type=str, required=True,
                   help="Name of the reversed dataset directory under --data_dir.")
    p.add_argument("--task_label", type=str, default=None,
                   help="Task string written on every frame (default: derived from "
                        "--out with a trailing '_lerobot' stripped).")
    p.add_argument("--episodes", nargs="+", type=int, default=None,
                   help="Only reverse these source episode indices (default: all). "
                        "Useful for a one-episode trial before committing.")
    p.add_argument("--redebounce_gripper", action="store_true",
                   help="Re-apply debounce_gripper's GRIPPER_HOLD hysteresis in the "
                        "reversed time direction, correcting the up-to-8-frame early "
                        "release described in the module docstring. Off by default so "
                        "the output is a pure reversal of the source.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing output directory.")
    return p.parse_args()


def debounce(states):
    """Require GRIPPER_HOLD consecutive frames in a new state before switching.

    A port of convert_pkl_human_to_robot.py's debounce_gripper() so the reversed
    channel gets the same hysteresis the forward one did, applied in the
    direction the reversed episode actually plays.
    """
    out = list(states)
    current = candidate = states[0]
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
    return np.asarray(out, dtype=states.dtype)


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    src_root = data_dir / args.src
    out_root = data_dir / args.out
    task_label = args.task_label or args.out.removesuffix("_lerobot")

    if not (src_root / "meta" / "info.json").exists():
        sys.exit(f"ERROR: {src_root} is not a LeRobotDataset (no meta/info.json)")
    if out_root.exists():
        if not args.overwrite:
            sys.exit(f"ERROR: {out_root} already exists (pass --overwrite to replace)")
        shutil.rmtree(out_root)

    src = LeRobotDataset(args.src, root=src_root)
    meta = src.meta

    # Carry fps/robot_type/features straight over rather than restating them, so
    # the output matches by construction -- merge_lerobot_datasets.py's
    # preflight() rejects any dataset whose fps, robot_type or feature
    # shape/dtype disagrees with its peers.
    features = {
        name: {"dtype": f["dtype"], "shape": tuple(f["shape"]), "names": f.get("names")}
        for name, f in meta.features.items()
        if name in ("observation.state", "observation.environment_state", "action")
    }
    missing = {"observation.state", "observation.environment_state", "action"} - set(features)
    if missing:
        sys.exit(f"ERROR: source is missing expected feature(s): {sorted(missing)}")
    if meta.video_keys:
        sys.exit(f"ERROR: source has video keys {meta.video_keys}; this script only "
                 f"handles the state-vector datasets convert_pkl_to_lerobot.py writes")

    episodes = meta.episodes
    ep_rows = episodes.select(range(episodes.num_rows)).to_list()
    if args.episodes is not None:
        wanted = set(args.episodes)
        unknown = sorted(wanted - {r["episode_index"] for r in ep_rows})
        if unknown:
            sys.exit(f"ERROR: episode(s) {unknown} not present in {args.src}")
        ep_rows = [r for r in ep_rows if r["episode_index"] in wanted]

    print(f"Reversing {len(ep_rows)} episode(s) from {src_root}")
    print(f"  task label : {task_label!r}")
    print(f"  output     : {out_root}")
    if args.redebounce_gripper:
        print(f"  re-debouncing gripper with GRIPPER_HOLD={GRIPPER_HOLD}")

    dst = LeRobotDataset.create(
        repo_id=args.out,
        fps=meta.fps,
        features=features,
        root=out_root,
        robot_type=meta.robot_type,
        use_videos=False,
    )

    n_frames = 0
    for row in ep_rows:
        a, b = row["dataset_from_index"], row["dataset_to_index"]
        frames = [src[i] for i in range(a, b)]
        state = np.stack([f["observation.state"].numpy() for f in frames])
        env = np.stack([f["observation.environment_state"].numpy() for f in frames])

        # Reverse observations only. The action is re-derived below from the
        # reversed state sequence -- see the module docstring.
        state = state[::-1].copy()
        env = env[::-1].copy()

        if args.redebounce_gripper:
            state[:, GRIPPER_IDX] = debounce(state[:, GRIPPER_IDX])

        for k in range(len(state) - 1):
            dst.add_frame({
                "observation.state": state[k],
                "observation.environment_state": env[k],
                "action": state[k + 1],
                "task": task_label,
            })
        dst.save_episode()
        n_frames += len(state) - 1
        print(f"Episode {row['episode_index']}: reversed, "
              f"{row['length']} -> {len(state) - 1} frames")

    dst.finalize()
    print(f"\nDone. {len(ep_rows)} episode(s), {n_frames} frames -> {out_root}")


if __name__ == "__main__":
    main()
