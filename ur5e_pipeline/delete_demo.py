"""
Delete one or more noisy episodes from the pipeline.

A demo lives in up to three places:
    processed_data/<task>/demonstration_N/                          -- raw videos + states.csv
    processed_data_pkl/<task>/demo_{N:04d}.pkl                       -- human pkl (tracked points)
    processed_data_pkl/expert_demos/<env>/<task>/demo_{N:04d}.pkl    -- robot pkl

Deleting from both pkl directories means convert_pkl_human_to_robot.py does
NOT need to be re-run -- only convert_pkl_to_lerobot.py --overwrite, to
regenerate the LeRobot dataset without the deleted episode(s).

Also recomputes each pkl directory's meta.pkl (max/min cartesian & gripper)
over the remaining demos, since a naive delete would leave those stats
reflecting a demo that no longer exists.

Usage:
    python delete_demo.py --data_dir /path/to/data --task_name pick_place_02 \\
        --demo_ids 5 12 --dry_run

    python delete_demo.py --data_dir /path/to/data --task_name pick_place_02 \\
        --demo_ids 5 12 --yes
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "point_policy"))
from point_utils import task_pkl_io


def recompute_meta(task_dir: Path) -> None:
    """Rebuild meta.pkl from scratch over whatever demos remain in task_dir."""
    max_cartesian, min_cartesian = None, None
    max_gripper, min_gripper = None, None
    n = 0

    for _demo_id, demo in task_pkl_io.iter_demos(task_dir):
        n += 1
        cartesian_states = demo["cartesian_states"]
        gripper_states = demo["gripper_states"]

        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

    if n == 0:
        print(f"  No demos remain in {task_dir}; leaving meta.pkl as-is.")
        return

    task_pkl_io.write_meta(task_dir, {
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
    })
    print(f"  Recomputed meta.pkl for {task_dir} over {n} remaining demo(s).")


def main():
    parser = argparse.ArgumentParser(description="Delete one or more episodes from the pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--env_name", type=str, default="franka_env",
                        help="Subfolder under expert_demos/ (default: franka_env)")
    parser.add_argument("--demo_ids", type=int, nargs="+", required=True,
                        help="One or more demo numbers to delete (matches demonstration_N / demo_NNNN.pkl)")
    parser.add_argument("--use_gt_depth", action="store_true", help="Task was processed with gt depth")
    parser.add_argument("--dry_run", action="store_true", help="Report what would be deleted, delete nothing")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="Only used to build the printed convert_pkl_to_lerobot.py command "
                             "(pass this if you used a custom --repo_id when exporting)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Only used to build the printed convert_pkl_to_lerobot.py command")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    task_name = args.task_name
    if args.use_gt_depth:
        task_name += "_gt_depth"

    raw_dir = data_dir / "processed_data" / task_name
    human_pkl_dir = data_dir / "processed_data_pkl" / task_name
    robot_pkl_dir = data_dir / "processed_data_pkl" / "expert_demos" / args.env_name / task_name

    targets = []
    for demo_id in args.demo_ids:
        paths = [
            raw_dir / f"demonstration_{demo_id}",
            human_pkl_dir / f"demo_{demo_id:04d}.pkl",
            robot_pkl_dir / f"demo_{demo_id:04d}.pkl",
        ]
        existing = [p for p in paths if p.exists()]
        if not existing:
            print(f"Demo {demo_id}: nothing found at any of the expected locations, skipping.")
            continue
        targets.append((demo_id, existing))
        print(f"Demo {demo_id}: will delete")
        for p in existing:
            print(f"    {p}")

    if not targets:
        print("Nothing to delete.")
        return

    if args.dry_run:
        print("\n--dry_run: nothing was actually deleted.")
        return

    if not args.yes:
        confirm = input(f"\nPermanently delete {len(targets)} demo(s) listed above? (y/n): ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    for demo_id, paths in targets:
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        print(f"Demo {demo_id}: deleted.")

    print("\nRecomputing meta.pkl over remaining demos...")
    recompute_meta(human_pkl_dir)
    recompute_meta(robot_pkl_dir)

    print(
        f"\nDone. Re-run convert_pkl_to_lerobot.py --overwrite for '{task_name}' "
        "to regenerate the LeRobot dataset without the deleted episode(s)."
    )


if __name__ == "__main__":
    main()
