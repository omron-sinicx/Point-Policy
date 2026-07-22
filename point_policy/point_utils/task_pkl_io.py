"""
Shared I/O helpers for "task pkl" directories (the human pkl produced by
convert_to_pkl_human.py and the robot pkl produced by convert_pkl_human_to_robot.py).

These used to be a single pickle file (`{task_name}.pkl`) containing
`{"observations": [obs_0, obs_1, ...], "max_cartesian": ..., ...}`. Writers
built the whole `observations` list in memory (every demo's full-resolution
video frames included) before a single final `pkl.dump` -- with enough demos
this exceeds available RAM (observed: OOM-killed at ~30 demos).

Instead, a "task pkl" is now a directory:
    {task_name}/
        meta.pkl        -- {"max_cartesian", "min_cartesian", "max_gripper", "min_gripper"}
        demo_0000.pkl   -- one demo's observation dict
        demo_0001.pkl
        ...
one file per demo (zero-padded to match the source demonstration_N numbering),
so a writer never needs to hold more than one demo's frames in memory at a
time, and a reader that only wants one demo (label_points.py, the visualize_*
scripts) gets true O(1) random access instead of loading everything.
"""

import pickle as pkl
import re
from pathlib import Path

_DEMO_RE = re.compile(r"demo_(\d+)\.pkl$")


def _demo_path(task_dir: Path, demo_id: int) -> Path:
    return Path(task_dir) / f"demo_{demo_id:04d}.pkl"


def _meta_path(task_dir: Path) -> Path:
    return Path(task_dir) / "meta.pkl"


def write_meta(task_dir, meta: dict) -> None:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(_meta_path(task_dir), "wb") as f:
        pkl.dump(meta, f)


def read_meta(task_dir) -> dict:
    with open(_meta_path(task_dir), "rb") as f:
        return pkl.load(f)


def write_demo(task_dir, demo_id: int, observation: dict) -> None:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(_demo_path(task_dir, demo_id), "wb") as f:
        pkl.dump(observation, f)


def read_demo(task_dir, demo_id: int) -> dict:
    with open(_demo_path(task_dir, demo_id), "rb") as f:
        return pkl.load(f)


def iter_demo_ids(task_dir) -> list:
    """Sorted list of demo ids present, parsed from demo_NNNN.pkl filenames."""
    task_dir = Path(task_dir)
    if not task_dir.exists():
        return []
    ids = []
    for p in task_dir.iterdir():
        m = _DEMO_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def iter_demos(task_dir):
    """Generator yielding (demo_id, observation) pairs, one at a time --
    O(1) peak memory regardless of task size."""
    task_dir = Path(task_dir)
    for demo_id in iter_demo_ids(task_dir):
        yield demo_id, read_demo(task_dir, demo_id)


def load_all_demos(task_dir) -> list:
    """Materializes every demo into a list. O(N) memory -- prefer iter_demos()
    for anything processing a whole task; use this only for small tasks /
    one-off analysis."""
    return [obs for _, obs in iter_demos(task_dir)]
