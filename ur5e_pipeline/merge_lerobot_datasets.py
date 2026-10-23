"""
Merge several LeRobotDatasets into one, optionally dropping bad episodes,
balancing each source's contribution, and collapsing everything to a single
task name.

Built entirely on lerobot's own APIs -- nothing here reimplements the index
bookkeeping:
  delete_episodes()   datasets/dataset_tools.py -- writes a NEW dataset with the
                      kept episodes re-indexed, so sources are never modified
  modify_tasks()      datasets/dataset_tools.py -- rewrites every place a task
                      string lives (meta/tasks.parquet, task_index in
                      data/**/*.parquet, the tasks column in
                      meta/episodes/**/*.parquet, and total_tasks in info.json)
  aggregate_datasets() datasets/aggregate.py    -- concatenates the datasets

Why the rename happens BEFORE aggregating: aggregate_datasets unifies tasks with
`pd.concat([m.tasks ...]).index.unique()`. If every source already carries the
same task string that collapses to exactly one row with task_index 0 and
total_tasks: 1, with no changes to aggregate.py. Renaming afterwards instead
would mean hand-editing the merged parquet files, and aggregate's
update_meta_data() never touches the episodes `tasks` column -- easy to leave the
dataset silently disagreeing with itself.

Usage:
    python3 ur5e_pipeline/merge_lerobot_datasets.py \
        --data_dir /root/osx-ur/data \
        --out peg_hole_merged_lerobot \
        --task_name peg_hole \
        --max_episodes 40 \
        --sources peg_hole_dir1_01_lerobot:4,7,30,32,33,42 \
                  peg_hole_dir3_00_lerobot:25,34,36,37,43,44,45,46,47,48 \
                  peg_hole_dir4_01_lerobot \
                  peg_hole_dir5_00_lerobot:26,35,39,41

Each source is NAME[:excluded,episode,ids] so exclusions stay attached to their
dataset rather than sitting in a positionally-matched second list.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import delete_episodes, modify_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_source(spec):
    """'name:1,2,3' -> ('name', [1,2,3]);  'name' -> ('name', [])"""
    if ":" not in spec:
        return spec, []
    name, _, ids = spec.partition(":")
    ids = ids.strip()
    if not ids:
        return name, []
    try:
        return name, sorted({int(x) for x in ids.split(",") if x.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad exclusion list in {spec!r} -- expected NAME:1,2,3"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge LeRobotDatasets, filtering episodes and unifying the task name.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory holding the source datasets and the output.")
    p.add_argument("--sources", nargs="+", type=str, required=True,
                   help="Source datasets as NAME[:excluded,episode,ids], e.g. "
                        "peg_hole_dir1_01_lerobot:4,7,30")
    p.add_argument("--out", type=str, required=True,
                   help="Name of the merged dataset directory under --data_dir.")
    p.add_argument("--task_name", type=str, default=None,
                   help="Task string applied to every episode (default: derived "
                        "from --out with a trailing '_lerobot' stripped).")
    p.add_argument("--keep_task_names", action="store_true",
                   help="Keep each source's own task string, producing a genuine "
                        "multi-task merge instead of one unified task.")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="After exclusions, keep only the first N episodes of each "
                        "source (drops the highest-numbered ones), so every source "
                        "contributes equally. Default: keep all.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing output directory.")
    return p.parse_args()


def preflight(roots):
    """Fail fast, naming the offending key, rather than letting
    aggregate_datasets' validate_all_metadata() raise a full-dict diff deep into
    the run. This is what rejects datasets annotated with a different
    object-keypoint count (environment_state width)."""
    infos = {}
    for name, root in roots.items():
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            sys.exit(f"ERROR: {root} is not a LeRobotDataset (no meta/info.json)")
        infos[name] = json.load(open(info_path))

    ref_name, ref = next(iter(infos.items()))
    for name, info in infos.items():
        for field in ("fps", "robot_type", "codebase_version"):
            if info.get(field) != ref.get(field):
                sys.exit(f"ERROR: {field} mismatch -- {name}={info.get(field)!r} "
                         f"but {ref_name}={ref.get(field)!r}")
        for key in set(info["features"]) | set(ref["features"]):
            a, b = info["features"].get(key), ref["features"].get(key)
            if a is None or b is None:
                sys.exit(f"ERROR: feature {key!r} present in only one dataset "
                         f"({name} vs {ref_name})")
            if tuple(a["shape"]) != tuple(b["shape"]) or a["dtype"] != b["dtype"]:
                sys.exit(f"ERROR: feature {key!r} differs -- {name}: "
                         f"{a['dtype']}{tuple(a['shape'])} vs {ref_name}: "
                         f"{b['dtype']}{tuple(b['shape'])}")
    return infos


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_root = data_dir / args.out

    task_name = args.task_name
    if task_name is None:
        task_name = args.out[: -len("_lerobot")] if args.out.endswith("_lerobot") else args.out

    parsed = [parse_source(s) for s in args.sources]
    if len({n for n, _ in parsed}) != len(parsed):
        sys.exit("ERROR: duplicate source names")
    roots = {n: data_dir / n for n, _ in parsed}
    for n, r in roots.items():
        if not r.exists():
            sys.exit(f"ERROR: source not found: {r}")

    if out_root.exists():
        if not args.overwrite:
            sys.exit(f"ERROR: {out_root} already exists (pass --overwrite to replace). "
                     f"Refusing, since aggregate_datasets would append into it.")
        shutil.rmtree(out_root)

    infos = preflight(roots)

    # Resolve the keep/drop set per source and show it before doing any work.
    plan = {}
    print(f"Merging {len(parsed)} dataset(s) -> {out_root}")
    print(f"  task string: " + ("<per-source, unchanged>" if args.keep_task_names else repr(task_name)))
    for name, excluded in parsed:
        total = infos[name]["total_episodes"]
        bad = [i for i in excluded if i < total]
        unknown = [i for i in excluded if i >= total]
        if unknown:
            sys.exit(f"ERROR: {name} has {total} episodes; excluded ids out of range: {unknown}")
        remaining = [i for i in range(total) if i not in set(bad)]
        if args.max_episodes is not None:
            if len(remaining) < args.max_episodes:
                sys.exit(f"ERROR: {name} has only {len(remaining)} episodes after "
                         f"exclusions, fewer than --max_episodes {args.max_episodes}")
            keep, trim = remaining[: args.max_episodes], remaining[args.max_episodes:]
        else:
            keep, trim = remaining, []
        plan[name] = {"keep": keep, "drop": sorted(set(bad) | set(trim))}
        print(f"  {name}: {total} eps | bad={bad or '-'} | trailing_trim={trim or '-'} "
              f"| keep={len(keep)}")

    staging = Path(tempfile.mkdtemp(prefix="lerobot_merge_"))
    try:
        staged_roots, staged_ids = [], []
        for name, _ in parsed:
            drop, keep = plan[name]["drop"], plan[name]["keep"]
            staged_id = f"{name}_staged"
            staged_root = staging / staged_id

            src = LeRobotDataset(name, root=roots[name])
            if drop:
                print(f"\n[{name}] dropping {len(drop)} episode(s) -> {len(keep)} kept")
                ds = delete_episodes(src, drop, output_dir=staged_root, repo_id=staged_id)
            else:
                # delete_episodes raises on an empty list, so copy instead.
                print(f"\n[{name}] nothing to drop; copying {len(keep)} episode(s)")
                shutil.copytree(roots[name], staged_root)
                ds = LeRobotDataset(staged_id, root=staged_root)

            if not args.keep_task_names:
                modify_tasks(ds, new_task=task_name)

            staged_roots.append(staged_root)
            staged_ids.append(staged_id)

        print(f"\nAggregating -> {out_root}")
        aggregate_datasets(
            repo_ids=staged_ids,
            aggr_repo_id=args.out,
            roots=staged_roots,
            aggr_root=out_root,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    merged = json.load(open(out_root / "meta" / "info.json"))
    print(f"\nDone -> {out_root}")
    print(f"  episodes={merged['total_episodes']}  frames={merged['total_frames']}  "
          f"tasks={merged['total_tasks']}")
    expected = sum(len(p['keep']) for p in plan.values())
    if merged["total_episodes"] != expected:
        sys.exit(f"ERROR: expected {expected} episodes, got {merged['total_episodes']}")


if __name__ == "__main__":
    main()
