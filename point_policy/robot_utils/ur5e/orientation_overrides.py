"""Per-demo frame-0 gripper orientation corrections.

The orientation reference in coordinates/<task>/orientation_reference.npz is
task-wide: convert_pkl_human_to_robot.py loads it once before the demo loop and
every demo's frame-0 pose is Procrustes-aligned against it. That is the right
default -- but individual demos still come out wrong, usually because DIFT
mislocalized that demo's frame-0 orientation keypoints, and a task-wide knob
cannot fix one demo without moving all the others.

A frame-0 error is a CONSTANT rotational bias over the whole episode: Step 3
sets robot_ori_0 at t=0 and every later frame is the hand's rigid motion
left-multiplied onto it (convert_pkl_human_to_robot.py's `rot @ robot_ori_0`).
So one constant per-demo rotation is a complete fix, not an approximation.

Stored as coordinates/<task>/orientation_overrides.json:

    {
      "12": [0.0, 0.0, 0.31],
      "75": {"method": "fingertip"},
      "76": {"method": "fingertip", "rotvec": [0.12, -0.05, 0.0]}
    }

Two independent things can be overridden per demo:

  rotvec  a rotation (radians) in the gripper's OWN frame, applied
          RIGHT-multiplied -- the same local-frame convention as
          build_orientation_reference.py's --adjust_rotvec and
          interactive_orientation_adjust.py's sliders, so a value tuned in the
          picker means the same thing here.

  method  which estimator derives this demo's frame-0 orientation, overriding
          the run's --orientation_method. The reason this exists: the
          object_keypoints method anchors on the grasped object, which is the
          right choice only when the object IS grasped at frame 0. A demo that
          opens with an empty hand reaching for the object has its keypoints
          somewhere unrelated to the gripper, and no rotation can repair that --
          but the hand is perfectly good, so "fingertip" or "plane_fit" works.

A bare list is shorthand for {"rotvec": [...]} and stays readable for the common
case. Both keys are optional; a demo may set either or both. When both are set
the rotvec is applied on top of the pose the chosen method produced.

Deliberately a separate module rather than part of object_keypoint_orientation:
overrides apply whatever --orientation_method produced the frame-0 pose, while
that module is specific to the object_keypoints method.
"""

import json
from pathlib import Path

import numpy as np

# .../point_policy/robot_utils/ur5e -> repo root, matching the REPO_ROOT the
# other ur5e scripts derive for locating coordinates/.
REPO_ROOT = Path(__file__).resolve().parents[3]

FILENAME = "orientation_overrides.json"


def overrides_path(task_name):
    return REPO_ROOT / "coordinates" / task_name / FILENAME


VALID_METHODS = ("fingertip", "plane_fit", "object_keypoints")


def load_overrides(task_name):
    """{demo_id: {"rotvec": (3,) array or None, "method": str or None}}.

    Empty dict if nothing is set -- a missing file is the normal case, not an
    error, so the override layer is inert unless somebody tuned a demo.
    """
    path = overrides_path(task_name)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)

    out = {}
    for key, entry in raw.items():
        if isinstance(entry, dict):
            rotvec, method = entry.get("rotvec"), entry.get("method")
            unknown = set(entry) - {"rotvec", "method"}
            if unknown:
                raise ValueError(f"{path}: demo {key} has unknown key(s) {sorted(unknown)}")
        else:
            rotvec, method = entry, None      # bare list shorthand

        if rotvec is not None:
            rotvec = np.asarray(rotvec, dtype=np.float64)
            if rotvec.shape != (3,):
                raise ValueError(
                    f"{path}: demo {key} rotvec has shape {rotvec.shape}, expected "
                    f"a 3-element rotation vector"
                )
        if method is not None and method not in VALID_METHODS:
            raise ValueError(
                f"{path}: demo {key} method {method!r} is not one of {VALID_METHODS}"
            )
        out[int(key)] = {"rotvec": rotvec, "method": method}
    return out


def save_override(task_name, demo_id, rotvec=None, method=None, drop_rotvec=False):
    """Merge one demo's correction into the file, leaving the others alone.

    Read-modify-write rather than overwrite: demos are tuned one at a time in
    separate runs of the picker, so a plain dump would discard every previously
    tuned demo. Passing only `method` likewise preserves that demo's existing
    rotvec, and vice versa.

    drop_rotvec: forget any saved rotation for this demo. Use it when changing
    `method`, since a rotation tuned against one estimator's output is
    meaningless on top of a different one's.
    """
    if method is not None and method not in VALID_METHODS:
        raise ValueError(f"method {method!r} is not one of {VALID_METHODS}")

    path = overrides_path(task_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    key = str(int(demo_id))
    prev = existing.get(key)
    prev = prev if isinstance(prev, dict) else ({"rotvec": prev} if prev else {})

    entry = dict(prev)
    if drop_rotvec:
        entry.pop("rotvec", None)
    if rotvec is not None:
        entry["rotvec"] = [float(v) for v in np.asarray(rotvec).ravel()]
    if method is not None:
        entry["method"] = method

    # Collapse to the bare-list shorthand when a rotvec is all there is.
    existing[key] = entry["rotvec"] if set(entry) == {"rotvec"} else entry

    # Numeric key order, so the file stays readable as it accumulates demos.
    ordered = {k: existing[k] for k in sorted(existing, key=int)}
    with open(path, "w") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")
    return path


def clear_override(task_name, demo_id):
    """Drop one demo's correction. Returns True if there was one to drop."""
    path = overrides_path(task_name)
    if not path.exists():
        return False
    with open(path) as f:
        existing = json.load(f)
    if str(int(demo_id)) not in existing:
        return False
    del existing[str(int(demo_id))]
    with open(path, "w") as f:
        json.dump({k: existing[k] for k in sorted(existing, key=int)}, f, indent=2)
        f.write("\n")
    return True
