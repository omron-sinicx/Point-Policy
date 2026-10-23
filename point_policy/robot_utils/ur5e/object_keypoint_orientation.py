"""Object-keypoint + Procrustes hand orientation estimate.

Third alternative to franka/utils.py's compute_pinch_orientation() and
plane_fit_orientation.py's compute_pinch_orientation(), for tasks where the
hand geometry itself is unreliable throughout the whole demo (a sustained
pinch collapses both the fingertip-vector and plane-fit hand methods). The
grasped object, however, is rigid: its pose *is* the gripper's pose up to a
fixed offset. So this method reads orientation off object keypoints (detected
once via DIFT + label_points.py under the "gripper_orientation" label,
tracked the same way the existing "objects" keypoints are) instead of the
hand.

Unlike the two hand-based methods, this one does NOT need a "pre-negated"
convention relative to gripper_points.py's robot_base_orientation -- see
build_orientation_reference.py, which folds robot_base_orientation into the
saved reference orientation once, up front. compute_orientation()'s return
value is already final; callers must NOT compose it with
robot_base_orientation again.
"""

import importlib.util
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent

# franka/utils.py is loaded by explicit file path rather than via
# sys.path.insert(0, ".../franka"). Prepending that directory would put it
# AHEAD of ur5e/ on the path, so any consumer that imports `gripper_points`
# after importing this module would silently get franka/gripper_points.py --
# a different Tshift and different extrapoints, i.e. the wrong gripper
# geometry. (The ur5e scripts happen to import gripper_points first, but that
# made correctness depend on import order within each file, which an
# auto-import-sorter would happily break.) Loading by path touches nothing
# global.
_spec = importlib.util.spec_from_file_location(
    "_franka_utils_for_orientation", _here.parent / "franka" / "utils.py"
)
_franka_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_franka_utils)
rigid_transform_3D = _franka_utils.rigid_transform_3D


def compute_orientation(orient_points, ref_pts, ref_ori):
    """Frame-0 gripper orientation from N (N>=3) object keypoints.

    orient_points: (N, 3) object orientation keypoints, this demo's frame 0.
    ref_pts:       (N, 3) same keypoints' layout in the reference pose
                   (same point count and click order as orient_points).
    ref_ori:       (3, 3) trusted, FINAL gripper orientation at the reference
                   pose (already includes robot_base_orientation).

    Returns the (3, 3) final gripper orientation for this demo's frame 0.
    """
    R_rel, _ = rigid_transform_3D(ref_pts, orient_points)
    return R_rel @ ref_ori


def warn_if_degenerate(pts, ratio_threshold=0.15, min_extent_m=0.005):
    """Warn (print) if `pts` (N,3) are too close to COLLINEAR for Procrustes
    to pin down a rotation. Returns sv[1]/sv[0], the second/largest singular
    value ratio.

    Deliberately measures collinearity (sv[1]/sv[0]), NOT coplanarity
    (sv[2]/sv[0]). Three non-collinear points determine a rotation uniquely,
    so a flat, coplanar keypoint layout is perfectly well-conditioned -- an
    earlier version of this check flagged coplanarity and produced false
    alarms on good annotations. Measured on real data: a coplanar set
    spanning 25.8 x 11.4 x 0.4 mm recovered rotation to 2.26 deg median
    under 1 mm triangulation noise, BETTER than a synthetic volumetric set
    of comparable scale (3.61 deg), because in-plane spread is what supplies
    the lever arm.

    Coplanar points do make the Kabsch determinant sign noise-sensitive, so
    "reflection detected" messages from rigid_transform_3D are expected and
    harmless -- the correction returns the correct best-fit proper rotation.

    min_extent_m also warns when the points are simply too small overall:
    rotation accuracy scales with spread relative to triangulation noise,
    which matters far more than the layout's shape.
    """
    centroid = pts.mean(axis=0)
    sv = np.linalg.svd(pts - centroid, compute_uv=False)
    ratio = sv[1] / sv[0]
    if ratio < ratio_threshold:
        print(
            f"WARNING: orientation keypoints are nearly collinear "
            f"(second/largest singular value ratio = {ratio:.4f}) -- rotation "
            f"about that line will be poorly constrained. Annotate points that "
            f"are not all along one line (a flat, spread-out layout is fine)."
        )
    extent = sv[0] / np.sqrt(len(pts))
    if extent < min_extent_m:
        print(
            f"WARNING: orientation keypoints span only ~{extent * 1000:.1f} mm -- "
            f"too small relative to triangulation noise for a stable rotation. "
            f"Annotate points further apart."
        )
    return ratio


def load_orientation_inputs(
    data_dir, task_name, pixel_key, demo_idx,
    num_hand_points=9, orientation_label="gripper_orientation",
):
    """Load one demo's frame-0 hand keypoints and orientation-label object
    keypoints -- shared by build_orientation_reference.py and
    interactive_orientation_adjust.py. Raises RuntimeError with an actionable
    message if the task wasn't tracked with `orientation_label`, or has too
    few of its keypoints to constrain orientation via Procrustes.

    Returns (hand_point, ref_pts): (9,3) and (N,3) arrays.
    """
    from point_utils import task_pkl_io  # deferred: relies on caller's sys.path

    human_pkl_dir = Path(data_dir) / "processed_data_pkl" / task_name
    meta = task_pkl_io.read_meta(human_pkl_dir)
    object_labels = meta.get("object_labels")
    object_label_sizes = meta.get("object_label_sizes")
    if object_labels is None or object_label_sizes is None:
        raise RuntimeError(
            f"{human_pkl_dir}/meta.pkl has no object_labels/object_label_sizes -- "
            f"re-run Step 2 (convert_to_pkl_human_hamer.py) with "
            f"--object_labels objects {orientation_label} first."
        )
    if orientation_label not in object_labels:
        raise RuntimeError(
            f"Task '{task_name}' was not tracked with the '{orientation_label}' "
            f"label. Annotate it (label_points.py --label {orientation_label}) and "
            f"re-run Step 2 with --object_labels objects {orientation_label} first."
        )

    observation = task_pkl_io.read_demo(human_pkl_dir, demo_idx)
    human_tracks_3d = observation[f"human_tracks_3d_{pixel_key}"]
    frame0 = human_tracks_3d[0]  # (N, 3)
    hand_point = frame0[:num_hand_points]
    _, ref_pts = split_tracks_by_label(
        frame0, num_hand_points, object_labels, object_label_sizes,
        orientation_label=orientation_label,
    )

    if ref_pts.shape[0] < 3:
        raise RuntimeError(
            f"Only {ref_pts.shape[0]} '{orientation_label}' keypoint(s) found -- "
            f"need at least 3 to constrain orientation via Procrustes."
        )
    return hand_point, ref_pts


def save_reference(path, ref_pts, ref_ori):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ref_pts=ref_pts, ref_ori=ref_ori)


def load_reference(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No orientation reference found at {path}. Run "
            f"build_orientation_reference.py for this task first."
        )
    data = np.load(path)
    return data["ref_pts"], data["ref_ori"]


def split_tracks_by_label(
    tracks,
    num_hand_points,
    object_labels,
    object_label_sizes,
    orientation_label="gripper_orientation",
    axis=-2,
):
    """Split the object-point portion of a hand+object track array by label.

    tracks: (..., num_hand_points + sum(object_label_sizes), C) array -- works
    for both a single frame (N, C) and a whole episode (T, N, C), since axis=-2
    is the point axis in both layouts.

    object_labels / object_label_sizes: parallel lists describing the label
    (and point count) of each contiguous block of object points, in the same
    order they were concatenated during tracking (see points_class.py's
    track_points, which torch.cats semantic_similar_points in object_labels
    order).

    Returns (object_points, orientation_points): object_points concatenates
    every label except orientation_label, in their original relative order
    (this is exactly today's object_points_3d when orientation_label is
    absent from object_labels); orientation_points is that label's block, or
    an empty (..., 0, C) array if the label isn't present.
    """

    def _slice(a, start, end):
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(start, end)
        return a[tuple(idx)]

    obj_parts = []
    orient_part = None
    start = num_hand_points
    for label, size in zip(object_labels, object_label_sizes):
        end = start + size
        part = _slice(tracks, start, end)
        if label == orientation_label:
            orient_part = part
        else:
            obj_parts.append(part)
        start = end

    empty = _slice(tracks, num_hand_points, num_hand_points)
    object_points = np.concatenate(obj_parts, axis=axis) if obj_parts else empty
    orientation_points = orient_part if orient_part is not None else empty
    return object_points, orientation_points
