"""Plane-fit + level-paired-correspondence hand orientation estimate.

Alternative to franka/utils.py's compute_pinch_orientation(), for hands that
stay in a sustained pinch for the whole demo (e.g. peg-in-hole): that version
derives orientation from single index/thumb-tip vectors, which become noisy
once the tips converge. This version fits over all 8 finger keypoints instead
of single vectors, so it stays well-conditioned in a tight, sustained pinch.

hand_point: (9, 3) array in the point order defined by
point_policy/point_utils/points_class.py:387-412 --
0=wrist 1=idx_MCP 2=idx_PIP 3=idx_DIP 4=idx_TIP
5=thm_CMC 6=thm_MCP 7=thm_IP 8=thm_TIP
"""

import numpy as np


def _fit_plane_normal(points):
    """Least-squares plane normal through `points` (Nx3): SVD's smallest
    right singular vector of the mean-centered points."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    return Vt[2]


def _fit_dominant_direction(vectors):
    """Single direction best explaining a set of vectors (Nx3): SVD's
    largest right singular vector, i.e. least-squares line-of-best-fit
    through the vectors treated as points around the origin."""
    _, _, Vt = np.linalg.svd(vectors)
    return Vt[0]


def compute_pinch_orientation(hand_point):
    """Estimate the hand's world-frame orientation from all 8 index/thumb
    keypoints, instead of single fingertip vectors.

    1. Lateral (X) axis: plane fit through all 8 finger keypoints (index +
       thumb, excluding wrist) -- fixes 2 of 3 orientation DOF from a
       least-squares fit over all 8 points rather than a cross product of
       two single vectors.
    2. Spread (Y) axis: pair joints by depth using the fixed landmark order
       (idx_MCP<->thm_CMC, idx_PIP<->thm_MCP, idx_DIP<->thm_IP,
       idx_TIP<->thm_TIP), take the 4 thumb-minus-index vectors, and fit the
       single direction that best explains all 4. This never depends on how
       parallel the two fingers are, unlike a common-perpendicular-of-two-
       independent-lines approach (which degenerates when the fingers are
       nearly parallel -- common during a sustained pinch).
    3. Forward (Z) axis: cross(X, Y), completing a right-handed orthonormal
       frame. Callers compose this matrix with gripper_points.py's
       robot_base_orientation (a pi rotation about X), which flips the Y and Z
       columns of whatever it's composed with -- spread is already in the
       pre-negated (index->thumb) convention that lands correctly after that
       flip (see franka/utils.py's compute_pinch_orientation docstring for the
       full explanation), and lateral's sign is resolved against a
       *toward-the-wrist* reference for the same reason: since forward here is
       derived from lateral (cross(X, Y) = Z), lateral's sign must already
       encode the pre-negation so forward comes out pre-negated too, without
       breaking right-handedness by negating forward directly.
    """
    idx_pts = hand_point[1:5]
    thm_pts = hand_point[5:9]
    wrist = hand_point[0]

    finger_pts = np.concatenate([idx_pts, thm_pts], axis=0)
    lateral = _fit_plane_normal(finger_pts)
    lateral = lateral / np.linalg.norm(lateral)

    thm_minus_idx = thm_pts - idx_pts
    spread = _fit_dominant_direction(thm_minus_idx)
    if np.dot(spread, thm_minus_idx.mean(axis=0)) < 0:
        spread = -spread  # SVD gives an axis; pin it to point index->thumb
    spread = spread - np.dot(spread, lateral) * lateral  # orthogonalize against lateral
    spread = spread / np.linalg.norm(spread)

    # Resolve the plane normal's sign ambiguity (SVD gives an axis, not a
    # direction) against a coarse, non-degenerate reference: the wrist is
    # never close to the fingers, even mid-pinch, so fingers->wrist is a
    # reliable rough "pre-negated forward" (toward-wrist) direction to check
    # against -- using the away-from-wrist direction here would make `forward`
    # below come out already in final (not pre-negated) form, which
    # robot_base_orientation's Y/Z flip would then invert to point the wrong
    # way (toward the wrist instead of away from it).
    rough_forward = wrist - finger_pts.mean(axis=0)
    if np.dot(lateral, np.cross(spread, rough_forward)) < 0:
        lateral = -lateral

    forward = np.cross(lateral, spread)
    return np.column_stack([lateral, spread, forward])  # X, Y, Z columns
