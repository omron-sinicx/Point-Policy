"""
UR5e end-effector geometry for Point-Policy.

This file defines the 9 keypoints that represent the robot's gripper:
  - 1 center point (TCP midpoint between fingers)
  - 8 body points spread along the gripper structure

These are analogous to point_policy/robot_utils/franka/gripper_points.py,
but parameterized for the UR5e flange + your custom gripper.

=============================================================================
HOW TO MEASURE AND FILL IN YOUR CUSTOM GRIPPER VALUES
=============================================================================

Reference frame: UR5e tool0 flange
  - Z axis points AWAY from the robot (toward the work surface)
  - Y axis separates the two fingers
  - X axis is the remaining orthogonal axis

1. Measure Tshift (flange → TCP center):
   - Hold a ruler from the flange face to the midpoint between the fingertips.
   - If the TCP is 10 cm below the flange face along Z, set Tshift[2,3] = -0.10.
   - If there is also a Y or X offset (uncommon), set those too.

2. Measure extrapoints (8 body points, relative to TCP):
   Current layout (same as Franka; adjust Y offsets to your finger spacing):

     Points 0-1: fingertips (+Y and -Y at z = finger_length from TCP center)
     Points 2-4: mid-finger cross-section (+Y, 0, -Y at z = finger_length/2)
     Points 5-7: near-base cross-section (+Y, 0, -Y at z = finger_length/4)

   Key measurements needed:
     - finger_length:  distance from TCP center to fingertip tip (meters)
     - finger_spread:  maximum Y-distance from center to each finger (meters)
     - mid_spread:     finger spacing at mid-finger height (often ≈ finger_spread)
     - base_spread:    finger spacing near the palm (often slightly wider)

   Example defaults below assume:
     finger_length = 0.08 m
     finger_spread = 0.04 m  (Franka-like; update for your gripper)

3. Measure robot_base_orientation:
   This is the rotation applied at frame 0 to align the MediaPipe hand-tracking
   coordinate frame with the UR5e base frame.
   Start with R.from_rotvec([pi, 0, 0]) (same as Franka) and adjust if the
   predicted robot tracks look mirrored or rotated in the visualization step.

=============================================================================
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Tshift: 4×4 rigid transform — UR5e tool0 flange → TCP center
#
# UPDATE: set the Z offset to match your gripper's flange-to-TCP distance.
# Positive Z in the flange frame points outward (away from robot).
# ---------------------------------------------------------------------------
_flange_to_tcp_z = -0.155  # meters — MEASURE YOUR GRIPPER and set this value

Tshift = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, _flange_to_tcp_z],
    [0, 0, 0, 1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# extrapoints: 8 × 4×4 transforms — body keypoints relative to TCP center
#
# UPDATE: set finger_length, finger_spread, etc. to match your gripper.
# ---------------------------------------------------------------------------
# finger_length must be LARGER than abs(_flange_to_tcp_z) so that the fingertip
# extrapoints extend past TCP level back toward the fingers.
# Formula: finger_length = abs(_flange_to_tcp_z) + overshoot
# where overshoot = how far the fingertips extend below the TCP center (meters).
# Franka example: Tshift=0.127m, finger_length=0.16m → 0.033m overshoot.
# Update _finger_overshoot to match your UR5e gripper's actual fingertip extension.
_finger_overshoot = 0.00   # meters — fingertips extend this far past TCP center
_finger_length = abs(_flange_to_tcp_z) + _finger_overshoot  # = 0.185m

_finger_spread = 0.04  # meters — Y distance from center to each fingertip
_mid_spread = 0.04  # meters — Y spread at mid-finger height
_base_spread = 0.04  # meters — Y spread near palm / base of fingers


def _T(y, z):
    """4×4 translation-only transform (pure Y, Z offset from TCP)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=np.float64)


extrapoints = [
    # Fingertips (points 0-1): left/right fingertips at full extension
    _T(+_finger_spread,  _finger_length),
    _T(-_finger_spread,  _finger_length),

    # Mid-finger cross-section (points 2-4)
    _T(0,               _finger_length / 2),
    _T(+_mid_spread,     _finger_length / 2),
    _T(-_mid_spread,     _finger_length / 2),

    # Near-base cross-section (points 5-7)
    _T(0,               _finger_length / 4),
    _T(+_base_spread,    _finger_length / 4),
    _T(-_base_spread,    _finger_length / 4),
]


# ---------------------------------------------------------------------------
# robot_base_orientation: 3×3 rotation matrix applied at the first demo frame
# to align MediaPipe hand coordinates with the UR5e base frame.
#
# Start with R.from_rotvec([pi, 0, 0]) (flip around X axis, same as Franka).
# If tracked robot points look upside-down or mirrored in the visualization,
# adjust this rotation until the overlay looks correct.
# ---------------------------------------------------------------------------
robot_base_orientation = R.from_rotvec([np.pi, 0, 0]).as_matrix()
