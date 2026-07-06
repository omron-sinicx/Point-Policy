#!/usr/bin/env python3
"""
calibrate_cameras.py

Calibrates fixed cameras to the UR5e robot base frame using an ArUco GridBoard
mounted on the robot end-effector.

Also automatically determines the board-to-EE rigid offset (board_in_ee)
through calibration — no manual measurement required.

Procedure
---------
Phase 1  CAPTURE
  Freedrive the robot so the ArUco GridBoard is visible.  Press Enter to capture
  the current pose.  Repeat for ~15-20 diverse poses, then press q.
  At each pose the script records:
    - world_T_ee    : EE pose in robot base frame (from TF / robot FK)
    - cam_T_board_i : board pose in each camera frame (solvePnP, board-local coords)
    - raw detected 2-D corners and matching 3-D board-frame coords (for refinement)

Phase 2  HAND-EYE SOLVE (per camera, skipped if board_in_ee given in config)
  Using the constraint  cam_T_board_i = cam_T_world * world_T_ee_i * ee_T_board
  and taking relative motions between pose pairs gives AX = XB with X = cam_T_world.
  Solved with cv2.calibrateHandEye (inputs swapped for eye-to-hand setup).
  ee_T_board is recovered algebraically per pose and averaged.

Phase 3  REFINEMENT (per camera)
  With ee_T_board known, board corners are expressed in robot base frame via FK.
  A single cv2.solvePnP over ALL poses gives the final refined cam_T_world.

Output
------
  calib/calib.npy      : { "cam_1": {"int":(3,3), "ext":(4,4), "dist_coeff":(1,5)}, … }
  calib/board_in_ee.npy: ee_T_board (4×4), written only when auto-calibrated

Usage
-----
    rosrun osx_camera_calibration calibrate_cameras.py
    rosrun osx_camera_calibration calibrate_cameras.py _config:=/abs/path/calib_config.yaml
"""

import os
import sys
import math

import numpy as np
import cv2
import yaml

import rospy
import rospkg
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge


# ─── math helpers ─────────────────────────────────────────────────────────────

def quaternion_to_matrix(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def matrix_to_quaternion(R):
    """3×3 rotation matrix → (qx, qy, qz, qw)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x = 0.25/s, (R[2,1]-R[1,2])*s
        y, z  = (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x = (R[2,1]-R[1,2])/s, 0.25*s
        y, z  = (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s
        y, z  = 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s
        y, z  = (R[1,2]+R[2,1])/s, 0.25*s
    return x, y, z, w


def transform_stamped_to_mat(ts: TransformStamped) -> np.ndarray:
    t = ts.transform.translation
    r = ts.transform.rotation
    mat = np.eye(4)
    mat[:3, :3] = quaternion_to_matrix(r.x, r.y, r.z, r.w)
    mat[:3,  3] = [t.x, t.y, t.z]
    return mat


def board_in_ee_cfg_to_mat(b: dict) -> np.ndarray:
    """board_in_ee config dict (x/y/z/qx/qy/qz/qw) → 4×4 matrix."""
    mat = np.eye(4)
    mat[:3, :3] = quaternion_to_matrix(b["qx"], b["qy"], b["qz"], b["qw"])
    mat[:3,  3] = [b["x"], b["y"], b["z"]]
    return mat


def camera_info_to_KD(msg: CameraInfo):
    K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
    d = list(msg.D)
    while len(d) < 5:
        d.append(0.0)
    D = np.array(d[:5], dtype=np.float64).reshape(1, 5)
    return K, D


def average_rotations(Rs):
    R_sum = sum(Rs)
    U, _, Vt = np.linalg.svd(R_sum)
    return U @ Vt


def reproj_rmse(pts3d, pts2d, K, D, rvec, tvec):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.linalg.norm(pts2d - proj, axis=1) ** 2)))


# ─── ArUco GridBoard helpers ──────────────────────────────────────────────────

ARUCO_DICT_MAP = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":  cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50":  cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


def build_aruco_gridboard(cfg: dict):
    """Build (aruco_dict, board, obj_pts_map) from the aruco_grid config section.

    obj_pts_map: dict  marker_id → (4, 3) float64 array of 3-D corners in board
    frame (x right, y down, z out of board).  Row-major layout, IDs start at 0.
    """
    c = cfg["aruco_grid"]
    markers_x  = int(c["markers_x"])
    markers_y  = int(c["markers_y"])
    ml = float(c["marker_length"])
    ms = float(c["marker_separation"])
    dict_name  = c["aruco_dict"]

    if dict_name not in ARUCO_DICT_MAP:
        raise ValueError(f"Unknown aruco_dict '{dict_name}'. Choose from {list(ARUCO_DICT_MAP)}")

    dict_id = ARUCO_DICT_MAP[dict_name]
    if hasattr(cv2.aruco, "Dictionary_get"):
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        board = cv2.aruco.GridBoard_create(
            markersX=markers_x, markersY=markers_y,
            markerLength=ml, markerSeparation=ms,
            dictionary=aruco_dict,
        )
    else:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.GridBoard(
            (markers_x, markers_y), ml, ms, aruco_dict,
        )

    # Pre-compute 3-D corners for every marker (board-local frame)
    step = ml + ms
    obj_pts_map = {}
    for idx in range(markers_x * markers_y):
        row = idx // markers_x
        col = idx % markers_x
        x0, y0 = col * step, row * step
        obj_pts_map[idx] = np.array([
            [x0,      y0,      0],
            [x0 + ml, y0,      0],
            [x0 + ml, y0 + ml, 0],
            [x0,      y0 + ml, 0],
        ], dtype=np.float64)

    return aruco_dict, board, obj_pts_map


def _detect_raw(img_bgr, aruco_dict, aruco_params):
    """API-agnostic wrapper around ArUco marker detection."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        return detector.detectMarkers(img_bgr)
    return cv2.aruco.detectMarkers(img_bgr, aruco_dict, parameters=aruco_params)


def detect_aruco_grid(img_bgr, aruco_dict, aruco_params, obj_pts_map, min_corners: int):
    """
    Detect ArUco markers and return matched 2-D / 3-D correspondences.

    Returns
    -------
    pts2d : (N, 2) float32  detected image corners  (4 per marker)
    pts3d : (N, 3) float64  matching board-frame 3-D coords
    or (None, None) if fewer than min_corners corner points were detected.
    """
    corners, ids, _ = _detect_raw(img_bgr, aruco_dict, aruco_params)

    if ids is None or len(ids) == 0:
        return None, None

    pts2d_list, pts3d_list = [], []
    for corner_group, mid in zip(corners, ids.flatten()):
        if mid in obj_pts_map:
            pts2d_list.append(corner_group.reshape(4, 2))
            pts3d_list.append(obj_pts_map[mid])

    total = sum(len(p) for p in pts2d_list)
    if total < min_corners:
        return None, None

    return (np.vstack(pts2d_list).astype(np.float32),
            np.vstack(pts3d_list).astype(np.float64))


def estimate_cam_T_board(pts2d, pts3d_local, K, D):
    """solvePnP with board-local 3-D coords.  Returns cam_T_board (4×4) or None."""
    ok, rvec, tvec = cv2.solvePnP(
        pts3d_local.astype(np.float64), pts2d.astype(np.float64),
        K, D, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        pts3d_local.astype(np.float64), pts2d.astype(np.float64),
        K, D, rvec, tvec,
    )
    R, _ = cv2.Rodrigues(rvec)
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3,  3] = tvec.ravel()
    return mat


# ─── hand-eye solve ───────────────────────────────────────────────────────────

def solve_hand_eye(cam_T_boards, world_T_ees):
    """
    Solve for ee_T_board and cam_T_world from N pose observations.

    Constraint:  cam_T_board_i = cam_T_world * world_T_ee_i * ee_T_board

    Taking relative motions between pose pairs cancels cam_T_world and yields
    AX = XB with X = ee_T_board:
      A_ij = world_T_ee_i^{-1} * world_T_ee_j   → pass world_T_ee as gripper2base
      B_ij = cam_T_board_i^{-1} * cam_T_board_j  → pass board_T_cam as target2cam
      X    = ee_T_board                           ← calibrateHandEye output

    cam_T_world is then recovered per-pose from the constraint:
      cam_T_world = cam_T_board_i * ee_T_board^{-1} * world_T_ee_i^{-1}
    and averaged.

    Quality metric: std of per-pose cam_T_world translations.  Should be < 5 mm
    for a good calibration with diverse rotational poses.

    Returns cam_T_world (4×4) and ee_T_board (4×4).
    """
    # board_T_cam = inv(cam_T_board) — used as "target2cam"
    board_T_cams = [np.linalg.inv(m) for m in cam_T_boards]

    R_g2b = [m[:3, :3] for m in world_T_ees]
    t_g2b = [m[:3, 3:4] for m in world_T_ees]
    R_t2c = [m[:3, :3] for m in board_T_cams]
    t_t2c = [m[:3, 3:4] for m in board_T_cams]

    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    best, best_std = None, np.inf
    for name, method in methods.items():
        try:
            R_X, t_X = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
        except cv2.error:
            continue

        ee_T_board_cand = np.eye(4)
        ee_T_board_cand[:3, :3] = R_X
        ee_T_board_cand[:3,  3] = t_X.ravel()
        board_T_ee_cand = np.linalg.inv(ee_T_board_cand)

        # Recover cam_T_world per pose; if the solve is good they should agree
        cam_T_worlds = []
        for cTb, wTe in zip(cam_T_boards, world_T_ees):
            cTw = cTb @ board_T_ee_cand @ np.linalg.inv(wTe)
            cam_T_worlds.append(cTw)

        ts = np.array([m[:3, 3] for m in cam_T_worlds])
        std = float(np.std(ts, axis=0).mean())
        if std < best_std:
            best_std = std
            best = (ee_T_board_cand, cam_T_worlds, name, std)

    if best is None:
        raise RuntimeError("All calibrateHandEye methods failed.")

    ee_T_board, cam_T_worlds, best_name, std = best

    # Average cam_T_world across poses
    Rs = [m[:3, :3] for m in cam_T_worlds]
    ts = [m[:3, 3]  for m in cam_T_worlds]
    cam_T_world = np.eye(4)
    cam_T_world[:3, :3] = average_rotations(Rs)
    cam_T_world[:3,  3] = np.mean(ts, axis=0)

    rospy.loginfo(f"  Best solver: {best_name}  "
                  f"(cam_T_world translation std = {std*1000:.2f} mm)")
    return cam_T_world, ee_T_board


# ─── live ArUco visualizer ────────────────────────────────────────────────────

class ArucoVisualizer:
    """
    Subscribes to each camera's image topic, draws detected ArUco markers, and
    re-publishes to /osx_camera_calibration/<cam_key>/image_detection.
    View in RViz or `rqt_image_view`.
    """
    def __init__(self, cameras, aruco_dict, aruco_params, bridge):
        self._aruco_dict   = aruco_dict
        self._aruco_params = aruco_params
        self._bridge       = bridge
        self._pubs = {}
        for cam_key, cam_cfg in cameras.items():
            topic = f"/osx_camera_calibration/{cam_key}/image_detection"
            self._pubs[cam_key] = rospy.Publisher(topic, Image, queue_size=1)
            rospy.Subscriber(cam_cfg["image_topic"], Image,
                             lambda msg, k=cam_key: self._cb(msg, k), queue_size=1)
            rospy.loginfo(f"  Visualizer: {cam_cfg['image_topic']} → {topic}")

    def _cb(self, msg, cam_key):
        if self._pubs[cam_key].get_num_connections() == 0:
            return
        try:
            img_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        vis = img_bgr.copy()
        corners, ids, _ = _detect_raw(vis, self._aruco_dict, self._aruco_params)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.putText(vis, f"{len(ids)} markers", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No markers", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        if rospy.is_shutdown():
            return
        out = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out.header = msg.header
        try:
            self._pubs[cam_key].publish(out)
        except rospy.ROSException:
            pass


# ─── joint state monitor ──────────────────────────────────────────────────────

class JointStateMonitor:
    def __init__(self, topic: str):
        self._msg = None
        rospy.Subscriber(topic, JointState, lambda m: setattr(self, '_msg', m), queue_size=1)

    def ready(self) -> bool:
        return self._msg is not None

    def positions_str(self) -> str:
        if self._msg is None:
            return "(waiting …)"
        return "[" + ", ".join(f"{p:7.4f}" for p in self._msg.position) + "]"


# ─── capture one pose ─────────────────────────────────────────────────────────

def capture_pose(idx, tf_buffer, base_frame, ee_frame,
                 cameras, bridge, aruco_dict, aruco_params, obj_pts_map,
                 cam_K, cam_D, min_corners, observations):
    """
    Records one calibration pose.  Appends to observations[cam_key], where each
    entry is a dict: world_T_ee, cam_T_board, pts2d, pts3d_local.
    """
    try:
        ts = tf_buffer.lookup_transform(base_frame, ee_frame,
                                        rospy.Time(0), rospy.Duration(2.0))
    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn(f"  TF lookup failed: {e} — skipping.")
        return

    world_T_ee = transform_stamped_to_mat(ts)

    for cam_key, cam_cfg in cameras.items():
        try:
            img_msg = rospy.wait_for_message(cam_cfg["image_topic"], Image, timeout=5.0)
        except rospy.ROSException:
            rospy.logwarn(f"  {cam_key}: image timeout, skipping.")
            continue

        img_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        pts2d, pts3d_local = detect_aruco_grid(
            img_bgr, aruco_dict, aruco_params, obj_pts_map, min_corners)

        if pts2d is None:
            rospy.logwarn(f"  {cam_key}: not enough corners detected — skipping.")
            continue

        K = cam_K[cam_key]
        D = cam_D[cam_key]
        cam_T_board = estimate_cam_T_board(pts2d, pts3d_local, K, D)
        if cam_T_board is None:
            rospy.logwarn(f"  {cam_key}: solvePnP (board-local) failed — skipping.")
            continue

        observations[cam_key].append({
            "world_T_ee":   world_T_ee,
            "cam_T_board":  cam_T_board,
            "pts2d":        pts2d,
            "pts3d_local":  pts3d_local,
        })
        n_markers = len(pts2d) // 4
        rospy.loginfo(f"  {cam_key}: {n_markers} markers / {len(pts2d)} corners  "
                      f"(total {len(observations[cam_key])} poses)")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("osx_camera_calibration", anonymous=False)

    rospack  = rospkg.RosPack()
    pkg_path = rospack.get_path("osx_camera_calibration")
    default_cfg = os.path.join(pkg_path, "config", "calib_config.yaml")
    cfg_path = rospy.get_param("~config", default_cfg)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    rospy.loginfo(f"Loaded config: {cfg_path}")

    cameras     = cfg["cameras"]
    robot_cfg   = cfg["robot"]
    base_frame  = robot_cfg["base_frame"]
    ee_frame    = robot_cfg["ee_frame"]
    joint_topic = robot_cfg.get("joint_states_topic", "/joint_states")
    min_corners = int(cfg.get("min_corners", 16))   # 4 corners × 4 markers minimum
    output_path = cfg.get("output_path", "calib/calib.npy")

    aruco_dict, board, obj_pts_map = build_aruco_gridboard(cfg)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_params = cv2.aruco.DetectorParameters_create()
    else:
        aruco_params = cv2.aruco.DetectorParameters()
    bridge = CvBridge()

    # ── camera intrinsics ────────────────────────────────────────────────────
    rospy.loginfo("Waiting for camera_info …")
    cam_K, cam_D = {}, {}
    for cam_key, cam_cfg in cameras.items():
        msg = rospy.wait_for_message(cam_cfg["info_topic"], CameraInfo, timeout=10.0)
        K, D = camera_info_to_KD(msg)
        cam_K[cam_key] = K
        cam_D[cam_key] = D
        rospy.loginfo(f"  {cam_key}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                      f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # ── TF ───────────────────────────────────────────────────────────────────
    tf_buffer   = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)   # noqa: F841
    rospy.sleep(1.0)

    # ── joint state monitor ───────────────────────────────────────────────────
    js_monitor = JointStateMonitor(joint_topic)
    rospy.loginfo(f"Monitoring joints on {joint_topic} …")
    while not rospy.is_shutdown() and not js_monitor.ready():
        rospy.sleep(0.2)

    # ── live detection visualizer ────────────────────────────────────────────
    rospy.loginfo("Starting ArUco visualizer …")
    ArucoVisualizer(cameras, aruco_dict, aruco_params, bridge)

    # ── Phase 1: capture ─────────────────────────────────────────────────────
    observations = {k: [] for k in cameras}
    capture_kwargs = dict(
        tf_buffer=tf_buffer, base_frame=base_frame, ee_frame=ee_frame,
        cameras=cameras, bridge=bridge,
        aruco_dict=aruco_dict, aruco_params=aruco_params, obj_pts_map=obj_pts_map,
        cam_K=cam_K, cam_D=cam_D,
        min_corners=min_corners, observations=observations,
    )

    c = cfg["aruco_grid"]
    rospy.loginfo(f"\nArUco GridBoard: {c['markers_x']}×{c['markers_y']} markers  "
                  f"dict={c['aruco_dict']}  "
                  f"markerLen={c['marker_length']*1000:.0f}mm  "
                  f"sep={c['marker_separation']*1000:.0f}mm")
    print("\nFreedrive the robot so the ArUco GridBoard is visible to the cameras.")
    print("Aim for 15-20 diverse poses (vary position AND wrist orientation).")
    print("  Enter      → capture current pose")
    print("  q + Enter  → finish and calibrate\n")

    capture_idx = 0
    while not rospy.is_shutdown():
        print(f"joints: {js_monitor.positions_str()}")
        try:
            cmd = input(f"[{capture_idx+1}] Enter to capture, q to finish > ")
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.strip().lower() == "q":
            break
        rospy.loginfo(f"─── Capturing pose {capture_idx+1} ───")
        capture_pose(capture_idx + 1, **capture_kwargs)
        capture_idx += 1

    rospy.loginfo(f"\nCollection done — {capture_idx} poses attempted.")

    # ── Phase 2: board-to-EE offset ──────────────────────────────────────────
    board_in_ee_cfg = cfg.get("board_in_ee")
    cam_T_world_init = {}

    if board_in_ee_cfg is not None:
        rospy.loginfo("\n=== Phase 2: Using board_in_ee from config (skipping hand-eye) ===")
        ee_T_board = board_in_ee_cfg_to_mat(board_in_ee_cfg)
        tx, ty, tz = ee_T_board[:3, 3]
        qx, qy, qz, qw = matrix_to_quaternion(ee_T_board[:3, :3])
        rospy.loginfo(f"  ee_T_board  x={tx:.4f} y={ty:.4f} z={tz:.4f}  "
                      f"qx={qx:.4f} qy={qy:.4f} qz={qz:.4f} qw={qw:.4f}")
        save_board_in_ee = False
    else:
        rospy.loginfo("\n=== Phase 2: Hand-eye calibration (auto board_in_ee) ===")
        ee_T_board_per_cam = {}
        for cam_key, obs_list in observations.items():
            n = len(obs_list)
            if n < 3:
                rospy.logerr(f"{cam_key}: only {n} valid poses (need ≥3).")
                continue
            rospy.loginfo(f"\n{cam_key}: hand-eye solve with {n} poses …")
            try:
                cTw, eTb = solve_hand_eye(
                    [o["cam_T_board"] for o in obs_list],
                    [o["world_T_ee"]  for o in obs_list],
                )
            except RuntimeError as e:
                rospy.logerr(f"{cam_key}: {e}")
                continue
            cam_T_world_init[cam_key]   = cTw
            ee_T_board_per_cam[cam_key] = eTb

        if not ee_T_board_per_cam:
            rospy.logerr("Hand-eye failed for all cameras. Exiting.")
            sys.exit(1)

        rospy.loginfo("\n=== Phase 3: Board-to-EE offset (averaged across cameras) ===")
        ee_T_board = np.eye(4)
        ee_T_board[:3, :3] = average_rotations([m[:3, :3] for m in ee_T_board_per_cam.values()])
        ee_T_board[:3,  3] = np.mean([m[:3, 3] for m in ee_T_board_per_cam.values()], axis=0)

        tx, ty, tz = ee_T_board[:3, 3]
        qx, qy, qz, qw = matrix_to_quaternion(ee_T_board[:3, :3])
        rospy.loginfo(f"  ee_T_board  x={tx:.4f} y={ty:.4f} z={tz:.4f}")
        rospy.loginfo(f"              qx={qx:.4f} qy={qy:.4f} qz={qz:.4f} qw={qw:.4f}")
        rospy.loginfo("  Tip: paste into calib_config.yaml as board_in_ee to skip")
        rospy.loginfo("  hand-eye on future runs.")
        save_board_in_ee = True

    # ── Phase 4: refined solvePnP per camera ─────────────────────────────────
    rospy.loginfo("\n=== Phase 4: Refined solvePnP ===")
    calib_dict = {}

    for cam_key, obs_list in observations.items():
        if not obs_list:
            rospy.logerr(f"{cam_key}: no valid observations. Skipping.")
            continue

        all3d, all2d = [], []
        for obs in obs_list:
            world_T_board = obs["world_T_ee"] @ ee_T_board
            pts3d_local   = obs["pts3d_local"]
            ones = np.ones((len(pts3d_local), 1))
            corners_world = (world_T_board @ np.hstack([pts3d_local, ones]).T)[:3].T
            all3d.append(corners_world)
            all2d.append(obs["pts2d"])

        pts3d = np.vstack(all3d).astype(np.float64)
        pts2d = np.vstack(all2d).astype(np.float64)
        K = cam_K[cam_key]
        D = cam_D[cam_key]

        use_guess = cam_key in cam_T_world_init
        if use_guess:
            R_i = cam_T_world_init[cam_key][:3, :3]
            t_i = cam_T_world_init[cam_key][:3, 3:4]
            rvec_i, _ = cv2.Rodrigues(R_i)
            ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, D,
                                           rvec=rvec_i, tvec=t_i,
                                           useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, D,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            rospy.logerr(f"{cam_key}: solvePnP failed.")
            continue
        rvec, tvec = cv2.solvePnPRefineLM(pts3d, pts2d, K, D, rvec, tvec)

        R, _ = cv2.Rodrigues(rvec)
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = R
        ext[:3,  3] = tvec.ravel()

        rmse = reproj_rmse(pts3d, pts2d, K, D, rvec, tvec)
        rospy.loginfo(f"\n{cam_key}: reprojection RMSE = {rmse:.3f} px  "
                      f"({len(pts3d)} corners from {len(obs_list)} poses)")
        rospy.loginfo(f"  ext (world→cam):\n{np.round(ext, 4)}")

        calib_dict[cam_key] = {"int": K, "ext": ext, "dist_coeff": D}

    if not calib_dict:
        rospy.logerr("No cameras calibrated. Exiting without saving.")
        sys.exit(1)

    # ── save ─────────────────────────────────────────────────────────────────
    if not os.path.isabs(output_path):
        output_path = os.path.join(pkg_path, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, calib_dict)
    rospy.loginfo(f"\nSaved calib.npy → {output_path}")

    if save_board_in_ee:
        board_path = os.path.join(os.path.dirname(output_path), "board_in_ee.npy")
        np.save(board_path, ee_T_board)
        rospy.loginfo(f"Saved board_in_ee.npy → {board_path}")

    loaded = np.load(output_path, allow_pickle=True).item()
    for k in loaded:
        assert loaded[k]["int"].shape == (3, 3)
        assert loaded[k]["ext"].shape == (4, 4)
        assert loaded[k]["dist_coeff"].shape == (1, 5)
    rospy.loginfo("Load-back check passed.")


if __name__ == "__main__":
    main()
