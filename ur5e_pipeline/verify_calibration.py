#!/usr/bin/env python3
"""
verify_calibration.py

Triangulation-based calibration verifier.

For every frame where the ArUco GridBoard is visible to BOTH cameras:
  1. Detect the board corners in each camera image.
  2. For each corner seen by both cameras, triangulate a 3-D world-frame point
     using both projection matrices  P = K @ ext  (from calib.npy).
  3. Compute the FK ground-truth position of the same corner:
         world_T_ee  (TF)  @  ee_T_board  (board_in_ee.npy)  @  p_board
  4. Report the 3-D error  ||triangulated − FK||  in millimetres.

This directly measures end-to-end calibration accuracy: intrinsics, extrinsics,
and hand-eye offset all contribute to the error.

Annotated images are published per camera so you can view them in rqt_image_view:
  /osx_camera_calibration/cam_1/verify
  /osx_camera_calibration/cam_2/verify

Usage
-----
    rosrun osx_camera_calibration verify_calibration.py
    rosrun osx_camera_calibration verify_calibration.py _config:=/path/calib_config.yaml
"""

import os
import sys
from collections import deque

import numpy as np
import cv2
import yaml

import rospy
import rospkg
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(__file__))
from calibrate_cameras import (
    _detect_raw,
    build_aruco_gridboard,
    transform_stamped_to_mat,
    board_in_ee_cfg_to_mat,
)


# ─── triangulation ────────────────────────────────────────────────────────────

def triangulate_stereo(P1, P2, pts1, pts2):
    """
    Triangulate N point pairs from two cameras.

    P1, P2  : (3, 4) projection matrices  K @ ext
    pts1    : (N, 2) pixel coords in camera 1
    pts2    : (N, 2) pixel coords in camera 2
    Returns : (N, 3) world-frame XYZ
    """
    X4 = cv2.triangulatePoints(
        P1, P2,
        pts1.T.astype(np.float64),
        pts2.T.astype(np.float64),
    )                            # (4, N)
    X4 /= X4[3:4]               # normalise homogeneous
    return X4[:3].T              # (N, 3)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _proj_matrix(calib, cam_key):
    """Return 3×4 projection matrix P = K @ ext."""
    K   = np.asarray(calib[cam_key]["int"], dtype=np.float64)
    ext = np.asarray(calib[cam_key]["ext"], dtype=np.float64)
    return K @ ext[:3]   # (3, 4)


def _draw_corners(vis, pts2d, color, radius=5):
    for p in pts2d.reshape(-1, 2):
        cv2.circle(vis, (int(p[0]), int(p[1])), radius, color, -1)


def _overlay_text(vis, lines, start_y=30, dy=28, scale=0.7, color=(255, 255, 0)):
    for i, txt in enumerate(lines):
        cv2.putText(vis, txt, (10, start_y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("osx_camera_calibration_verify", anonymous=False)

    rospack  = rospkg.RosPack()
    pkg_path = rospack.get_path("osx_camera_calibration")
    default_cfg = os.path.join(pkg_path, "config", "calib_config.yaml")
    cfg_path = rospy.get_param("~config", default_cfg)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_path = cfg.get("output_path", "calib/calib.npy")
    if not os.path.isabs(output_path):
        output_path = os.path.join(pkg_path, output_path)

    if not os.path.exists(output_path):
        rospy.logerr(f"Calibration file not found: {output_path}")
        sys.exit(1)

    calib = np.load(output_path, allow_pickle=True).item()
    rospy.loginfo(f"Loaded calibration: {output_path}")

    # ── board & detection ────────────────────────────────────────────────────
    aruco_dict, board, obj_pts_map = build_aruco_gridboard(cfg)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_params = cv2.aruco.DetectorParameters_create()
    else:
        aruco_params = cv2.aruco.DetectorParameters()

    min_corners = int(cfg.get("min_corners", 16))

    # ── board-to-EE offset ───────────────────────────────────────────────────
    board_in_ee_cfg = cfg.get("board_in_ee")
    if board_in_ee_cfg is not None:
        ee_T_board = board_in_ee_cfg_to_mat(board_in_ee_cfg)
        rospy.loginfo("board_in_ee: from config")
    else:
        board_npy = os.path.join(os.path.dirname(output_path), "board_in_ee.npy")
        if not os.path.exists(board_npy):
            rospy.logerr("board_in_ee not in config and board_in_ee.npy not found.")
            sys.exit(1)
        ee_T_board = np.load(board_npy)
        rospy.loginfo(f"board_in_ee: {board_npy}")

    # ── projection matrices ──────────────────────────────────────────────────
    cameras     = cfg["cameras"]
    cam_keys    = list(cameras.keys())
    if len(cam_keys) < 2:
        rospy.logerr("Need at least 2 cameras in config for triangulation.")
        sys.exit(1)

    cam_key_1, cam_key_2 = cam_keys[0], cam_keys[1]
    P1 = _proj_matrix(calib, cam_key_1)
    P2 = _proj_matrix(calib, cam_key_2)
    rospy.loginfo(f"Triangulating between {cam_key_1} and {cam_key_2}")

    # ── TF ───────────────────────────────────────────────────────────────────
    base_frame = cfg["robot"]["base_frame"]
    ee_frame   = cfg["robot"]["ee_frame"]
    tf_buffer   = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)   # noqa: F841
    rospy.sleep(1.0)

    # ── ROS I/O ──────────────────────────────────────────────────────────────
    bridge   = CvBridge()
    vis_pubs = {}
    for cam_key in cameras:
        topic = f"/osx_camera_calibration/{cam_key}/verify"
        vis_pubs[cam_key] = rospy.Publisher(topic, Image, queue_size=1)
        rospy.loginfo(f"  Publishing: {topic}")

    rospy.loginfo("\nMove the robot so the board is visible to BOTH cameras.")
    rospy.loginfo("Ctrl-C to quit.\n")

    # running error statistics
    errors_mm = deque(maxlen=200)
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():

        # ── FK ground truth ──────────────────────────────────────────────────
        try:
            ts = tf_buffer.lookup_transform(base_frame, ee_frame,
                                            rospy.Time(0), rospy.Duration(0.5))
            world_T_ee = transform_stamped_to_mat(ts)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn_throttle(5.0, "Waiting for TF …")
            rate.sleep()
            continue

        world_T_board = world_T_ee @ ee_T_board

        # ── grab images ──────────────────────────────────────────────────────
        try:
            msg1 = rospy.wait_for_message(
                cameras[cam_key_1]["image_topic"], Image, timeout=2.0)
            msg2 = rospy.wait_for_message(
                cameras[cam_key_2]["image_topic"], Image, timeout=2.0)
        except rospy.ROSException:
            rate.sleep()
            continue

        img1 = bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        img2 = bridge.imgmsg_to_cv2(msg2, desired_encoding="bgr8")
        vis1 = img1.copy()
        vis2 = img2.copy()

        # ── detect markers in each camera ────────────────────────────────────
        corners1, ids1, _ = _detect_raw(img1, aruco_dict, aruco_params)
        corners2, ids2, _ = _detect_raw(img2, aruco_dict, aruco_params)

        status_lines = []
        tri_errors   = []

        if ids1 is not None and len(ids1) > 0:
            cv2.aruco.drawDetectedMarkers(vis1, corners1, ids1)
        if ids2 is not None and len(ids2) > 0:
            cv2.aruco.drawDetectedMarkers(vis2, corners2, ids2)

        if ids1 is None or ids2 is None:
            status_lines.append("Need markers in BOTH cameras")
        else:
            # ── find common marker IDs ────────────────────────────────────
            ids1_flat = ids1.flatten()
            ids2_flat = ids2.flatten()
            common_ids = set(ids1_flat) & set(ids2_flat) & set(obj_pts_map.keys())

            if not common_ids:
                status_lines.append(f"cam1: {len(ids1_flat)} markers  "
                                    f"cam2: {len(ids2_flat)} markers  "
                                    f"common: 0")
            else:
                pts2d_1_all, pts2d_2_all, pts3d_fk_all = [], [], []

                for mid in common_ids:
                    idx1 = int(np.where(ids1_flat == mid)[0][0])
                    idx2 = int(np.where(ids2_flat == mid)[0][0])
                    c1 = corners1[idx1].reshape(4, 2)   # pixel corners in cam1
                    c2 = corners2[idx2].reshape(4, 2)   # pixel corners in cam2
                    p3d_board = obj_pts_map[mid]         # (4, 3) board-frame

                    pts2d_1_all.append(c1)
                    pts2d_2_all.append(c2)

                    # FK ground truth: board-frame → world-frame
                    ones = np.ones((4, 1))
                    p3d_world = (world_T_board @ np.hstack([p3d_board, ones]).T)[:3].T
                    pts3d_fk_all.append(p3d_world)

                pts2d_1 = np.vstack(pts2d_1_all)    # (N, 2)
                pts2d_2 = np.vstack(pts2d_2_all)    # (N, 2)
                pts3d_fk = np.vstack(pts3d_fk_all)  # (N, 3)

                # ── triangulate ──────────────────────────────────────────
                pts3d_tri = triangulate_stereo(P1, P2, pts2d_1, pts2d_2)

                # ── per-corner 3-D error ──────────────────────────────────
                errs_m = np.linalg.norm(pts3d_tri - pts3d_fk, axis=1)
                mean_mm  = float(errs_m.mean()  * 1000)
                std_mm   = float(errs_m.std()   * 1000)
                max_mm   = float(errs_m.max()   * 1000)
                tri_errors.extend(errs_m * 1000)
                errors_mm.extend(errs_m * 1000)

                status_lines = [
                    f"common markers: {len(common_ids)}  "
                    f"corners: {len(errs_m)}",
                    f"3-D error   mean={mean_mm:.1f}mm  "
                    f"std={std_mm:.1f}mm  max={max_mm:.1f}mm",
                ]
                rospy.loginfo_throttle(
                    2.0,
                    f"Triangulation vs FK — "
                    f"mean={mean_mm:.1f}mm  std={std_mm:.1f}mm  "
                    f"max={max_mm:.1f}mm  (N={len(errs_m)})"
                )

                # draw triangulated corners re-projected back to each image
                ext1 = np.asarray(calib[cam_key_1]["ext"], dtype=np.float64)
                ext2 = np.asarray(calib[cam_key_2]["ext"], dtype=np.float64)
                K1   = np.asarray(calib[cam_key_1]["int"], dtype=np.float64)
                K2   = np.asarray(calib[cam_key_2]["int"], dtype=np.float64)
                D1   = np.asarray(calib[cam_key_1]["dist_coeff"], dtype=np.float64)
                D2   = np.asarray(calib[cam_key_2]["dist_coeff"], dtype=np.float64)
                rv1, tv1 = cv2.Rodrigues(ext1[:3, :3])[0], ext1[:3, 3:]
                rv2, tv2 = cv2.Rodrigues(ext2[:3, :3])[0], ext2[:3, 3:]

                # yellow = triangulated position re-projected into each image
                proj_tri_1, _ = cv2.projectPoints(
                    pts3d_tri.astype(np.float64), rv1, tv1, K1, D1)
                proj_tri_2, _ = cv2.projectPoints(
                    pts3d_tri.astype(np.float64), rv2, tv2, K2, D2)
                _draw_corners(vis1, proj_tri_1, color=(0, 220, 220), radius=6)
                _draw_corners(vis2, proj_tri_2, color=(0, 220, 220), radius=6)

                # magenta = FK ground truth re-projected
                proj_fk_1, _ = cv2.projectPoints(
                    pts3d_fk.astype(np.float64), rv1, tv1, K1, D1)
                proj_fk_2, _ = cv2.projectPoints(
                    pts3d_fk.astype(np.float64), rv2, tv2, K2, D2)
                _draw_corners(vis1, proj_fk_1, color=(255, 0, 255), radius=4)
                _draw_corners(vis2, proj_fk_2, color=(255, 0, 255), radius=4)

        # running stats
        if errors_mm:
            a = np.array(errors_mm)
            status_lines.append(
                f"running  n={len(a)}  "
                f"mean={a.mean():.1f}mm  "
                f"std={a.std():.1f}mm  "
                f"max={a.max():.1f}mm"
            )

        legend = ("CYAN=triangulated  MAGENTA=FK ground-truth  "
                  "GREEN(outline)=detected")
        for vis in (vis1, vis2):
            _overlay_text(vis, status_lines)
            cv2.putText(vis, legend, (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)

        # ── publish ──────────────────────────────────────────────────────────
        for cam_key, vis, msg in (
            (cam_key_1, vis1, msg1),
            (cam_key_2, vis2, msg2),
        ):
            if vis_pubs[cam_key].get_num_connections() == 0:
                continue
            out = bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            out.header = msg.header
            try:
                vis_pubs[cam_key].publish(out)
            except rospy.ROSException:
                pass

        rate.sleep()


if __name__ == "__main__":
    main()
