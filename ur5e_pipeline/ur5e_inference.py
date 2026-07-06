"""
Real-time Point-Policy inference on a UR5e arm.

Streams from 2 RealSense D435 cameras, tracks keypoints with CoTracker online,
feeds observations to the trained policy, and sends predicted end-effector poses
to the UR5e via ur_rtde.

Prerequisites:
    pip install pyrealsense2 ur-rtde torch
    (Point-Policy conda env must be active)

Usage:
    # Dry run (no real robot — shows predicted actions only):
    python ur5e_pipeline/ur5e_inference.py \
        --checkpoint checkpoints/pick_cup/snapshot.pt \
        --calib_path calib/calib.npy \
        --cam_serials <serial1> <serial2> \
        --data_pkl /data/expert_demos/franka_env/pick_cup.pkl \
        --dry_run

    # Real robot:
    python ur5e_pipeline/ur5e_inference.py \
        --checkpoint checkpoints/pick_cup/snapshot.pt \
        --calib_path calib/calib.npy \
        --cam_serials <serial1> <serial2> \
        --data_pkl /data/expert_demos/franka_env/pick_cup.pkl \
        --ur5e_ip 192.168.1.100

Episode controls (keyboard):
    ENTER  — start a new episode
    q      — quit
"""

import sys
import os
import argparse
import pickle as pkl
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths: allow importing from point_policy and co-tracker
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent  # ur5e_pipeline/ is one level inside the repo root
sys.path.insert(0, str(REPO_ROOT / "point_policy"))
sys.path.insert(0, str(REPO_ROOT / "co-tracker"))
sys.path.insert(0, str(REPO_ROOT / "point_policy" / "robot_utils" / "franka"))

from utils import triangulate_points, rigid_transform_3D   # franka/utils.py
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Point-Policy inference on UR5e")
    p.add_argument("--checkpoint", required=True, help="Path to snapshot.pt")
    p.add_argument("--calib_path", required=True, help="Path to calib.npy")
    p.add_argument("--cam_serials", nargs=2, required=True, metavar=("S1", "S2"))
    p.add_argument("--data_pkl", required=True,
                   help="Path to expert_demos pkl (for norm_stats and base_robot_points)")
    p.add_argument("--ur5e_ip", default=None, help="UR5e IP address (omit for --dry_run)")
    p.add_argument("--dry_run", action="store_true", help="Print actions without moving robot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--move_speed", type=float, default=0.05, help="UR5e moveL speed (m/s)")
    p.add_argument("--move_accel", type=float, default=0.1, help="UR5e moveL acceleration (m/s^2)")
    p.add_argument("--history_len", type=int, default=10, help="Observation history length")
    p.add_argument("--num_robot_points", type=int, default=9)
    p.add_argument("--point_dim", type=int, default=3, help="2 or 3 (must match training config)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# RealSense camera
# ---------------------------------------------------------------------------
import pyrealsense2 as rs

class RealSenseCamera:
    def __init__(self, serial, width=640, height=480, fps=30):
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        # Intrinsics from SDK (not used here — we load from calib.npy)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data()).copy()  # BGR uint8
        depth = np.asanyarray(aligned.get_depth_frame().get_data()).copy()  # uint16 mm
        return color, depth

    def stop(self):
        self.pipeline.stop()


# ---------------------------------------------------------------------------
# UR5e controller (wraps ur_rtde)
# ---------------------------------------------------------------------------
class UR5eController:
    def __init__(self, ip):
        import rtde_control
        import rtde_receive
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        print(f"Connected to UR5e at {ip}")

    def get_tcp_pose(self):
        """Return current TCP pose as [x, y, z, rx, ry, rz] (rotation vector)."""
        return np.array(self.rtde_r.getActualTCPPose())

    def move_to(self, position, quaternion, speed, accel):
        """
        Move TCP to target pose.
        position:   (3,) numpy array in meters
        quaternion: (4,) numpy array [qx, qy, qz, qw]
        """
        rotvec = R.from_quat(quaternion).as_rotvec()
        pose = list(position) + list(rotvec)
        self.rtde_c.moveL(pose, speed, accel, asynchronous=True)

    def stop(self):
        self.rtde_c.stopL(0.5)
        self.rtde_c.disconnect()


# ---------------------------------------------------------------------------
# CoTracker online helper
# ---------------------------------------------------------------------------
def load_cotracker(checkpoint_path, device):
    from cotracker.predictor import CoTrackerOnlinePredictor
    tracker = CoTrackerOnlinePredictor(checkpoint=checkpoint_path)
    tracker = tracker.to(device)
    tracker.eval()
    return tracker


def build_cotracker_query(points_2d):
    """
    points_2d: (N, 2) array of [x, y] pixel coordinates at frame 0.
    Returns: (1, N, 3) tensor of [0, x, y] (CoTracker online query format).
    """
    N = len(points_2d)
    queries = np.zeros((N, 3), dtype=np.float32)
    queries[:, 0] = 0        # frame index
    queries[:, 1:] = points_2d
    return torch.from_numpy(queries).unsqueeze(0)  # (1, N, 3)


# ---------------------------------------------------------------------------
# Keypoint selection (interactive)
# ---------------------------------------------------------------------------
_selected_points = []

def _mouse_callback(event, x, y, flags, param):
    global _selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        _selected_points.append([x, y])
        print(f"  Selected point {len(_selected_points)}: ({x}, {y})")


def select_initial_keypoints(frame_bgr, num_points, window_name="Select keypoints"):
    """
    Ask user to click num_points keypoints on the frame.
    Returns: (num_points, 2) array of [x, y] pixel coords.
    """
    global _selected_points
    _selected_points = []
    disp = frame_bgr.copy()
    print(f"\nClick {num_points} keypoints on the image.")
    print("  Click on gripper/hand position(s) first, then object points.")
    print("  Press SPACE or ENTER when done.")
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)

    while True:
        tmp = disp.copy()
        for pt in _selected_points:
            cv2.circle(tmp, tuple(pt), 5, (0, 255, 0), -1)
        cv2.putText(tmp, f"{len(_selected_points)}/{num_points} points selected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, tmp)
        key = cv2.waitKey(30)
        if key in (13, 32) and len(_selected_points) >= num_points:
            break

    cv2.destroyWindow(window_name)
    return np.array(_selected_points[:num_points], dtype=np.float32)


# ---------------------------------------------------------------------------
# Action conversion: predicted robot keypoints → UR5e TCP pose
# ---------------------------------------------------------------------------
class ActionConverter:
    """
    Converts the policy's predicted 9 robot keypoints to a UR5e TCP pose.
    Mirrors the logic in point_policy/suite/point_policy.py:point2action().
    """

    def __init__(self, calib_data, num_robot_points, point_dim, Tshift):
        self.calib = calib_data
        self.num_robot_points = num_robot_points
        self.point_dim = point_dim
        self.Tshift = Tshift                   # 4×4 flange→TCP matrix from gripper_points.py
        self.base_robot_points = None          # set at episode start from first-frame keypoints
        self.robot_base_orientation = None     # set at episode start
        self.prev_gripper_state = -1

        # Projection matrices (3×4) for triangulation
        self.P = []
        for cam_name in ["cam_1", "cam_2"]:
            K = calib_data[cam_name]["int"]
            ext = calib_data[cam_name]["ext"]
            K4 = np.concatenate([K, np.zeros((3, 1))], axis=1)
            self.P.append(K4 @ ext)

    def set_base_points(self, points3d_9, base_orientation):
        """Call at the start of each episode with the first-frame 3D robot points."""
        self.base_robot_points = points3d_9.copy()
        self.robot_base_orientation = base_orientation.copy()
        self.prev_gripper_state = -1

    def convert(self, action_dict):
        """
        action_dict: return value from agent.act() — contains:
            "future_tracks_pixels1": (num_robot_points, point_dim)
            "future_tracks_pixels2": (num_robot_points, point_dim)
            "gripper":               (1, 1) or (1,)

        Returns: dict with
            "flange_pos":  (3,) flange target position, world frame — what to
                           actually command the robot with (robot.move_to).
            "flange_quat": (4,) flange target orientation, [qx, qy, qz, qw].
            "tcp_pos":     (3,) TCP / end-effector position, world frame —
                           the flange pose shifted forward by Tshift, i.e.
                           where the gripper's grasp point actually ends up.
            "tcp_quat":    (4,) TCP orientation, [qx, qy, qz, qw]. Identical
                           to flange_quat since Tshift is a pure translation.
            "gripper":     float, -1.0 (open) or 1.0 (closed).
        """
        pts = []
        for pkey in ["pixels1", "pixels2"]:
            pts.append(action_dict[f"future_tracks_{pkey}"][:self.num_robot_points, :self.point_dim])

        # Get 3D robot points
        if self.point_dim == 2:
            points3d = triangulate_points(self.P, pts)[:, :3]
        else:  # point_dim == 3
            points3d = np.mean(pts, axis=0)

        # Rigid transform: match predicted gripper cloud to base cloud → get R, t
        robot_pos = points3d[0, :3]
        if self.base_robot_points is not None:
            rot, _ = rigid_transform_3D(self.base_robot_points, points3d)
            robot_ori = rot @ self.robot_base_orientation
        else:
            robot_ori = self.robot_base_orientation

        # Build 4×4 TCP pose, then undo Tshift to get flange pose
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = robot_ori
        T_tcp[:3, 3] = robot_pos
        T_flange = T_tcp @ np.linalg.inv(self.Tshift)

        target_pos = T_flange[:3, 3]
        target_quat = R.from_matrix(T_flange[:3, :3]).as_quat()  # [qx, qy, qz, qw]

        # Gripper state with hysteresis
        raw_gripper = float(action_dict["gripper"].ravel()[0])
        if self.prev_gripper_state == -1 and raw_gripper > -0.3:
            gripper = 1.0
        elif self.prev_gripper_state == 1 and raw_gripper < 0.6:
            gripper = -1.0
        else:
            gripper = float(self.prev_gripper_state)
        self.prev_gripper_state = gripper

        return {
            "flange_pos": target_pos,
            "flange_quat": target_quat,
            "tcp_pos": T_tcp[:3, 3],
            "tcp_quat": R.from_matrix(T_tcp[:3, :3]).as_quat(),
            "gripper": gripper,
        }


# ---------------------------------------------------------------------------
# Norm stats from pkl
# ---------------------------------------------------------------------------
def load_norm_stats(data_pkl_path):
    """
    Load normalization statistics from the training pkl.
    These must match what BCDataset.stats returns during training.
    """
    data = pkl.load(open(data_pkl_path, "rb"))
    observations = data["observations"]

    all_tracks = []
    all_gripper = []
    for obs in observations:
        for key in ["pixels1", "pixels2"]:
            robot_key = f"robot_tracks_{key}"
            if robot_key in obs:
                all_tracks.append(obs[robot_key])
        if "gripper_states" in obs:
            all_gripper.append(obs["gripper_states"])

    all_tracks = np.concatenate(all_tracks, axis=0)
    all_gripper = np.concatenate(all_gripper, axis=0)

    stats = {
        "past_tracks": {
            "min": all_tracks.min(axis=0).min(),
            "max": all_tracks.max(axis=0).max(),
        },
        "gripper_states": {
            "min": float(all_gripper.min()),
            "max": float(all_gripper.max()),
        },
    }
    return stats


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_inference(args):
    device = torch.device(args.device)

    # Load calibration
    calib_data = np.load(args.calib_path, allow_pickle=True).item()

    # Load UR5e gripper geometry
    sys.path.insert(0, str(REPO_ROOT / "point_policy" / "robot_utils" / "ur5e"))
    import gripper_points as gp
    Tshift = gp.Tshift
    robot_base_orientation = gp.robot_base_orientation

    # Action converter
    converter = ActionConverter(calib_data, args.num_robot_points, args.point_dim, Tshift)

    # Load model
    print("Loading model checkpoint...")
    payload = torch.load(args.checkpoint, map_location=device)
    # The snapshot contains the agent state; rebuild using saved config
    # (assumes checkpoint was saved with agent.save_snapshot())
    from agent.point_policy import BCAgent
    agent_cfg = payload.get("cfg", None)
    if agent_cfg is not None:
        agent = BCAgent(**agent_cfg)
    else:
        raise RuntimeError(
            "Checkpoint does not contain agent config. "
            "Make sure the checkpoint was saved with agent.save_snapshot() from train.py."
        )
    agent.load_snapshot(payload, eval=True)
    agent.to(device)
    agent.eval()
    print("Model loaded.")

    # Norm stats from training data
    norm_stats = load_norm_stats(args.data_pkl)

    # Load CoTracker (find checkpoint path from points_cfg.yaml)
    import yaml
    cfg_path = REPO_ROOT / "point_policy" / "cfgs" / "suite" / "points_cfg.yaml"
    with open(cfg_path) as f:
        pts_cfg = yaml.safe_load(f)
    cotracker_ckpt = Path(pts_cfg["root_dir"]) / pts_cfg["cotracker_checkpoint"]
    if not cotracker_ckpt.exists():
        raise FileNotFoundError(f"CoTracker checkpoint not found: {cotracker_ckpt}")
    cotracker1 = load_cotracker(str(cotracker_ckpt), device)
    cotracker2 = load_cotracker(str(cotracker_ckpt), device)
    print("CoTracker loaded.")

    # Initialize cameras
    print("Starting cameras...")
    cam1 = RealSenseCamera(args.cam_serials[0], args.width, args.height, args.fps)
    cam2 = RealSenseCamera(args.cam_serials[1], args.width, args.height, args.fps)

    # Warm up
    print("Warming up cameras (2s)...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        cam1.get_frames()
        cam2.get_frames()

    # UR5e controller
    robot = None
    if not args.dry_run:
        if args.ur5e_ip is None:
            raise ValueError("--ur5e_ip is required when not using --dry_run")
        robot = UR5eController(args.ur5e_ip)

    print("\nReady. Press ENTER to start an episode, 'q' + ENTER to quit.")

    total_num_points = args.num_robot_points  # 9 robot + possibly object points
    # For inference we track only robot points (the policy predicts robot tracks)
    # Object points are useful for training but not required at inference time.

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break

            # ----------------------------------------------------------------
            # Episode start: capture first frame, select keypoints
            # ----------------------------------------------------------------
            img1_bgr, depth1 = cam1.get_frames()
            img2_bgr, depth2 = cam2.get_frames()

            img1_rgb = img1_bgr[:, :, ::-1].copy()
            img2_rgb = img2_bgr[:, :, ::-1].copy()

            print(f"\nSelect {args.num_robot_points} keypoints on CAMERA 1 (hand/gripper region).")
            init_pts_cam1 = select_initial_keypoints(img1_bgr, args.num_robot_points,
                                                     "Camera 1 — select keypoints")
            print(f"\nSelect {args.num_robot_points} keypoints on CAMERA 2 (same points, different view).")
            init_pts_cam2 = select_initial_keypoints(img2_bgr, args.num_robot_points,
                                                     "Camera 2 — select keypoints")

            # Get initial 3D robot points from depth + calibration
            def unproject(pts_2d, depth_img, cam_name):
                K = calib_data[cam_name]["int"]
                ext = calib_data[cam_name]["ext"]
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                pts_3d = []
                for pt in pts_2d:
                    px, py = int(pt[0]), int(pt[1])
                    d = float(depth_img[py, px]) / 1000.0  # mm → meters
                    if d < 0.01:
                        d = 0.5  # fallback if depth invalid
                    xc = (px - cx) / fx * d
                    yc = (py - cy) / fy * d
                    zc = d
                    pt_cam = np.array([xc, yc, zc, 1.0])
                    pt_world = np.linalg.inv(ext) @ pt_cam
                    pts_3d.append(pt_world[:3])
                return np.array(pts_3d, dtype=np.float32)

            base_pts_3d = unproject(init_pts_cam1, depth1, "cam_1")
            converter.set_base_points(base_pts_3d, robot_base_orientation)

            # Initialize CoTracker online — one instance per camera to avoid shared state
            def make_video_tensor(img_rgb):
                t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                return t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)

            queries1 = build_cotracker_query(init_pts_cam1).to(device)
            queries2 = build_cotracker_query(init_pts_cam2).to(device)

            video1 = make_video_tensor(img1_rgb)
            video2 = make_video_tensor(img2_rgb)

            cotracker1(video_chunk=video1, is_first_step=True, queries=queries1)
            cotracker2(video_chunk=video2, is_first_step=True, queries=queries2)

            agent.buffer_reset()

            print("\nEpisode started. Executing...")

            for step in range(args.max_steps):
                # Grab frames — capture fresh depth every step for accurate 3D unprojection
                img1_bgr, depth1 = cam1.get_frames()
                img2_bgr, depth2 = cam2.get_frames()
                img1_rgb = img1_bgr[:, :, ::-1].copy()
                img2_rgb = img2_bgr[:, :, ::-1].copy()

                video1 = make_video_tensor(img1_rgb)
                video2 = make_video_tensor(img2_rgb)

                # Track points — each camera has its own tracker instance
                pred1, _ = cotracker1(video_chunk=video1, is_first_step=False)
                pred2, _ = cotracker2(video_chunk=video2, is_first_step=False)

                # pred shape: (1, T, N, 2) — take last frame, all points
                tracks1 = pred1[0, -1].cpu().numpy()   # (N, 2) pixel coords
                tracks2 = pred2[0, -1].cpu().numpy()

                # Build observation dict
                if args.point_dim == 3:
                    # Unproject to 3D using live depth (requires depth stream)
                    pts3d_1 = unproject(tracks1, depth1, "cam_1")
                    pts3d_2 = unproject(tracks2, depth2, "cam_2")
                    obs = {
                        "point_tracks_pixels1": pts3d_1.astype(np.float32),
                        "point_tracks_pixels2": pts3d_2.astype(np.float32),
                        "features": np.array([0.0] * 7 + [float(converter.prev_gripper_state)]),
                    }
                else:
                    obs = {
                        "point_tracks_pixels1": tracks1.astype(np.float32),
                        "point_tracks_pixels2": tracks2.astype(np.float32),
                        "features": np.array([0.0] * 7 + [float(converter.prev_gripper_state)]),
                    }

                # Policy inference
                with torch.no_grad():
                    action_dict = agent.act(obs, norm_stats, step, 0, eval_mode=True)

                # Convert keypoints → UR5e TCP pose
                robot_action = converter.convert(action_dict)
                pos = robot_action["flange_pos"]
                quat = robot_action["flange_quat"]
                gripper = robot_action["gripper"]
                tcp_pos = robot_action["tcp_pos"]

                print(f"  Step {step:03d} | flange_pos={pos.round(4)} | "
                      f"tcp_pos={tcp_pos.round(4)} | gripper={'CLOSE' if gripper>0 else 'OPEN'}")

                if robot is not None:
                    robot.move_to(pos, quat, args.move_speed, args.move_accel)
                    # Gripper control (requires separate gripper driver — see your gripper's API)
                    # e.g.: gripper_ctrl.move(0 if gripper < 0 else 255)

                # Visualize
                vis = np.hstack([img1_bgr.copy(), img2_bgr.copy()])
                for pt in tracks1:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                for pt in tracks2:
                    cv2.circle(vis, (int(pt[0]) + img1_bgr.shape[1], int(pt[1])), 4, (0, 255, 0), -1)
                cv2.putText(vis, f"Step {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Inference", cv2.resize(vis, (1280, 480)))
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

            print("Episode done.")
            cv2.destroyAllWindows()

    finally:
        cam1.stop()
        cam2.stop()
        if robot is not None:
            robot.stop()
        cv2.destroyAllWindows()
        print("Inference stopped.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
