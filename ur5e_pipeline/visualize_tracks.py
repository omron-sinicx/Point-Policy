"""
Visualize tracked keypoints from a processed pkl file.

Two modes:
  human (default) — loads human_tracks_pixels* from convert_to_pkl_human.py output
  robot           — loads robot_tracks_pixels* + gripper_states from
                    convert_pkl_human_to_robot.py output (expert_demos/.../task.pkl)

Usage:
    # Human tracks (side-by-side cam1 | cam2)
    python ur5e_pipeline/visualize_tracks.py \
        --pkl_path data/processed_data_pkl/07011044_test \
        --out_path data/tracks_vis.mp4 \
        --demo_idx 0

    # Robot action keypoints + gripper state
    python ur5e_pipeline/visualize_tracks.py \
        --pkl_path data/processed_data_pkl/expert_demos/franka_env/07011044_test \
        --out_path data/robot_action_vis.mp4 \
        --demo_idx 0 \
        --mode robot
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "point_policy"))
from point_utils import task_pkl_io


# Colors for up to 9 hand/gripper keypoints
POINT_COLORS = [
    (0,   255, 0),    # 0 — center / wrist
    (255, 128, 0),    # 1
    (255, 200, 0),    # 2
    (255, 255, 0),    # 3
    (0,   200, 255),  # 4
    (200, 0,   255),  # 5
    (255, 0,   200),  # 6
    (255, 0,   100),  # 7
    (255, 0,   0),    # 8
]


def draw_points(frame_bgr, points_xy, radius=5):
    """Overlay 2-D keypoints on a BGR frame. points_xy: (N, 2) float array."""
    out = frame_bgr.copy()
    for i, (x, y) in enumerate(points_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        color = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(out, (x, y), radius, color, -1)
        cv2.circle(out, (x, y), radius + 1, (0, 0, 0), 1)
    return out


OBJECT_COLOR = (255, 0, 255)  # magenta (BGR)


def draw_object_points(frame_bgr, points_xy, radius=5):
    """Overlay object keypoints as magenta squares (distinct from the circular
    hand/robot markers), labeled by index. points_xy: (N, 2) float array."""
    out = frame_bgr
    for i, (x, y) in enumerate(points_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        cv2.rectangle(out, (x - radius, y - radius), (x + radius, y + radius),
                      OBJECT_COLOR, -1)
        cv2.rectangle(out, (x - radius - 1, y - radius - 1),
                      (x + radius + 1, y + radius + 1), (0, 0, 0), 1)
        cv2.putText(out, str(i), (x + radius + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, OBJECT_COLOR, 1, cv2.LINE_AA)
    return out


def put_label(frame, text, y=22):
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Visualize keypoints from pkl")
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--out_path", default="tracks_vis.mp4")
    parser.add_argument("--demo_idx", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cam", choices=["1", "2", "both"], default="both")
    parser.add_argument("--mode", choices=["human", "robot"], default="human",
                        help="'human' — human_tracks_pixels*; "
                             "'robot' — robot_tracks_pixels* with gripper state overlay")
    parser.add_argument("--frames_pkl_path", default=None,
                        help="Optional: draw on pixels{1,2} frames loaded from this pkl "
                             "instead of --pkl_path's own frames (e.g. the human pkl, "
                             "which keeps the original camera resolution instead of the "
                             "training-resized copy the robot pkl stores). Track points "
                             "are rescaled to match the substituted frame size.")
    args = parser.parse_args()

    print(f"Loading {args.pkl_path} ...")
    task_pkl_dir = Path(args.pkl_path)
    demo_ids = task_pkl_io.iter_demo_ids(task_pkl_dir)
    n_demos = len(demo_ids)
    print(f"Found {n_demos} demo(s). Visualizing demo {args.demo_idx} [{args.mode} mode].")

    if args.demo_idx >= n_demos:
        raise ValueError(f"demo_idx {args.demo_idx} out of range (only {n_demos} demos)")

    obs = task_pkl_io.read_demo(task_pkl_dir, demo_ids[args.demo_idx])

    # Determine cameras
    cam_keys = [k for k in ["pixels1", "pixels2"] if k in obs]
    if not cam_keys:
        raise RuntimeError("No pixels1/pixels2 found in observation.")
    if args.cam == "1":
        cam_keys = [k for k in cam_keys if k == "pixels1"]
    elif args.cam == "2":
        cam_keys = [k for k in cam_keys if k == "pixels2"]

    # Pick track key prefix based on mode
    track_prefix = "human_tracks_" if args.mode == "human" else "robot_tracks_"

    # Optionally substitute full-resolution frames from another pkl (e.g. the
    # robot pkl only stores frames resized to the training resolution).
    frames_obs = None
    if args.frames_pkl_path:
        frames_pkl_dir = Path(args.frames_pkl_path)
        frames_demo_ids = task_pkl_io.iter_demo_ids(frames_pkl_dir)
        if args.demo_idx < len(frames_demo_ids):
            frames_obs = task_pkl_io.read_demo(frames_pkl_dir, frames_demo_ids[args.demo_idx])
        else:
            print(f"  Warning: --frames_pkl_path has no demo {args.demo_idx}; "
                  f"falling back to --pkl_path's own frames")

    frames_per_cam = {}
    tracks_per_cam = {}
    scale_per_cam = {}
    for k in cam_keys:
        low_frames = obs[k]
        if frames_obs is not None and k in frames_obs:
            frames_per_cam[k] = frames_obs[k]
            low_H, low_W = low_frames.shape[1:3]
            full_H, full_W = frames_per_cam[k].shape[1:3]
            scale_per_cam[k] = np.array([full_W / low_W, full_H / low_H], dtype=np.float64)
        else:
            frames_per_cam[k] = low_frames
            scale_per_cam[k] = np.array([1.0, 1.0])

        track_key = f"{track_prefix}{k}"
        if track_key in obs:
            tracks_per_cam[k] = obs[track_key]  # (T, N, 2[+])
        else:
            print(f"  Warning: '{track_key}' not found — no points drawn for {k}")
            tracks_per_cam[k] = None

    # Object keypoints: in robot mode the main tracks are robot_tracks (no
    # objects), so overlay object_tracks_* separately. In human mode the object
    # points are already part of human_tracks, so skip to avoid double-drawing.
    object_tracks_per_cam = {}
    for k in cam_keys:
        obj_key = f"object_tracks_{k}"
        object_tracks_per_cam[k] = (
            obs[obj_key] if (args.mode == "robot" and obj_key in obs) else None
        )

    # Gripper states (robot mode only)
    gripper_states = None
    if args.mode == "robot" and "gripper_states" in obs:
        gripper_states = obs["gripper_states"]  # (T,) array of -1/1
        print(f"  Gripper state range: min={gripper_states.min()}, max={gripper_states.max()}")

    T_candidates = [frames_per_cam[k].shape[0] for k in cam_keys]
    T_candidates += [tracks_per_cam[k].shape[0] for k in cam_keys if tracks_per_cam[k] is not None]
    T = min(T_candidates)
    H, W = frames_per_cam[cam_keys[0]].shape[1:3]

    out_w = W * len(cam_keys)
    out_h = H
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))

    print(f"Writing {T} frames → {out_path}  ({out_w}×{out_h} @ {args.fps} fps) ...")

    for t in range(T):
        panels = []
        for k in cam_keys:
            frame = frames_per_cam[k][t].copy()

            if tracks_per_cam[k] is not None:
                pts = tracks_per_cam[k][t, :, :2] * scale_per_cam[k]
                frame = draw_points(frame, pts)

            if object_tracks_per_cam[k] is not None:
                obj_pts = object_tracks_per_cam[k][t, :, :2] * scale_per_cam[k]
                frame = draw_object_points(frame, obj_pts)

            # Top label: frame info
            put_label(frame, f"demo {args.demo_idx}  t={t}/{T-1}  {k}")

            # Bottom label: gripper state (robot mode)
            if gripper_states is not None:
                gs = gripper_states[t]
                gs_text = "CLOSED" if gs > 0 else "OPEN"
                gs_color = (0, 80, 255) if gs > 0 else (0, 220, 0)  # red=closed, green=open
                cv2.putText(frame, gs_text, (8, H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, gs_color, 2, cv2.LINE_AA)

            panels.append(frame)

        writer.write(np.hstack(panels))
        if t % 50 == 0:
            print(f"  {t}/{T}")

    writer.release()
    print(f"Done. Saved to {out_path}")

    for k in cam_keys:
        if tracks_per_cam[k] is not None:
            n_pts = tracks_per_cam[k].shape[1]
            print(f"  {k}: {T} frames, {n_pts} tracked points per frame")


if __name__ == "__main__":
    main()
