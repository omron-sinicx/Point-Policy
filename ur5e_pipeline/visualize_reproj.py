"""
Diagnostic: compare original 2D hand tracks vs reprojection of 3D hand points.

For each camera:
  LEFT  — original 2D tracks from CoTracker (human_tracks_pixels*)
  RIGHT — 3D hand points (human_tracks_3d_pixels*) reprojected to 2D via calib.npy

If LEFT and RIGHT match → calibration is correct; any downstream error is in the
    human-hand → robot-gripper conversion (gripper_points.py / Tshift).
If LEFT and RIGHT differ → calibration / triangulation is the root problem.

Usage:
    python ur5e_pipeline/visualize_reproj.py \
        --pkl_path  data/processed_data_pkl/07011044_test \
        --calib_path calib/calib.npy \
        --out_path  data/reproj_check.mp4 \
        --demo_idx  0
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "point_policy"))
from point_utils import task_pkl_io


COLORS = [
    (0,   255, 0),
    (255, 128, 0),
    (255, 200, 0),
    (255, 255, 0),
    (0,   200, 255),
    (200, 0,   255),
    (255, 0,   200),
    (255, 0,   100),
    (255, 0,   0),
]


def draw_pts(frame, pts_xy, radius=5):
    out = frame.copy()
    for i, (x, y) in enumerate(pts_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        c = COLORS[i % len(COLORS)]
        cv2.circle(out, (x, y), radius, c, -1)
        cv2.circle(out, (x, y), radius + 1, (0, 0, 0), 1)
    return out


OBJECT_COLOR = (255, 0, 255)  # magenta (BGR)


def draw_obj_pts(frame, pts_xy, radius=5):
    """Draw object keypoints as magenta squares (distinct from the circular hand
    markers), in place, labeled by index."""
    for i, (x, y) in enumerate(pts_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        cv2.rectangle(frame, (x - radius, y - radius), (x + radius, y + radius),
                      OBJECT_COLOR, -1)
        cv2.rectangle(frame, (x - radius - 1, y - radius - 1),
                      (x + radius + 1, y + radius + 1), (0, 0, 0), 1)
        cv2.putText(frame, str(i), (x + radius + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, OBJECT_COLOR, 1, cv2.LINE_AA)
    return frame


def label(frame, text, y=22):
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)


def project(pts3d, K, E, D):
    """Project (N,3) world points to (N,2) pixel coords."""
    r, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(r)
    proj, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, t, K, D)
    return proj.squeeze(axis=1)   # (N, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_path",   required=True)
    p.add_argument("--calib_path", required=True)
    p.add_argument("--out_path",   default="reproj_check.mp4")
    p.add_argument("--demo_idx",   type=int, default=0)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--cam", choices=["1", "2", "both"], default="both")
    args = p.parse_args()

    task_pkl_dir = Path(args.pkl_path)
    demo_ids = task_pkl_io.iter_demo_ids(task_pkl_dir)
    calib = np.load(args.calib_path, allow_pickle=True).item()

    n_demos = len(demo_ids)
    if args.demo_idx >= n_demos:
        print(
            f"ERROR: demo_idx={args.demo_idx} but only {n_demos} demo(s) in "
            f"{args.pkl_path}."
            + (
                " The pkl has 0 demos -- check the tracking step's output for "
                "'Error in tracking hand points' (demos are dropped when "
                "per-episode tracking fails, e.g. no hand detected in frame 0)."
                if n_demos == 0 else ""
            )
        )
        sys.exit(1)

    obs   = task_pkl_io.read_demo(task_pkl_dir, demo_ids[args.demo_idx])

    cam_keys = [k for k in ["pixels1", "pixels2"] if k in obs]
    if args.cam == "1":
        cam_keys = [k for k in cam_keys if k == "pixels1"]
    elif args.cam == "2":
        cam_keys = [k for k in cam_keys if k == "pixels2"]

    T = obs[cam_keys[0]].shape[0]
    H, W = obs[cam_keys[0]].shape[1:3]

    # Layout: for each camera → [original | reprojected], cameras stacked side-by-side
    panels_per_cam = 2
    out_w = W * panels_per_cam * len(cam_keys)
    out_h = H

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))

    # Pre-project all 3D tracks so we can also print per-frame error
    reproj_per_cam = {}
    err_per_cam    = {}
    obj_reproj_per_cam = {}   # reprojected object 3D points (magenta squares)
    obj_orig_per_cam   = {}   # original object 2D tracks
    for k in cam_keys:
        cam = "cam_1" if k == "pixels1" else "cam_2"
        K = calib[cam]["int"]
        E = calib[cam]["ext"]
        D = calib[cam]["dist_coeff"]

        tk3d = obs.get(f"human_tracks_3d_{k}")   # (T, N, 3)
        tk2d = obs.get(f"human_tracks_{k}")       # (T, N, 2 or 3)

        # Object tracks (optional). The robot pkl exposes dedicated
        # object_tracks_* keys; the human pkl instead concatenates object points
        # after the 9 hand landmarks in human_tracks_* (indices 9:). Prefer the
        # dedicated keys, else fall back to the tail of human_tracks_*.
        NUM_HAND_POINTS = 9
        obj3d = obs.get(f"object_tracks_3d_{k}")  # (T, M, 3)
        obj2d = obs.get(f"object_tracks_{k}")     # (T, M, 2 or 3)
        if obj3d is None and tk3d is not None and tk3d.shape[1] > NUM_HAND_POINTS:
            obj3d = tk3d[:, NUM_HAND_POINTS:]
        if obj2d is None and tk2d is not None and tk2d.shape[1] > NUM_HAND_POINTS:
            obj2d = tk2d[:, NUM_HAND_POINTS:]
        obj_reproj_per_cam[k] = (
            np.array([project(obj3d[t], K, E, D) for t in range(T)])
            if obj3d is not None else None
        )
        obj_orig_per_cam[k] = obj2d[:, :, :2] if obj2d is not None else None

        if tk3d is None or tk2d is None:
            reproj_per_cam[k] = None
            err_per_cam[k]    = None
            continue

        reproj = np.array([project(tk3d[t, :9], K, E, D) for t in range(T)])  # (T,9,2)
        orig   = tk2d[:, :9, :2]                                                # (T,9,2)
        errs   = np.linalg.norm(reproj - orig, axis=-1)   # (T, 9)
        reproj_per_cam[k] = reproj
        err_per_cam[k]    = errs
        print(f"{cam}: mean reproj error = {errs.mean():.1f} px  "
              f"max = {errs.max():.1f} px  "
              f"per-point mean: {errs.mean(axis=0).round(1)}")
        if obj_reproj_per_cam[k] is not None and obj_orig_per_cam[k] is not None:
            obj_errs = np.linalg.norm(obj_reproj_per_cam[k] - obj_orig_per_cam[k], axis=-1)
            print(f"{cam}: object mean reproj error = {obj_errs.mean():.1f} px  "
                  f"max = {obj_errs.max():.1f} px")

    print(f"\nWriting {T} frames → {out}  ({out_w}×{out_h} @ {args.fps} fps) ...")

    for t in range(T):
        row = []
        for k in cam_keys:
            cam = "cam_1" if k == "pixels1" else "cam_2"
            frame = obs[k][t].copy()

            # LEFT: original 2D tracks (hand circles + object squares)
            tk2d = obs.get(f"human_tracks_{k}")
            orig_panel = draw_pts(frame, tk2d[t, :9, :2]) if tk2d is not None else frame.copy()
            if obj_orig_per_cam[k] is not None:
                draw_obj_pts(orig_panel, obj_orig_per_cam[k][t])
            label(orig_panel, f"{cam} ORIGINAL  t={t}")

            # RIGHT: reprojected 3D→2D (hand circles + object squares)
            rp = reproj_per_cam[k]
            if rp is not None:
                reproj_panel = draw_pts(frame, rp[t])
                if obj_reproj_per_cam[k] is not None:
                    draw_obj_pts(reproj_panel, obj_reproj_per_cam[k][t])
                err_str = f"err={err_per_cam[k][t].mean():.1f}px"
                label(reproj_panel, f"{cam} REPROJ ({err_str})  t={t}")
            else:
                reproj_panel = frame.copy()
                label(reproj_panel, f"{cam} REPROJ (no 3D data)  t={t}")

            row.extend([orig_panel, reproj_panel])

        writer.write(np.hstack(row))
        if t % 50 == 0:
            print(f"  {t}/{T}")

    writer.release()
    print(f"Done → {out}")


if __name__ == "__main__":
    main()
