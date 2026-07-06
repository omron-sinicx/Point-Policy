"""
Standalone data collection script for 2x RealSense D435 cameras.
Replaces the Franka-Teach data collection stack.

Depth is always recorded (aligned to color via rs.align) and saved as float32
meters in cam_{id}_depth.pkl. Use --no_depth to disable if storage is limited.

Saved per demo:
    cam_1_rgb_video.avi / .metadata   — colour video + per-frame timestamps (ms)
    cam_2_rgb_video.avi / .metadata
    cam_1_depth.pkl                    — list[H×W float32 metres], holes filled
    cam_2_depth.pkl
    cam_1_depth.metadata               — per-frame timestamps (ms)
    cam_2_depth.metadata
    states.pkl                         — list[DummyState] (pos/quat zeros; timestamp=cam1 ts)

Usage:
    # Reads defaults (data_dir, cam_serials, width/height/fps) from
    # collect_data_config.yaml next to this script; only --task_name is
    # typically needed per recording session:
    python ur5e_pipeline/collect_data_realsense.py --task_name pick_cup

    # Any config value can still be overridden on the CLI, or a different
    # config file supplied entirely:
    python ur5e_pipeline/collect_data_realsense.py \
        --task_name pick_cup \
        --data_dir /path/to/data \
        --cam_serials <serial1> <serial2> \
        --config /path/to/other_config.yaml

Controls (OpenCV window must have focus):
    SPACE / ENTER  — start recording a new demo
    ENTER          — stop and SAVE the current demo
    r              — discard current demo and immediately re-record (same demo ID)
    q              — discard current demo (if recording) and quit
"""

import time
import argparse
import pickle as pkl
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "collect_data_config.yaml"


# ---------------------------------------------------------------------------
# Dummy robot state (replaces FrankaState from Franka-Teach).
# process_data_human.py reads: .pos, .quat, .gripper, .timestamp, .start_teleop
# ---------------------------------------------------------------------------
@dataclass
class DummyState:
    pos: np.ndarray       # zeros (3,)        — unused; hand tracking replaces this
    quat: np.ndarray      # [0, 0, 0, 1]      — identity quaternion
    gripper: np.ndarray   # [0.0]             — unused
    timestamp: float      # ms, matches camera frame timestamp
    start_teleop: bool    # always True → no idle trimming in process_data_human.py


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------
def _fill_depth_holes(depth_m: np.ndarray, max_hole_radius: int = 4) -> np.ndarray:
    """
    Fill small depth holes (0.0 pixels) using nearest valid neighbour via
    morphological dilation. Holes larger than ~max_hole_radius pixels are left
    as 0.0 so the caller can decide how to handle missing data.

    depth_m: float32 (H, W) depth in meters, 0.0 = no reading.
    Returns: float32 (H, W) with small holes patched.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max_hole_radius + 1, 2 * max_hole_radius + 1)
    )
    # Dilate valid pixels outward to cover nearby holes, then restore valid pixels.
    dilated = cv2.dilate(depth_m, kernel)
    filled = np.where(depth_m == 0.0, dilated, depth_m)
    return filled


# ---------------------------------------------------------------------------
# RealSense pipeline for a single camera
# ---------------------------------------------------------------------------
class RealSenseCamera:
    def __init__(self, serial: str, width=640, height=480, fps=30, collect_depth=True):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.collect_depth = collect_depth
        self.pipeline = rs.pipeline()
        self.align = None

    def start(self):
        if self.collect_depth:
            profile = self._start_streams(depth=True)
            if profile is None:
                print(f"  [Camera {self.serial}] WARNING: depth stream failed to start "
                      f"(USB bandwidth or format error).\n"
                      f"  Falling back to colour-only. Connect via USB 3.0 or use --no_depth "
                      f"to suppress this warning.\n"
                      f"  Without depth you cannot use --use_gt_depth in later steps.")
                self.collect_depth = False
                profile = self._start_streams(depth=False)
        else:
            profile = self._start_streams(depth=False)
        return profile

    def _start_streams(self, depth: bool):
        """Start pipeline with or without depth. Returns profile, or None on failure."""
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            if depth:
                return None   # caller will retry without depth
            raise RuntimeError(f"Camera {self.serial}: colour stream failed: {e}") from e

        if depth:
            self.align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            # depth_scale converts raw uint16 → metres (typically 0.001 for D435).
            self.depth_scale = depth_sensor.get_depth_scale()
            try:
                depth_sensor.set_option(rs.option.visual_preset, 1)  # high accuracy
            except Exception:
                pass

        return profile

    def get_frames(self):
        """Returns (color_bgr, depth_meters_float32_or_None, timestamp_ms)."""
        frames = self.pipeline.wait_for_frames()
        timestamp_ms = frames.get_timestamp()

        if self.collect_depth and self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        # .copy() required: np.asanyarray() is a view into the SDK's internal buffer;
        # that buffer is overwritten on the next wait_for_frames() call.
        color_image = np.asanyarray(color_frame.get_data()).copy()

        depth_image = None
        if self.collect_depth:
            depth_frame = frames.get_depth_frame()
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()  # uint16, sensor units
            # Convert to float32 meters. Zero pixels are depth holes (no valid reading).
            depth_image = depth_raw.astype(np.float32) * self.depth_scale
            # Fill depth holes (0.0) at small scales — fingertips often land on edges
            # where the sensor returns no reading. 5×5 median over valid neighbours.
            depth_image = _fill_depth_holes(depth_image)

        return color_image, depth_image, timestamp_ms

    def stop(self):
        self.pipeline.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Collect demo data from 2 RealSense D435 cameras")
    p.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="YAML file with task_name/data_dir/cam_serials/width/height/fps. "
             "Any of the flags below, if passed, override the config value."
    )
    p.add_argument("--task_name", default=None, help="Task name (used as folder name)")
    p.add_argument("--data_dir", default=None, help="Root data directory")
    p.add_argument(
        "--cam_serials",
        nargs=2,
        default=None,
        metavar=("SERIAL1", "SERIAL2"),
        help="Camera serial numbers, in [cam_1, cam_2] order (run "
             "`rs-enumerate-devices` to find). Must match the cam_1/cam_2 "
             "order calib.npy was calibrated with — do not reorder casually."
    )
    p.add_argument("--no_depth", action="store_true", help="Disable depth recording (not recommended)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    args = p.parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    for key, cfg_key in [
        ("task_name", "task_name"), ("data_dir", "data_dir"),
        ("cam_serials", "cam_serials"), ("width", "width"),
        ("height", "height"), ("fps", "fps"),
    ]:
        if getattr(args, key) is None and cfg.get(cfg_key) is not None:
            setattr(args, key, cfg[cfg_key])

    # Fall back to the script's original hard-coded defaults if still unset.
    if args.cam_serials is None:
        args.cam_serials = ["040322073651", "143322073893"]
    if args.width is None:
        args.width = 640
    if args.height is None:
        args.height = 480
    if args.fps is None:
        args.fps = 30

    if not args.task_name:
        p.error("--task_name is required (pass on the CLI or set it in the config file)")
    if not args.data_dir:
        p.error("--data_dir is required (pass on the CLI or set it in the config file)")
    args.cam_serials = [str(s) for s in args.cam_serials]

    args.collect_depth = not args.no_depth
    return args


def get_next_demo_id(task_dir: Path) -> int:
    existing = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("demo_")]
    if not existing:
        return 0
    return max(int(d.name.split("_")[-1]) for d in existing) + 1


def save_demo(demo_dir: Path, cam_id: int, rgb_frames, rgb_timestamps,
              depth_frames, depth_timestamps, collect_depth: bool):
    demo_dir.mkdir(parents=True, exist_ok=True)

    h, w = rgb_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(demo_dir / f"cam_{cam_id}_rgb_video.avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for frame in rgb_frames:
        writer.write(frame)
    writer.release()

    with open(demo_dir / f"cam_{cam_id}_rgb_video.metadata", "wb") as f:
        pkl.dump({"timestamps": rgb_timestamps}, f)

    if collect_depth and depth_frames:
        with open(demo_dir / f"cam_{cam_id}_depth.pkl", "wb") as f:
            pkl.dump(depth_frames, f)
        with open(demo_dir / f"cam_{cam_id}_depth.metadata", "wb") as f:
            pkl.dump({"timestamps": depth_timestamps}, f)


def build_dummy_states(timestamps_ms):
    return [
        DummyState(
            pos=np.zeros(3, dtype=np.float32),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            gripper=np.array([0.0], dtype=np.float32),
            timestamp=ts,
            start_teleop=True,
        )
        for ts in timestamps_ms
    ]


def save_and_write_states(task_dir, demo_id, cam_ids,
                          rec_rgb, rec_ts, rec_depth, rec_dts,
                          n_frames, collect_depth):
    demo_dir = task_dir / f"demo_{demo_id}"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for cid in cam_ids:
        save_demo(demo_dir, cid,
                  rec_rgb[cid][:n_frames], rec_ts[cid][:n_frames],
                  rec_depth[cid][:n_frames] if collect_depth else [],
                  rec_dts[cid][:n_frames] if collect_depth else [],
                  collect_depth)
    states = build_dummy_states(rec_ts[cam_ids[0]][:n_frames])
    with open(demo_dir / "states.pkl", "wb") as f:
        pkl.dump(states, f)
    print(f"[Demo {demo_id}] Saved to {demo_dir}")


# ---------------------------------------------------------------------------
# Main loop — continuous OpenCV event loop, state machine
# ---------------------------------------------------------------------------
def run(cameras, cam_ids, task_dir, collect_depth):
    """
    Continuous OpenCV event loop so windows never freeze and keys always register.

    State machine:
        IDLE      — live feed; SPACE/ENTER starts recording
        RECORDING — live feed + REC overlay; ENTER saves, R re-records, Q quits
    """
    IDLE, RECORDING = "idle", "recording"
    state   = IDLE
    demo_id = None

    latest_color = {cid: None for cid in cam_ids}
    rec_rgb      = {cid: [] for cid in cam_ids}
    rec_ts       = {cid: [] for cid in cam_ids}
    rec_depth    = {cid: [] for cid in cam_ids}
    rec_dts      = {cid: [] for cid in cam_ids}

    recording_flag = threading.Event()
    stop_event = threading.Event()

    def capture_thread(cam, cid):
        while not stop_event.is_set():
            try:
                color, depth, ts = cam.get_frames()
            except RuntimeError:
                # Pipeline was stopped (or is being stopped) from the main
                # thread — exit quietly instead of racing wait_for_frames()
                # against pipeline.stop(), which can abort the process.
                break
            latest_color[cid] = color
            if recording_flag.is_set():
                rec_rgb[cid].append(color)
                rec_ts[cid].append(ts)
                if collect_depth and depth is not None:
                    rec_depth[cid].append(depth)
                    rec_dts[cid].append(ts)

    threads = [
        threading.Thread(target=capture_thread, args=(cam, cid), daemon=True)
        for cam, cid in zip(cameras, cam_ids)
    ]
    for t in threads:
        t.start()

    # Let capture threads produce the first frames before entering the display loop
    time.sleep(0.5)

    print("\nSPACE/ENTER=start recording   Q=quit")

    while True:
        for cid in cam_ids:
            frame = latest_color[cid]
            if frame is None:
                continue
            disp = frame.copy()
            if state == RECORDING:
                cv2.circle(disp, (20, 20), 10, (0, 0, 255), -1)
                cv2.putText(disp, f"REC  {len(rec_rgb[cid])} frames", (35, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(disp, "ENTER=save   R=re-record   Q=quit",
                            (10, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(disp, "SPACE / ENTER = start recording   Q = quit",
                            (10, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(f"Camera {cid}", disp)

        key = cv2.waitKey(33) & 0xFF   # 33 ms ≈ 30 fps; drives the window event loop

        if state == IDLE:
            if key in (ord(" "), 13, 10):       # SPACE or ENTER → start recording
                demo_id = get_next_demo_id(task_dir)
                for cid in cam_ids:
                    rec_rgb[cid].clear(); rec_ts[cid].clear()
                    rec_depth[cid].clear(); rec_dts[cid].clear()
                recording_flag.set()
                state = RECORDING
                print(f"\n[Demo {demo_id}] Recording...  ENTER=save  R=re-record  Q=quit")
            elif key == ord("q"):
                break

        elif state == RECORDING:
            if key in (13, 10):                 # ENTER → save
                recording_flag.clear()
                n = min(len(rec_rgb[cid]) for cid in cam_ids)
                print(f"[Demo {demo_id}] {n} frames captured. Saving...")
                save_and_write_states(task_dir, demo_id, cam_ids,
                                      rec_rgb, rec_ts, rec_depth, rec_dts,
                                      n, collect_depth)
                state = IDLE
                print("SPACE/ENTER=next demo   Q=quit")

            elif key == ord("r"):               # R → discard + re-record same ID
                recording_flag.clear()
                n = min(len(rec_rgb[cid]) for cid in cam_ids)
                print(f"[Demo {demo_id}] Discarded ({n} frames). Re-recording...")
                for cid in cam_ids:
                    rec_rgb[cid].clear(); rec_ts[cid].clear()
                    rec_depth[cid].clear(); rec_dts[cid].clear()
                recording_flag.set()

            elif key == ord("q"):               # Q → discard + quit
                recording_flag.clear()
                print(f"[Demo {demo_id}] Discarded. Quitting.")
                break

    # Signal capture threads to exit their get_frames() loop and wait for
    # them to actually stop *before* returning — the caller stops the
    # RealSense pipelines right after this, and stopping a pipeline while a
    # thread is still blocked inside wait_for_frames() crashes the process.
    stop_event.set()
    for t in threads:
        t.join(timeout=2.0)

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    task_dir = Path(args.data_dir) / "extracted_data" / args.task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    cam_ids = [1, 2]
    cameras = []
    for serial, cid in zip(args.cam_serials, cam_ids):
        cam = RealSenseCamera(serial, args.width, args.height, args.fps, args.collect_depth)
        cam.start()
        cameras.append(cam)
        print(f"Camera {cid} (serial={serial}) started.")

    print("Warming up cameras (2s)...")
    end = time.time() + 2.0
    while time.time() < end:
        for cam in cameras:
            cam.get_frames()

    try:
        run(cameras, cam_ids, task_dir, args.collect_depth)
    finally:
        for cam in cameras:
            cam.stop()
        print("Cameras stopped.")


if __name__ == "__main__":
    main()
