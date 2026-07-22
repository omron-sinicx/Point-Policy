"""Script to label object points -- Tkinter version of label_points.ipynb.
Adapted from P3PO - https://github.com/mlevy2525/P3PO/blob/main/p3po/data_generation/label_points.ipynb

Usage:
    python label_points.py --task_name bottle_open_07
    python label_points.py --task_name bottle_open_07 --pixel_keys pixels1 pixels2

Opens one Tkinter window per --pixel_key, in order (default: pixels1 then
pixels2) -- click the same object point(s) in the same order in each, then
click "Save Points" to close that window and move to the next.
"""

import argparse
import importlib
import pickle
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

# point_policy/ (point_utils)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Loaded via importlib rather than a top-level "from point_utils import
# task_pkl_io" -- an editor's auto-organize-imports-on-save (isort etc.)
# will always hoist a real import statement above the sys.path.insert it
# depends on, breaking this script when run directly (not through
# run_pipeline.sh, which sets sys.path differently). A function call is
# invisible to import sorters, so this ordering survives save.
task_pkl_io = importlib.import_module("point_utils.task_pkl_io")


# point_policy/robot_utils/franka -> repo root, matching REPO_ROOT in run_pipeline.sh
REPO_ROOT = Path(__file__).resolve().parents[3]
# Data lives one level above Point-Policy (osx-ur/data), matching DATA_DIR in
# run_pipeline.sh -- not inside dependencies/Point-Policy/.
DEFAULT_DATA_ROOT = REPO_ROOT.parents[1] / "data"

# Fixed rather than exposed as CLI args -- nothing in the pipeline varies
# these today (object_labels is always just ["objects"] elsewhere too).
object_name = "objects"
traj_idx = 0
original_bgr = True
size_multiplier = 1


class PointLabeler:
    def __init__(self, root, pixel_key, img, coordinates_path, size_multiplier=1):
        self.root = root
        self.pixel_key = pixel_key
        self.img = img
        self.coordinates_path = coordinates_path
        self.size_multiplier = size_multiplier
        self.coords = []

        h, w = img.shape[:2]
        self.display_size = (w * size_multiplier, h * size_multiplier)

        self.base_image = Image.fromarray(img).resize(self.display_size)
        self.tk_image = ImageTk.PhotoImage(self.base_image)

        self.canvas = tk.Canvas(
            root, width=self.display_size[0], height=self.display_size[1]
        )
        self.canvas.pack()
        self.canvas_image_id = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.tk_image
        )

        self.coords_label = tk.Label(
            root, text=f"[{pixel_key}] Click on the image to select the coordinates"
        )
        self.coords_label.pack()

        self.save_button = tk.Button(
            root, text="Save Points", command=self.on_done
        )
        self.save_button.pack()

        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        x, y = event.x, event.y
        self.coords_label.config(text=f"[{self.pixel_key}] Coordinates: ({x}, {y})")
        self.coords.append((0, x, y))

        orig_x, orig_y = x // self.size_multiplier, y // self.size_multiplier
        r = 2
        self.canvas.create_oval(
            orig_x * self.size_multiplier - r,
            orig_y * self.size_multiplier - r,
            orig_x * self.size_multiplier + r,
            orig_y * self.size_multiplier + r,
            fill="red",
            outline="red",
        )

    def on_done(self):
        print(f"saving {self.pixel_key}")
        coords_dir = Path(self.coordinates_path) / "coords"
        coords_dir.mkdir(parents=True, exist_ok=True)
        with open(coords_dir / f"{self.pixel_key}_{object_name}.pkl", "wb") as f:
            pickle.dump(self.coords, f)

        images_dir = Path(self.coordinates_path) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(self.img)
        image.save(images_dir / f"{self.pixel_key}.png")
        print(f"saved {self.pixel_key}")
        self.root.destroy()


def label_pixel_key(pixel_key, task_pkl_dir, coordinates_path):
    """Open one Tkinter window for this pixel_key; blocks until the user
    clicks 'Save Points' (which closes the window)."""
    demo = task_pkl_io.read_demo(task_pkl_dir, traj_idx)
    img = demo[pixel_key][0]
    if original_bgr:
        img = img[:, :, ::-1]

    root = tk.Tk()
    root.title(f"Label Points -- {pixel_key}")
    PointLabeler(root, pixel_key, img, coordinates_path, size_multiplier)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually label object keypoint(s) for one or more camera views."
    )
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name -- matches PKL_TASK_NAME in run_pipeline.sh "
                             "(includes the _gt_depth suffix if that's in use).")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_ROOT),
                        help=f"Root data directory (default: {DEFAULT_DATA_ROOT}, "
                             "matching run_pipeline.sh's DATA_DIR).")
    parser.add_argument("--pixel_keys", nargs="+", default=["pixels1", "pixels2"],
                        help="Camera pixel keys to label, in order, one Tkinter "
                             "window each (default: pixels1 pixels2). Click the "
                             "same object point(s) in the same order in every one.")
    return parser.parse_args()


def main():
    args = parse_args()
    task_pkl_dir = Path(args.data_dir) / "processed_data_pkl" / args.task_name
    coordinates_path = str(REPO_ROOT / "coordinates" / args.task_name)

    for pixel_key in args.pixel_keys:
        print(f"\n=== Labeling {pixel_key} ({args.task_name}) ===")
        print("Click the object point(s) in the same order as every other "
              "pixel_key, then click 'Save Points' to continue.")
        label_pixel_key(pixel_key, task_pkl_dir, coordinates_path)

    print(f"\nAll pixel_keys labeled -- saved under {coordinates_path}/")


if __name__ == "__main__":
    main()
