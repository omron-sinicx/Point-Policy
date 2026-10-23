"""Script to label object points -- Tkinter version of label_points.ipynb.
Adapted from P3PO - https://github.com/mlevy2525/P3PO/blob/main/p3po/data_generation/label_points.ipynb

Usage:
    python label_points.py --task_name bottle_open_07
    python label_points.py --task_name bottle_open_07 --pixel_keys pixels1 pixels2
    python label_points.py --task_name peg_hole_15 --label gripper_orientation

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
# these today.
traj_idx = 0
original_bgr = True


class PointLabeler:
    """Resizable point-labeling canvas.

    The image scales to fill whatever size you drag the window to (aspect
    ratio preserved, letterboxed), so you can zoom in for precise clicks.

    Clicked points are stored in ORIGINAL image pixel coordinates, never
    canvas coordinates -- everything downstream assumes that space:
    PointsClass reads these coords straight from the pkl, and DIFT rescales
    them with `coord * self.width / self.original_size[...]`, where
    original_size is the saved reference PNG's own size. Storing zoomed
    canvas coordinates would silently offset every keypoint by the zoom
    factor.
    """

    MARKER_RADIUS = 4

    def __init__(self, root, pixel_key, img, coordinates_path, label, init_scale=1.5):
        self.root = root
        self.pixel_key = pixel_key
        self.img = img
        self.coordinates_path = coordinates_path
        self.label = label
        self.coords = []  # (0, x, y) in ORIGINAL image pixels

        self.base_image = Image.fromarray(img)
        self.orig_w, self.orig_h = self.base_image.size

        # Render state, recomputed on every resize.
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self._rendered_size = None
        self.tk_image = None  # must stay referenced or Tk garbage-collects it

        # Controls first with side=BOTTOM so the canvas expands into whatever
        # space is left instead of squeezing them off-window.
        self.save_button = tk.Button(root, text="Save Points", command=self.on_done)
        self.save_button.pack(side=tk.BOTTOM, pady=4)

        self.coords_label = tk.Label(
            root,
            text=f"[{pixel_key}] click to add a point  |  right-click / u = undo",
        )
        self.coords_label.pack(side=tk.BOTTOM)

        self.canvas = tk.Canvas(root, highlightthickness=0, background="black")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_undo)
        root.bind("<KeyPress-u>", self.on_undo)

        root.geometry(f"{int(self.orig_w * init_scale)}x{int(self.orig_h * init_scale) + 60}")
        root.minsize(320, 300)

    # -- rendering ---------------------------------------------------------
    def on_resize(self, event):
        # <Configure> also fires for non-size changes; re-rendering the image
        # every time would be wasted work.
        if (event.width, event.height) == self._rendered_size:
            return
        if event.width < 2 or event.height < 2:
            return
        self._rendered_size = (event.width, event.height)

        # Fit inside the canvas, preserving aspect ratio; centre the result.
        self.scale = min(event.width / self.orig_w, event.height / self.orig_h)
        disp_w = max(1, int(self.orig_w * self.scale))
        disp_h = max(1, int(self.orig_h * self.scale))
        self.off_x = (event.width - disp_w) // 2
        self.off_y = (event.height - disp_h) // 2

        # NEAREST keeps pixel edges crisp when zoomed in, which is what you
        # want for picking an exact pixel.
        resample = Image.NEAREST if self.scale >= 1 else Image.LANCZOS
        self.tk_image = ImageTk.PhotoImage(
            self.base_image.resize((disp_w, disp_h), resample)
        )
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)
        self.canvas.coords(self.canvas_image_id, self.off_x, self.off_y)
        self.redraw_markers()

    def to_canvas(self, x, y):
        return self.off_x + x * self.scale, self.off_y + y * self.scale

    def to_image(self, cx, cy):
        return (cx - self.off_x) / self.scale, (cy - self.off_y) / self.scale

    def redraw_markers(self):
        self.canvas.delete("marker")
        r = self.MARKER_RADIUS
        for i, (_, x, y) in enumerate(self.coords):
            cx, cy = self.to_canvas(x + 0.5, y + 0.5)  # centre of the pixel
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill="red", outline="white", tags="marker")
            # Index matters: click order must match across camera views.
            self.canvas.create_text(cx + r + 2, cy - r - 2, text=str(i), anchor=tk.SW,
                                    fill="yellow", tags="marker")

    def set_status(self, extra=""):
        self.coords_label.config(
            text=f"[{self.pixel_key}] {len(self.coords)} point(s){extra}"
            f"  |  right-click / u = undo"
        )

    # -- interaction -------------------------------------------------------
    def on_click(self, event):
        ix, iy = self.to_image(event.x, event.y)
        if not (0 <= ix < self.orig_w and 0 <= iy < self.orig_h):
            return  # clicked the letterbox margin, not the image
        self.coords.append((0, int(ix), int(iy)))
        self.redraw_markers()
        self.set_status(f"  last: ({int(ix)}, {int(iy)})")

    def on_undo(self, event=None):
        if self.coords:
            removed = self.coords.pop()
            self.redraw_markers()
            self.set_status(f"  removed ({removed[1]}, {removed[2]})")

    def on_done(self):
        print(f"saving {self.pixel_key}")
        coords_dir = Path(self.coordinates_path) / "coords"
        coords_dir.mkdir(parents=True, exist_ok=True)
        with open(coords_dir / f"{self.pixel_key}_{self.label}.pkl", "wb") as f:
            pickle.dump(self.coords, f)

        images_dir = Path(self.coordinates_path) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        # Always the unscaled original -- DIFT treats this as the coordinate
        # space the saved points live in.
        image = Image.fromarray(self.img)
        image.save(images_dir / f"{self.pixel_key}.png")
        print(f"saved {self.pixel_key}: {len(self.coords)} point(s) {self.coords}")
        self.root.destroy()


def label_pixel_key(pixel_key, task_pkl_dir, coordinates_path, label, init_scale=1.5):
    """Open one Tkinter window for this pixel_key; blocks until the user
    clicks 'Save Points' (which closes the window)."""
    demo = task_pkl_io.read_demo(task_pkl_dir, traj_idx)
    img = demo[pixel_key][0]
    if original_bgr:
        img = img[:, :, ::-1]

    root = tk.Tk()
    root.title(f"Label Points -- {pixel_key} ({label})")
    PointLabeler(root, pixel_key, img, coordinates_path, label, init_scale)
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
    parser.add_argument("--label", type=str, default="objects",
                        help="Label for this group of keypoints (default: objects). "
                             "Matches the label passed to convert_to_pkl_human_hamer.py's "
                             "--object_labels, and determines the saved filename "
                             "({pixel_key}_{label}.pkl). Run this script again with a "
                             "different --label to annotate a second, independent group "
                             "of points on the same images (e.g. --label "
                             "gripper_orientation for the object_keypoints orientation "
                             "method).")
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Initial window size as a multiple of the image's own "
                             "resolution (default: 1.5, i.e. 960x720 for a 640x480 "
                             "frame). The window is freely resizable afterwards -- the "
                             "image scales to fill it, so drag it larger to zoom in for "
                             "precise clicks.")
    return parser.parse_args()


def main():
    args = parse_args()
    task_pkl_dir = Path(args.data_dir) / "processed_data_pkl" / args.task_name
    coordinates_path = str(REPO_ROOT / "coordinates" / args.task_name)

    for pixel_key in args.pixel_keys:
        print(f"\n=== Labeling {pixel_key} ({args.task_name}, label={args.label}) ===")
        print("Click the object point(s) in the same order as every other "
              "pixel_key, then click 'Save Points' to continue.")
        label_pixel_key(pixel_key, task_pkl_dir, coordinates_path, args.label,
                        args.scale)

    print(f"\nAll pixel_keys labeled -- saved under {coordinates_path}/")


if __name__ == "__main__":
    main()
