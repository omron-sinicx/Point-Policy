"""Script to label object points -- Tkinter version of label_points.ipynb.

Adapted from P3PO - https://github.com/mlevy2525/P3PO/blob/main/p3po/data_generation/label_points.ipynb

Usage:
    python label_points.py
"""

import pickle
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

# point_policy/robot_utils/franka -> repo root, matching REPO_ROOT in run_pipeline.sh
REPO_ROOT = Path(__file__).resolve().parents[3]

# TODO: Set the task name here -- this will be used to save the output
task_name = "07031057_test"
object_name = "objects"

# If the image that shows at the bottom is bgr set original_bgr to True
pickle_path = str(REPO_ROOT / "data" / "processed_data_pkl" / f"{task_name}.pkl")
traj_idx = 0
original_bgr = True

# TODO: If its hard to see the image, you can increase the size_multiplier, this won't affect the selected coordinates
size_multiplier = 1

# Matches COORDS_DIR in ur5e_pipeline/run_pipeline.sh: $REPO_ROOT/coordinates/<task_name>
coordinates_path = str(REPO_ROOT / "coordinates" / task_name)

# NOTE: Label points for each pixel key. Make sure the order
# of points is the same across pixel keys.
pixel_key = "pixels2"


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
            root, text="Click on the image to select the coordinates"
        )
        self.coords_label.pack()

        self.save_button = tk.Button(
            root, text="Save Points", command=self.on_done
        )
        self.save_button.pack()

        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        x, y = event.x, event.y
        self.coords_label.config(text=f"Coordinates: ({x}, {y})")
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
        print("saving")
        coords_dir = Path(self.coordinates_path) / "coords"
        coords_dir.mkdir(parents=True, exist_ok=True)
        with open(coords_dir / f"{self.pixel_key}_{object_name}.pkl", "wb") as f:
            pickle.dump(self.coords, f)

        images_dir = Path(self.coordinates_path) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(self.img)
        image.save(images_dir / f"{self.pixel_key}.png")
        print("saved")
        self.root.destroy()


def main():
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    img = data["observations"][traj_idx][pixel_key][0]
    if original_bgr:
        img = img[:, :, ::-1]

    root = tk.Tk()
    root.title("Label Points")
    PointLabeler(root, pixel_key, img, coordinates_path, size_multiplier)
    root.mainloop()


if __name__ == "__main__":
    main()
