"""
Interactive 3D visualization: hand keypoints (red) vs robot keypoints (blue).

Generates a self-contained HTML file you can open in any browser.
The slider scrubs through frames; 3D scene is fully rotatable/zoomable.

Usage:
    python ur5e_pipeline/visualize_3d.py \\
        --pkl_path data/processed_data_pkl/expert_demos/franka_env/<task>.pkl

Options:
    --pkl_path   Path to robot expert_demos pkl          (required)
    --out_path   Output HTML path   (default: same dir as pkl, <task>_3d.html)
    --demo_idx   Which demonstration to visualize        (default: 0)
    --camera     cam_1 or cam_2                          (default: cam_1)
"""

import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl_path", required=True)
parser.add_argument("--out_path", default=None)
parser.add_argument("--demo_idx", type=int, default=0)
parser.add_argument("--camera", default="cam_1", choices=["cam_1", "cam_2"])
args = parser.parse_args()

_cam2key = {"cam_1": "pixels1", "cam_2": "pixels2"}
pixel_key = _cam2key[args.camera]

pkl_path = Path(args.pkl_path)
out_path = Path(args.out_path) if args.out_path else pkl_path.parent / (pkl_path.stem + "_3d.html")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA = pkl.load(open(pkl_path, "rb"))
n_demos = len(DATA["observations"])
if args.demo_idx >= n_demos:
    print(f"ERROR: demo_idx={args.demo_idx} but only {n_demos} demos in file.")
    sys.exit(1)

obs = DATA["observations"][args.demo_idx]

hand_key  = f"human_tracks_3d_{pixel_key}"
robot_key = f"robot_tracks_3d_{pixel_key}"

if hand_key not in obs:
    print(f"ERROR: '{hand_key}' not found. Available keys: {list(obs.keys())}")
    sys.exit(1)

hand_tracks  = np.array(obs[hand_key])    # (T, 9, 3) — hand landmarks
robot_tracks = np.array(obs[robot_key])   # (T, 9, 3) — robot body keypoints
T = hand_tracks.shape[0]

print(f"Demo {args.demo_idx}: {T} frames, camera={args.camera}")

# ---------------------------------------------------------------------------
# Skeleton connectivity
# Point 0 = wrist center (after Tshift)
# Points 1-2 = fingertip tips (left / right)
# Points 3-5 = mid-finger row
# Points 6-8 = near-wrist row
# ---------------------------------------------------------------------------
ROBOT_EDGES = [
    (0, 6), (0, 7), (0, 8),   # wrist → near row (near_C, near_L, near_R)
    (6, 3), (7, 4), (8, 5),   # near → mid row
    (3, 1), (4, 1),            # mid → tip_L
    (3, 2), (5, 2),            # mid → tip_R
    (4, 5), (7, 8),            # horizontal cross-bars
]

# Hand skeleton — landmark order from points_class.py track_points_hand():
#   0=wrist | 1=idx_MCP  2=idx_PIP  3=idx_DIP  4=idx_TIP
#            | 5=thm_CMC 6=thm_MCP  7=thm_IP   8=thm_TIP
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # index chain: wrist→MCP→PIP→DIP→TIP
    (0, 5), (5, 6), (6, 7), (7, 8),   # thumb chain: wrist→CMC→MCP→IP→TIP
    (4, 8),                             # pinch: index TIP ↔ thumb TIP
]

# Labels for hover
# Hand landmark order (from points_class.py track_points_hand):
#   0=wrist  1=idx_MCP  2=idx_PIP  3=idx_DIP  4=idx_TIP(used for TCP)
#   5=thm_CMC  6=thm_MCP  7=thm_IP  8=thm_TIP(used for TCP)
HAND_LABELS  = ["wrist",
                 "idx_MCP", "idx_PIP", "idx_DIP", "idx_TIP",
                 "thm_CMC", "thm_MCP", "thm_IP",  "thm_TIP"]

# Robot body point order (from gripper_points.py + convert_pkl_human_to_robot.py):
#   0=wrist_center(after Tshift)
#   1=tip_L  2=tip_R
#   3=mid_C  4=mid_L  5=mid_R
#   6=near_C 7=near_L 8=near_R
ROBOT_LABELS = ["wrist_center",
                 "tip_L", "tip_R",
                 "mid_C", "mid_L", "mid_R",
                 "near_C", "near_L", "near_R"]


def _edges_to_xyz(pts, edges):
    """Convert edge list + point array to x/y/z lists with None separators."""
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [pts[i, 0], pts[j, 0], None]
        ys += [pts[i, 1], pts[j, 1], None]
        zs += [pts[i, 2], pts[j, 2], None]
    return xs, ys, zs


# ---------------------------------------------------------------------------
# Static background traces: full trajectory of point 0 (center/wrist)
# ---------------------------------------------------------------------------
static_traces = [
    go.Scatter3d(
        x=hand_tracks[:, 0, 0], y=hand_tracks[:, 0, 1], z=hand_tracks[:, 0, 2],
        mode="lines",
        line=dict(color="rgba(220,80,80,0.25)", width=2),
        name="Hand center trajectory",
        showlegend=True,
        hoverinfo="skip",
    ),
    go.Scatter3d(
        x=robot_tracks[:, 0, 0], y=robot_tracks[:, 0, 1], z=robot_tracks[:, 0, 2],
        mode="lines",
        line=dict(color="rgba(60,100,220,0.25)", width=2),
        name="Robot center trajectory",
        showlegend=True,
        hoverinfo="skip",
    ),
]
N_STATIC = len(static_traces)

# ---------------------------------------------------------------------------
# Per-frame traces: 4 traces (hand scatter, robot scatter, hand edges, robot edges)
# We build them for frame 0 as the initial data, then use animation frames for the rest.
# ---------------------------------------------------------------------------

def make_frame_traces(t):
    h = hand_tracks[t]   # (9,3)
    r = robot_tracks[t]  # (9,3)

    hx, hy, hz = _edges_to_xyz(h, HAND_EDGES)
    rx, ry, rz = _edges_to_xyz(r, ROBOT_EDGES)

    # offset between fingertip midpoints (for the info annotation)
    hand_tip  = (h[3] + h[4]) / 2
    robot_tip = (r[1] + r[2]) / 2
    tip_dist  = np.linalg.norm(hand_tip - robot_tip) * 100   # cm
    wrist_dist = np.linalg.norm(h[0] - r[0]) * 100

    return [
        # 0 — hand skeleton lines
        go.Scatter3d(
            x=hx, y=hy, z=hz, mode="lines",
            line=dict(color="rgba(220,60,60,0.55)", width=3),
            name="Hand skeleton", showlegend=False, hoverinfo="skip",
        ),
        # 1 — robot skeleton lines
        go.Scatter3d(
            x=rx, y=ry, z=rz, mode="lines",
            line=dict(color="rgba(40,90,210,0.55)", width=3),
            name="Robot skeleton", showlegend=False, hoverinfo="skip",
        ),
        # 2 — hand keypoints (red)
        go.Scatter3d(
            x=h[:, 0], y=h[:, 1], z=h[:, 2],
            mode="markers+text",
            marker=dict(color="red", size=6, symbol="circle"),
            text=HAND_LABELS,
            textposition="top center",
            textfont=dict(size=8, color="darkred"),
            name="Hand keypoints",
            showlegend=True,
            customdata=np.column_stack([
                h * 100,
                np.arange(len(HAND_LABELS)),
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "x=%{x:.4f} m  y=%{y:.4f} m  z=%{z:.4f} m"
                "<extra></extra>"
            ),
        ),
        # 3 — robot keypoints (blue)
        go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2],
            mode="markers+text",
            marker=dict(color="blue", size=6, symbol="circle"),
            text=ROBOT_LABELS,
            textposition="top center",
            textfont=dict(size=8, color="navy"),
            name="Robot keypoints",
            showlegend=True,
            customdata=np.column_stack([
                r * 100,
                np.arange(len(ROBOT_LABELS)),
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "x=%{x:.4f} m  y=%{y:.4f} m  z=%{z:.4f} m"
                "<extra></extra>"
            ),
        ),
        # 4 — offset distance annotation (rendered as a single invisible marker
        #     so we can carry frame-level text via the title)
        go.Scatter3d(
            x=[], y=[], z=[], mode="markers",
            name=f"wrist Δ={wrist_dist:.1f}cm  tip Δ={tip_dist:.1f}cm",
            showlegend=True,
            hoverinfo="skip",
            marker=dict(size=0, opacity=0),
        ),
    ]


# ---------------------------------------------------------------------------
# Build animation frames
# ---------------------------------------------------------------------------
frames = []
for t in range(T):
    traces = make_frame_traces(t)
    frames.append(go.Frame(
        data=traces,
        name=str(t),
        traces=list(range(N_STATIC, N_STATIC + len(traces))),
    ))

# ---------------------------------------------------------------------------
# Initial figure
# ---------------------------------------------------------------------------
fig = go.Figure(
    data=static_traces + make_frame_traces(0),
    frames=frames,
)

# ---------------------------------------------------------------------------
# Slider and play button
# ---------------------------------------------------------------------------
slider_steps = []
for t in range(T):
    slider_steps.append(dict(
        args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
        label=str(t),
        method="animate",
    ))

sliders = [dict(
    active=0,
    currentvalue=dict(prefix="Frame: ", visible=True, xanchor="left"),
    pad=dict(b=10, t=10),
    steps=slider_steps,
    x=0.05, len=0.90,
)]

updatemenus = [dict(
    type="buttons", showactive=False,
    x=0.0, y=-0.08, xanchor="left",
    buttons=[
        dict(label="▶ Play",
             method="animate",
             args=[None, dict(frame=dict(duration=60, redraw=True),
                              fromcurrent=True, mode="immediate")]),
        dict(label="⏸ Pause",
             method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                mode="immediate")]),
    ],
)]

# ---------------------------------------------------------------------------
# Axis limits
# ---------------------------------------------------------------------------
all_pts = np.concatenate([hand_tracks.reshape(-1, 3), robot_tracks.reshape(-1, 3)])
pad = 0.05
x_rng = [all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad]
y_rng = [all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad]
z_rng = [all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad]

fig.update_layout(
    title=dict(
        text=f"3D comparison — demo {args.demo_idx}  |  {pkl_path.name}",
        font=dict(size=13),
    ),
    scene=dict(
        xaxis=dict(title="X (m)", range=x_rng),
        yaxis=dict(title="Y (m)", range=y_rng),
        zaxis=dict(title="Z (m)", range=z_rng),
        aspectmode="manual",
        aspectratio=dict(
            x=(x_rng[1] - x_rng[0]),
            y=(y_rng[1] - y_rng[0]),
            z=(z_rng[1] - z_rng[0]),
        ),
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)", borderwidth=1),
    sliders=sliders,
    updatemenus=updatemenus,
    margin=dict(l=0, r=0, b=80, t=60),
    height=750,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out_path), include_plotlyjs="cdn")
print(f"Saved → {out_path}")
print("Open in your browser. Use the slider to scrub frames, drag to rotate, scroll to zoom.")
