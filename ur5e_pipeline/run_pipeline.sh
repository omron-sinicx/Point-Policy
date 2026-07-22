#!/usr/bin/env bash
# =============================================================================
# UR5e data processing + visualization pipeline
#
# Steps:
#   1. process_data_human.py   — sync frames, extract videos
#   2. $CONVERT_SCRIPT (see below) — hand tracking + CoTracker object 2D
#      tracking + triangulation → 3D. TEMPORARILY set to
#      convert_to_pkl_human_omnihands.py to try OmniHands (native
#      multi-camera mode: both of this project's 2 synced cameras fed into
#      one forward pass per frame) for the hand. Set CONVERT_SCRIPT to
#      convert_to_pkl_human_hamer.py for HaMeR (per-camera) instead, or
#      convert_to_pkl_human.py to go back to MediaPipe.
#      (runs twice within a single invocation when annotating: once WITHOUT
#       --process_points to produce the raw pkl that label_points.py reads
#       from -- which this script then launches automatically -- then again
#       WITH --process_points once annotation is done, to compute the tracks)
#   3. convert_pkl_human_to_robot.py — hand 3D → robot gripper keypoints
#   4. visualize_reproj.py     — reprojection quality check (calibration sanity)
#   5. visualize_tracks.py     — robot action keypoints overlay on camera views
#   6. visualize_3d.py         — interactive 3D hand vs robot comparison (HTML)
#   7. convert_pkl_to_lerobot.py — export to a LeRobotDataset (tracked object
#      keypoints + current EE cartesian pose -> next EE cartesian pose)
#
# Usage:
#   bash ur5e_pipeline/run_pipeline.sh <TASK_NAME> [OPTIONS]
#
# Required:
#   TASK_NAME   Name of the task (matches folder under processed_data/)
#
# Options (override via environment variables before calling the script):
#   DATA_DIR       Root data directory            (default: osx-ur/data, i.e. two
#                                                   levels above $REPO_ROOT)
#   CALIB_PATH     Path to calib.npy              (default: $REPO_ROOT/calib/calib.npy)
#   REPO_ROOT      Repository root                (default: parent dir of this script,
#                                                   i.e. wherever Point-Policy is checked out)
#   ENV_NAME       expert_demos subfolder         (default: franka_env)
#   NUM_DEMOS      Limit demos processed          (default: all)
#   VIS_DEMO_IDX   Demo index for visualization, or "all" to visualize every
#                  episode in the dataset (one output file per demo)   (default: 0)
#   VIS_FPS        Output video FPS               (default: 30)
#   ANNOTATE       auto (default) — check coordinates/<task>/ and decide:
#                    if annotation files are missing, stop after producing
#                    the raw (untracked) pkl so you can run label_points.py;
#                    if they're all present, run the full pipeline.
#                  1 — force-stop after the raw pkl even if annotations
#                      already exist (e.g. to redo labeling).
#                  0 — skip the check entirely and always run the full
#                      pipeline (old default, pre-auto-detection behavior).
#   PIXEL_KEYS         Camera pixel keys to check/annotate, space-separated
#                      (default: "pixels1 pixels2" — must match
#                      camera_indices in convert_to_pkl_human.py)
#   ANNOTATED_LABELS   Object labels that require manual coords (default:
#                      "objects" — human_hand is auto-detected, no
#                      annotation needed)
#   SKIP_VIDEO     1 (default) — skip step 1. collect_image_data.py (the ROS
#                    collection script) writes straight into
#                    processed_data/<task>/demonstration_N/ already, so step 1
#                    (process_data_human.py) has nothing left to do for demos
#                    collected that way.
#                  auto — skip step 1 only if processed_data/<task>/ already
#                    has extracted videos, otherwise run it. Only useful for
#                    the legacy collect_data_realsense.py flow, which still
#                    needs process_data_human.py to sync/extract raw
#                    extracted_data/ into processed_data/.
#                  0 — force-run step 1 regardless (e.g. to add new demos
#                      collected via the legacy collect_data_realsense.py;
#                      note process_data_human.py will prompt interactively
#                      if it finds existing demos for the task).
#   SKIP_TRACKING  Set to 1 to skip step 2 (convert_to_pkl_human.py) (default: 0)
#   SKIP_PROCESS   Set to 1 to skip both step 1 and step 2 (alias for
#                  SKIP_VIDEO=1 SKIP_TRACKING=1)                     (default: 0)
#   SKIP_CONVERT   Set to 1 to skip step 3                          (default: 0)
#   SKIP_VIS       Set to 1 to skip steps 4-6                       (default: 0)
#   SKIP_LEROBOT   Set to 1 to skip step 7                          (default: 0)
#   LEROBOT_REPO_ID  repo_id for the step 7 LeRobotDataset export
#                    (default: ${PKL_TASK_NAME}_lerobot)
#   EXCLUDE_DEMOS  Space-separated demo id(s) to leave out of the step 7
#                  LeRobotDataset export (e.g. a bad demo)  (default: none)
#   HAND_SIDE      "left" or "right" to force which hand the HaMeR worker
#                  tracks (see --hand_side on convert_to_pkl_human_hamer.py).
#                  Empty (default) auto-picks the more confident side. Only
#                  meaningful when CONVERT_SCRIPT is the HaMeR script.
#
# Annotation workflow (labeling object keypoints partway through the pipeline):
#   Just run it — if coordinates/pick_cup/ is missing annotation files, the
#   script extracts videos, produces the raw (untracked) pkl, then
#   automatically launches label_points.py for you (one Tkinter window per
#   PIXEL_KEYS entry, in order). Click the same object point(s) in the same
#   order in each window, then click "Save Points" to move to the next one.
#   Once every pixel_key is labeled, the SAME run continues straight through
#   the tracked pass to the end -- no need to re-run the command yourself.
#   bash ur5e_pipeline/run_pipeline.sh pick_cup
#
# Examples:
#   # Full pipeline (annotation auto-detected — runs straight through if
#   # coordinates/<task>/ is already fully labeled)
#   bash ur5e_pipeline/run_pipeline.sh pick_cup
#
#   # With ground-truth depth
#   USE_GT_DEPTH=1 bash ur5e_pipeline/run_pipeline.sh pick_cup
#
#   # Skip heavy processing (steps 1-2), re-run step 3 + visualization
#   SKIP_PROCESS=1 bash ur5e_pipeline/run_pipeline.sh pick_cup
#
#   # Skip everything except visualization
#   SKIP_PROCESS=1 SKIP_CONVERT=1 bash ur5e_pipeline/run_pipeline.sh pick_cup
#
#   # Limit to first 5 demos
#   NUM_DEMOS=5 bash ur5e_pipeline/run_pipeline.sh pick_cup
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: bash ur5e_pipeline/run_pipeline.sh <TASK_NAME> [options as env vars]"
    exit 1
fi

TASK_NAME="$1"

# Defaults (override by setting env vars before calling the script)
# REPO_ROOT defaults to the parent of this script's directory (ur5e_pipeline/..),
# so it works regardless of where this repo is checked out.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
# Data lives one level above this repo (osx-ur/data), shared with the rest of
# osx-ur (e.g. LeRobotDataset output from osx_ur5e/scripts/data_collection.py),
# not inside dependencies/Point-Policy/.
OSX_UR_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$OSX_UR_ROOT/data}"
CALIB_PATH="${CALIB_PATH:-$REPO_ROOT/calib/calib.npy}"
ENV_NAME="${ENV_NAME:-franka_env}"
NUM_DEMOS="${NUM_DEMOS:-}"          # empty = process all
VIS_DEMO_IDX="${VIS_DEMO_IDX:-0}"
VIS_FPS="${VIS_FPS:-30}"
ANNOTATE="${ANNOTATE:-auto}"
PIXEL_KEYS="${PIXEL_KEYS:-pixels1 pixels2}"
ANNOTATED_LABELS="${ANNOTATED_LABELS:-objects}"
SKIP_PROCESS="${SKIP_PROCESS:-0}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
SKIP_TRACKING="${SKIP_TRACKING:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_VIS="${SKIP_VIS:-0}"
SKIP_LEROBOT="${SKIP_LEROBOT:-0}"
EXCLUDE_DEMOS="${EXCLUDE_DEMOS:-}"    # space-separated demo ids to leave out of the LeRobot export (step 7)
USE_GT_DEPTH="${USE_GT_DEPTH:-0}"
# HAND_SIDE: "left" or "right" to force which hand the HaMeR/OmniHands worker
# tracks, or empty (default) to let it auto-pick the more confident side.
# Understood by convert_to_pkl_human_hamer.py's and
# convert_to_pkl_human_omnihands.py's --hand_side flag -- leave this unset if
# CONVERT_SCRIPT is convert_to_pkl_human.py (MediaPipe), which has no such
# flag and would error on it. Useful when ViTPose confuses left/right on a
# close-up single-hand crop.
HAND_SIDE="${HAND_SIDE:-}"
# VITPOSE_ON_GPU: set to 1 to run ViTPose+-Huge (hand bbox localization) on
# GPU instead of CPU. ViTPose is the per-frame compute bottleneck, so this
# is much faster, but needs a few extra GB of VRAM -- 8GB is tight alongside
# the OmniHands/HaMeR model and this process's own CoTracker/DIFT models.
# Understood by convert_to_pkl_human_omnihands.py and
# convert_to_pkl_human_hamer.py; ignored by MediaPipe.
VITPOSE_ON_GPU="${VITPOSE_ON_GPU:-0}"
# BODY_DETECTOR: "vitdet" or "regnety" to pick HaMeR's person-detector
# backbone for hand-bbox localization, or empty (default) to use the
# worker's own default ("regnety", the lighter one). "vitdet" is HaMeR's
# original, heavier and more accurate ViTDet-H detector. Understood by
# convert_to_pkl_human_hamer.py's --body_detector flag; ignored by
# OmniHands/MediaPipe.
BODY_DETECTOR="${BODY_DETECTOR:-}"

# SKIP_PROCESS is a convenience alias covering both sub-steps
if [[ "$SKIP_PROCESS" == "1" ]]; then
    SKIP_VIDEO=1
    SKIP_TRACKING=1
fi

FRANKA_DIR="$REPO_ROOT/point_policy/robot_utils/franka"
UR5E_DIR="$REPO_ROOT/point_policy/robot_utils/ur5e"
VIS_DIR="$REPO_ROOT/ur5e_pipeline"

# TEMPORARY: swap the Step 2 hand-tracking script to try HaMeR (per-camera)
# instead of MediaPipe. Set to "convert_to_pkl_human_omnihands.py" for
# OmniHands (native multi-camera mode), or revert to "convert_to_pkl_human.py"
# for MediaPipe.
CONVERT_SCRIPT="convert_to_pkl_human_hamer.py"
# CONVERT_SCRIPT="convert_to_pkl_human.py"

# Build optional flags
GT_DEPTH_FLAG=""
if [[ "$USE_GT_DEPTH" == "1" ]]; then
    GT_DEPTH_FLAG="--use_gt_depth"
    PKL_TASK_NAME="${TASK_NAME}_gt_depth"
else
    PKL_TASK_NAME="$TASK_NAME"
fi

HAND_SIDE_FLAG=""
# --hand_side only exists on the HaMeR/OmniHands scripts; guard against
# passing it to the MediaPipe script (which would error on an unknown arg) --
# same gating as VITPOSE_ON_GPU_FLAG/BODY_DETECTOR_FLAG below.
if [[ -n "$HAND_SIDE" && ( "$CONVERT_SCRIPT" == "convert_to_pkl_human_omnihands.py" || "$CONVERT_SCRIPT" == "convert_to_pkl_human_hamer.py" ) ]]; then
    HAND_SIDE_FLAG="--hand_side $HAND_SIDE"
fi

VITPOSE_ON_GPU_FLAG=""
# --vitpose_on_gpu only exists on the OmniHands and HaMeR scripts; guard
# against passing it to the MediaPipe script (which would error on an
# unknown arg).
if [[ "$VITPOSE_ON_GPU" == "1" && ( "$CONVERT_SCRIPT" == "convert_to_pkl_human_omnihands.py" || "$CONVERT_SCRIPT" == "convert_to_pkl_human_hamer.py" ) ]]; then
    VITPOSE_ON_GPU_FLAG="--vitpose_on_gpu"
fi

BODY_DETECTOR_FLAG=""
# --body_detector only exists on the HaMeR script; guard against passing it
# to the OmniHands/MediaPipe scripts (which would error on an unknown arg).
if [[ -n "$BODY_DETECTOR" && "$CONVERT_SCRIPT" == "convert_to_pkl_human_hamer.py" ]]; then
    BODY_DETECTOR_FLAG="--body_detector $BODY_DETECTOR"
fi

VIS_OUT_DIR="$DATA_DIR/vis/${PKL_TASK_NAME}"
# RAW_PKL/HUMAN_PKL/ROBOT_PKL are directories of per-demo files (one pkl per
# demo + meta.pkl -- see point_utils/task_pkl_io.py), not single pickle files
# -- avoids holding every demo's data in memory at once.
RAW_PKL="$DATA_DIR/processed_data_pkl/${PKL_TASK_NAME}"
COORDS_DIR="$REPO_ROOT/coordinates/${PKL_TASK_NAME}"
HUMAN_PKL="$DATA_DIR/processed_data_pkl/${PKL_TASK_NAME}"
ROBOT_PKL="$DATA_DIR/processed_data_pkl/expert_demos/${ENV_NAME}/${PKL_TASK_NAME}"
LEROBOT_REPO_ID="${LEROBOT_REPO_ID:-${PKL_TASK_NAME}_lerobot}"

NUM_DEMOS_FLAG=""
if [[ -n "$NUM_DEMOS" ]]; then
    NUM_DEMOS_FLAG="--num_demos $NUM_DEMOS"
fi

EXCLUDE_DEMOS_FLAG=""
if [[ -n "$EXCLUDE_DEMOS" ]]; then
    EXCLUDE_DEMOS_FLAG="--exclude_demos $EXCLUDE_DEMOS"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

step() { echo -e "\n${BOLD}${GREEN}=== $* ===${RESET}"; }
info() { echo -e "${YELLOW}  $*${RESET}"; }
die()  { echo -e "${RED}ERROR: $*${RESET}" >&2; exit 1; }

# Returns 0 (true) if any expected annotation file is missing under COORDS_DIR.
annotations_missing() {
    local missing=()
    for pk in $PIXEL_KEYS; do
        [[ -f "$COORDS_DIR/images/${pk}.png" ]] || missing+=("images/${pk}.png")
        for label in $ANNOTATED_LABELS; do
            [[ -f "$COORDS_DIR/coords/${pk}_${label}.pkl" ]] || missing+=("coords/${pk}_${label}.pkl")
        done
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        info "Missing annotation file(s) under $COORDS_DIR:"
        for m in "${missing[@]}"; do info "  - $m"; done
        return 0
    fi
    return 1
}

# Returns 0 (true) if step 1 has already extracted videos for this task.
videos_exist() {
    local proc_dir="$DATA_DIR/processed_data/$TASK_NAME"
    [[ -d "$proc_dir" ]] || return 1
    local d
    for d in "$proc_dir"/*/; do
        [[ -d "${d}videos" ]] || continue
        [[ -n "$(ls -A "${d}videos" 2>/dev/null)" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
[[ -d "$DATA_DIR" ]]  || die "DATA_DIR not found: $DATA_DIR"
[[ -f "$CALIB_PATH" ]] || die "CALIB_PATH not found: $CALIB_PATH"

info "TASK_NAME  : $TASK_NAME"
info "DATA_DIR   : $DATA_DIR"
info "CALIB_PATH : $CALIB_PATH"
info "ENV_NAME   : $ENV_NAME"
[[ -n "$NUM_DEMOS" ]] && info "NUM_DEMOS  : $NUM_DEMOS"
[[ "$USE_GT_DEPTH" == "1" ]] && info "GT DEPTH   : enabled"

# ---------------------------------------------------------------------------
# Decide whether this run should stop after the raw pkl (for annotation) or
# proceed through the full tracked pipeline.
# ---------------------------------------------------------------------------
case "$ANNOTATE" in
    1)
        NEED_ANNOTATION=1
        info "ANNOTATE=1: will stop after raw pkl regardless of existing annotations"
        ;;
    0)
        NEED_ANNOTATION=0
        info "ANNOTATE=0: skipping annotation check, running full pipeline"
        ;;
    auto)
        if annotations_missing; then
            NEED_ANNOTATION=1
            info "Annotation data incomplete for $PKL_TASK_NAME -- will stop after raw pkl"
        else
            NEED_ANNOTATION=0
            info "Annotation data found for $PKL_TASK_NAME -- running full pipeline"
        fi
        ;;
    *)
        die "Invalid ANNOTATE value: $ANNOTATE (expected auto, 1, or 0)"
        ;;
esac

# ---------------------------------------------------------------------------
# Decide whether step 1 (video extraction) needs to run.
# ---------------------------------------------------------------------------
case "$SKIP_VIDEO" in
    1)
        info "SKIP_VIDEO=1: skipping step 1 regardless"
        ;;
    0)
        info "SKIP_VIDEO=0: running step 1 regardless"
        ;;
    auto)
        if videos_exist; then
            SKIP_VIDEO=1
            info "Videos already extracted for $TASK_NAME -- skipping step 1"
        else
            SKIP_VIDEO=0
            info "No extracted videos found for $TASK_NAME -- running step 1"
        fi
        ;;
    *)
        die "Invalid SKIP_VIDEO value: $SKIP_VIDEO (expected auto, 1, or 0)"
        ;;
esac

mkdir -p "$VIS_OUT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — process_data_human.py
# ---------------------------------------------------------------------------
if [[ "$SKIP_VIDEO" != "1" ]]; then
    step "Step 1 — process_data_human.py"
    (
        cd "$FRANKA_DIR"
        python3 process_data_human.py \
            --data_dir "$DATA_DIR" \
            --task_names "$TASK_NAME" \
            $NUM_DEMOS_FLAG \
            $GT_DEPTH_FLAG
    )
else
    info "SKIP_VIDEO=1: skipping step 1"
fi

# ---------------------------------------------------------------------------
# Step 2 — convert_to_pkl_human.py  (CoTracker + triangulation)
#
# When annotation is needed we deliberately omit --process_points: this
# produces the raw pkl (frames + states, no tracks) that label_points.py
# needs, then label_points.py is launched automatically (below) so you can
# annotate before the expensive tracking pass runs, all in this one run.
# ---------------------------------------------------------------------------
if [[ "$NEED_ANNOTATION" == "1" ]]; then
    step "Step 2 (raw pass) — $CONVERT_SCRIPT (no tracking, for annotation)"
    (
        cd "$FRANKA_DIR"
        python3 "$CONVERT_SCRIPT" \
            --data_dir "$DATA_DIR" \
            --calib_path "$CALIB_PATH" \
            --task_names "$TASK_NAME" \
            $NUM_DEMOS_FLAG \
            $GT_DEPTH_FLAG \
            $HAND_SIDE_FLAG \
            $VITPOSE_ON_GPU_FLAG \
            $BODY_DETECTOR_FLAG
    )

    echo -e "\n${BOLD}${GREEN}Raw pkl ready:${RESET} $RAW_PKL"

    step "Annotate object keypoints — label_points.py"
    info "A window will open for each camera view ($PIXEL_KEYS)."
    info "Click the same object point(s) in the same order in every view,"
    info "then click 'Save Points' to close that view and move to the next."
    (
        cd "$FRANKA_DIR"
        python3 label_points.py \
            --task_name "$PKL_TASK_NAME" \
            --data_dir "$DATA_DIR" \
            --pixel_keys $PIXEL_KEYS
    )
    info "Annotation complete -- continuing with the tracked pass."
fi

if [[ "$SKIP_TRACKING" != "1" ]]; then
    step "Step 2 — $CONVERT_SCRIPT (hand/object tracking + 3D triangulation)"
    (
        cd "$FRANKA_DIR"
        python3 "$CONVERT_SCRIPT" \
            --data_dir "$DATA_DIR" \
            --calib_path "$CALIB_PATH" \
            --task_names "$TASK_NAME" \
            --process_points \
            $NUM_DEMOS_FLAG \
            $GT_DEPTH_FLAG \
            $HAND_SIDE_FLAG \
            $VITPOSE_ON_GPU_FLAG \
            $BODY_DETECTOR_FLAG
    )
else
    info "SKIP_TRACKING=1: skipping step 2"
fi

# ---------------------------------------------------------------------------
# Step 3 — convert_pkl_human_to_robot.py  (hand → UR5e gripper keypoints)
# ---------------------------------------------------------------------------
if [[ "$SKIP_CONVERT" != "1" ]]; then
    step "Step 3 — convert_pkl_human_to_robot.py (hand → robot gripper keypoints)"
    (
        cd "$UR5E_DIR"
        python3 convert_pkl_human_to_robot.py \
            --data_dir "$DATA_DIR" \
            --calib_path "$CALIB_PATH" \
            --task_name "$TASK_NAME" \
            --env_name "$ENV_NAME" \
            $GT_DEPTH_FLAG
    )
else
    info "SKIP_CONVERT=1: skipping step 3"
fi

# ---------------------------------------------------------------------------
# Steps 4-6 — visualize_reproj.py / visualize_tracks.py / visualize_3d.py
# ---------------------------------------------------------------------------
if [[ "$SKIP_VIS" != "1" ]]; then
    if [[ "$VIS_DEMO_IDX" == "all" ]]; then
        N_DEMOS=$(python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT/point_policy')
from point_utils import task_pkl_io
print(len(task_pkl_io.iter_demo_ids('$ROBOT_PKL')))
")
        VIS_DEMO_INDICES=($(seq 0 $((N_DEMOS - 1))))
        info "VIS_DEMO_IDX=all: visualizing all $N_DEMOS demo(s)"
    else
        VIS_DEMO_INDICES=("$VIS_DEMO_IDX")
    fi

    for VIS_IDX in "${VIS_DEMO_INDICES[@]}"; do
        SUFFIX=""
        [[ "$VIS_DEMO_IDX" == "all" ]] && SUFFIX="_demo${VIS_IDX}"

        step "Step 4/6 — visualize_reproj.py (reprojection sanity check) [demo $VIS_IDX]"
        if [[ -d "$HUMAN_PKL" ]]; then
            python3 "$VIS_DIR/visualize_reproj.py" \
                --pkl_path  "$HUMAN_PKL" \
                --calib_path "$CALIB_PATH" \
                --out_path  "$VIS_OUT_DIR/reproj${SUFFIX}.mp4" \
                --demo_idx  "$VIS_IDX" \
                --fps       "$VIS_FPS"
            info "Saved → $VIS_OUT_DIR/reproj${SUFFIX}.mp4"
        else
            die "Human pkl not found: $HUMAN_PKL"
        fi

        step "Step 5/6 — visualize_tracks.py (robot action keypoints) [demo $VIS_IDX]"
        if [[ -d "$ROBOT_PKL" ]]; then
            python3 "$VIS_DIR/visualize_tracks.py" \
                --pkl_path  "$ROBOT_PKL" \
                --frames_pkl_path "$HUMAN_PKL" \
                --out_path  "$VIS_OUT_DIR/robot_action${SUFFIX}.mp4" \
                --demo_idx  "$VIS_IDX" \
                --mode      robot \
                --fps       "$VIS_FPS"
            info "Saved → $VIS_OUT_DIR/robot_action${SUFFIX}.mp4"
        else
            die "Robot pkl not found: $ROBOT_PKL"
        fi

        step "Step 6/6 — visualize_3d.py (3D hand vs robot keypoints) [demo $VIS_IDX]"
        if [[ -d "$ROBOT_PKL" ]]; then
            python3 "$VIS_DIR/visualize_3d.py" \
                --pkl_path "$ROBOT_PKL" \
                --out_path "$VIS_OUT_DIR/3d${SUFFIX}.html" \
                --demo_idx "$VIS_IDX"
            info "Saved → $VIS_OUT_DIR/3d${SUFFIX}.html"
        else
            info "Skipping 3D vis (robot pkl not found)"
        fi
    done
else
    info "SKIP_VIS=1: skipping steps 4–6"
fi

# ---------------------------------------------------------------------------
# Step 7 — convert_pkl_to_lerobot.py  (LeRobotDataset export)
# ---------------------------------------------------------------------------
LEROBOT_OUT_DIR="$DATA_DIR/$LEROBOT_REPO_ID"
if [[ "$SKIP_LEROBOT" != "1" ]]; then
    step "Step 7/7 — convert_pkl_to_lerobot.py (LeRobotDataset export)"
    if [[ -d "$ROBOT_PKL" ]]; then
        python3 "$UR5E_DIR/convert_pkl_to_lerobot.py" \
            --data_dir "$DATA_DIR" \
            --task_name "$TASK_NAME" \
            --env_name "$ENV_NAME" \
            --repo_id  "$LEROBOT_REPO_ID" \
            --fps      "$VIS_FPS" \
            --overwrite \
            $GT_DEPTH_FLAG \
            $EXCLUDE_DEMOS_FLAG
        info "Saved → $LEROBOT_OUT_DIR"
    else
        die "Robot pkl not found: $ROBOT_PKL"
    fi
else
    info "SKIP_LEROBOT=1: skipping step 7"
fi

echo -e "\n${BOLD}${GREEN}Pipeline complete.${RESET}"
if [[ "$VIS_DEMO_IDX" == "all" ]]; then
    echo "  Visualizations     : $VIS_OUT_DIR/{reproj,robot_action,3d}_demo*.{mp4,html}"
else
    echo "  Reprojection check : $VIS_OUT_DIR/reproj.mp4"
    echo "  Robot action vis   : $VIS_OUT_DIR/robot_action.mp4"
    echo "  3D comparison      : $VIS_OUT_DIR/3d.html"
fi
echo "  LeRobot dataset    : $LEROBOT_OUT_DIR"
