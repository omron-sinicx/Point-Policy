#!/usr/bin/env bash
# =============================================================================
# UR5e data processing + visualization pipeline
#
# Steps:
#   1. process_data_human.py   — sync frames, extract videos
#   2. convert_to_pkl_human.py — CoTracker 2D tracking + triangulation → 3D
#      (runs twice across two invocations of this script when annotating:
#       once WITHOUT --process_points to produce the raw pkl that
#       label_points.py reads from, then again WITH --process_points once
#       annotation is done, to actually compute the tracks)
#   3. convert_pkl_human_to_robot.py — hand 3D → robot gripper keypoints
#   4. visualize_reproj.py     — reprojection quality check (calibration sanity)
#   5. visualize_tracks.py     — robot action keypoints overlay on camera views
#   6. visualize_3d.py         — interactive 3D hand vs robot comparison (HTML)
#
# Usage:
#   bash ur5e_pipeline/run_pipeline.sh <TASK_NAME> [OPTIONS]
#
# Required:
#   TASK_NAME   Name of the task (matches folder under processed_data/)
#
# Options (override via environment variables before calling the script):
#   DATA_DIR       Root data directory            (default: /root/Point-Policy/data)
#   CALIB_PATH     Path to calib.npy              (default: /root/Point-Policy/calib/calib.npy)
#   REPO_ROOT      Repository root                (default: /root/Point-Policy)
#   ENV_NAME       expert_demos subfolder         (default: franka_env)
#   NUM_DEMOS      Limit demos processed          (default: all)
#   VIS_DEMO_IDX   Demo index for visualization   (default: 0)
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
#   SKIP_VIDEO     auto (default) — skip step 1 if processed_data/<task>/
#                    already has extracted videos, otherwise run it.
#                  1 — force-skip step 1 regardless.
#                  0 — force-run step 1 regardless (e.g. to add new demos;
#                      note process_data_human.py will prompt interactively
#                      if it finds existing demos for the task).
#   SKIP_TRACKING  Set to 1 to skip step 2 (convert_to_pkl_human.py) (default: 0)
#   SKIP_PROCESS   Set to 1 to skip both step 1 and step 2 (alias for
#                  SKIP_VIDEO=1 SKIP_TRACKING=1)                     (default: 0)
#   SKIP_CONVERT   Set to 1 to skip step 3                          (default: 0)
#   SKIP_VIS       Set to 1 to skip steps 4-6                       (default: 0)
#
# Annotation workflow (labeling object keypoints partway through the pipeline):
#   # 1. Just run it — if coordinates/pick_cup/ is missing annotation files,
#   #    the script extracts videos, produces the raw (untracked) pkl, and
#   #    stops with instructions.
#   bash ur5e_pipeline/run_pipeline.sh pick_cup
#
#   # 2. Label points — edit and run label_points.py (or the notebook) for
#   #    each pixel_key (pixels1, pixels2, ...), pointing pickle_path at
#   #    data/processed_data_pkl/pick_cup.pkl. This writes into
#   #    coordinates/pick_cup/.
#
#   # 3. Re-run the same command: annotations are now found, so the script
#   #    runs step 2 WITH --process_points straight through to the end.
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
REPO_ROOT="${REPO_ROOT:-/root/Point-Policy}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
CALIB_PATH="${CALIB_PATH:-$REPO_ROOT/calib/calib.npy}"
ENV_NAME="${ENV_NAME:-franka_env}"
NUM_DEMOS="${NUM_DEMOS:-}"          # empty = process all
VIS_DEMO_IDX="${VIS_DEMO_IDX:-0}"
VIS_FPS="${VIS_FPS:-30}"
ANNOTATE="${ANNOTATE:-auto}"
PIXEL_KEYS="${PIXEL_KEYS:-pixels1 pixels2}"
ANNOTATED_LABELS="${ANNOTATED_LABELS:-objects}"
SKIP_PROCESS="${SKIP_PROCESS:-0}"
SKIP_VIDEO="${SKIP_VIDEO:-auto}"
SKIP_TRACKING="${SKIP_TRACKING:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_VIS="${SKIP_VIS:-0}"
USE_GT_DEPTH="${USE_GT_DEPTH:-0}"

# SKIP_PROCESS is a convenience alias covering both sub-steps
if [[ "$SKIP_PROCESS" == "1" ]]; then
    SKIP_VIDEO=1
    SKIP_TRACKING=1
fi

FRANKA_DIR="$REPO_ROOT/point_policy/robot_utils/franka"
UR5E_DIR="$REPO_ROOT/point_policy/robot_utils/ur5e"
VIS_DIR="$REPO_ROOT/ur5e_pipeline"

# Build optional flags
GT_DEPTH_FLAG=""
if [[ "$USE_GT_DEPTH" == "1" ]]; then
    GT_DEPTH_FLAG="--use_gt_depth"
    PKL_TASK_NAME="${TASK_NAME}_gt_depth"
else
    PKL_TASK_NAME="$TASK_NAME"
fi

VIS_OUT_DIR="$DATA_DIR/vis/${PKL_TASK_NAME}"
RAW_PKL="$DATA_DIR/processed_data_pkl/${PKL_TASK_NAME}.pkl"
COORDS_DIR="$REPO_ROOT/coordinates/${PKL_TASK_NAME}"

NUM_DEMOS_FLAG=""
if [[ -n "$NUM_DEMOS" ]]; then
    NUM_DEMOS_FLAG="--num_demos $NUM_DEMOS"
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
# needs, then the script stops so you can annotate before the expensive
# tracking pass runs.
# ---------------------------------------------------------------------------
if [[ "$NEED_ANNOTATION" == "1" ]]; then
    step "Step 2 (raw pass) — convert_to_pkl_human.py (no tracking, for annotation)"
    (
        cd "$FRANKA_DIR"
        python3 convert_to_pkl_human.py \
            --data_dir "$DATA_DIR" \
            --calib_path "$CALIB_PATH" \
            --task_names "$TASK_NAME" \
            $NUM_DEMOS_FLAG \
            $GT_DEPTH_FLAG
    )

    echo -e "\n${BOLD}${GREEN}Raw pkl ready for annotation:${RESET} $RAW_PKL"
    echo -e "${YELLOW}Next steps:${RESET}"
    echo "  1. Edit point_policy/robot_utils/franka/label_points.py (or the .ipynb):"
    echo "       task_name    = \"$PKL_TASK_NAME\""
    echo "       pickle_path  = \"$RAW_PKL\""
    echo "       pixel_key    = \"pixels1\"   # then repeat for pixels2, etc."
    echo "  2. Run it once per camera pixel_key, clicking points in the same"
    echo "     order each time. This writes into $COORDS_DIR/."
    echo "  3. Re-run the same command — annotations will now be found, and"
    echo "     step 1 will auto-skip since videos are already extracted:"
    echo "       bash ur5e_pipeline/run_pipeline.sh $TASK_NAME"
    exit 0
fi

if [[ "$SKIP_TRACKING" != "1" ]]; then
    step "Step 2 — convert_to_pkl_human.py (CoTracker tracking + 3D triangulation)"
    (
        cd "$FRANKA_DIR"
        python3 convert_to_pkl_human.py \
            --data_dir "$DATA_DIR" \
            --calib_path "$CALIB_PATH" \
            --task_names "$TASK_NAME" \
            --process_points \
            $NUM_DEMOS_FLAG \
            $GT_DEPTH_FLAG
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
# Step 4 — visualize_reproj.py  (calibration / triangulation sanity check)
# ---------------------------------------------------------------------------
if [[ "$SKIP_VIS" != "1" ]]; then
    HUMAN_PKL="$DATA_DIR/processed_data_pkl/${PKL_TASK_NAME}.pkl"
    ROBOT_PKL="$DATA_DIR/processed_data_pkl/expert_demos/${ENV_NAME}/${PKL_TASK_NAME}.pkl"

    step "Step 4/6 — visualize_reproj.py (reprojection sanity check)"
    if [[ -f "$HUMAN_PKL" ]]; then
        python3 "$VIS_DIR/visualize_reproj.py" \
            --pkl_path  "$HUMAN_PKL" \
            --calib_path "$CALIB_PATH" \
            --out_path  "$VIS_OUT_DIR/reproj.mp4" \
            --demo_idx  "$VIS_DEMO_IDX" \
            --fps       "$VIS_FPS"
        info "Saved → $VIS_OUT_DIR/reproj.mp4"
    else
        die "Human pkl not found: $HUMAN_PKL"
    fi

    # ---------------------------------------------------------------------------
    # Step 5 — visualize_tracks.py  (robot action overlay)
    # ---------------------------------------------------------------------------
    step "Step 5/6 — visualize_tracks.py (robot action keypoints)"
    if [[ -f "$ROBOT_PKL" ]]; then
        python3 "$VIS_DIR/visualize_tracks.py" \
            --pkl_path  "$ROBOT_PKL" \
            --frames_pkl_path "$HUMAN_PKL" \
            --out_path  "$VIS_OUT_DIR/robot_action.mp4" \
            --demo_idx  "$VIS_DEMO_IDX" \
            --mode      robot \
            --fps       "$VIS_FPS"
        info "Saved → $VIS_OUT_DIR/robot_action.mp4"
    else
        die "Robot pkl not found: $ROBOT_PKL"
    fi

    # -------------------------------------------------------------------------
    # Step 6 — visualize_3d.py  (interactive 3D hand vs robot comparison)
    # -------------------------------------------------------------------------
    step "Step 6/6 — visualize_3d.py (3D hand vs robot keypoints)"
    if [[ -f "$ROBOT_PKL" ]]; then
        python3 "$VIS_DIR/visualize_3d.py" \
            --pkl_path "$ROBOT_PKL" \
            --out_path "$VIS_OUT_DIR/3d.html" \
            --demo_idx "$VIS_DEMO_IDX"
        info "Saved → $VIS_OUT_DIR/3d.html"
    else
        info "Skipping 3D vis (robot pkl not found)"
    fi
else
    info "SKIP_VIS=1: skipping steps 4–6"
fi

echo -e "\n${BOLD}${GREEN}Pipeline complete.${RESET}"
echo "  Reprojection check : $VIS_OUT_DIR/reproj.mp4"
echo "  Robot action vis   : $VIS_OUT_DIR/robot_action.mp4"
echo "  3D comparison      : $VIS_OUT_DIR/3d.html"
