#!/usr/bin/env bash
# =============================================================================
# Download models + deps needed for the TAPIR-based point-tracking pipeline.
#
#   1. einshape            — required Python dependency of TAPIR (torch model).
#   2. causal BootsTAPIR   — the point-tracking checkpoint (replaces CoTracker).
#   3. SD 2.1 (community)   — Stable Diffusion 2.1 weights for DIFT semantic
#                            correspondence. Stability AI removed the original
#                            `stabilityai/stable-diffusion-2-1`, so we use the
#                            `sd2-community/stable-diffusion-2-1` mirror.
#
# Idempotent: re-running skips anything already present. Run from anywhere.
#
# Usage:
#   bash download_tapir_models.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAPIR_CKPT_DIR="$REPO_ROOT/tapnet/checkpoints"
TAPIR_CKPT="$TAPIR_CKPT_DIR/causal_bootstapir_checkpoint.pt"
TAPIR_URL="https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt"
SD_REPO="sd2-community/stable-diffusion-2-1"

echo "==> [1/3] Installing Python deps (einshape for TAPIR, accelerate for DIFT)"
python3 -c "import einshape" 2>/dev/null && echo "    einshape already installed" || pip install einshape
python3 -c "import accelerate" 2>/dev/null && echo "    accelerate already installed" || pip install accelerate

echo "==> [2/3] Downloading causal BootsTAPIR checkpoint"
if [[ -f "$TAPIR_CKPT" ]]; then
    echo "    already present: $TAPIR_CKPT"
else
    mkdir -p "$TAPIR_CKPT_DIR"
    # -c resumes a partial download if re-run after an interruption.
    wget -c -O "$TAPIR_CKPT" "$TAPIR_URL"
fi

echo "==> [3/3] Fetching Stable Diffusion 2.1 weights for DIFT ($SD_REPO)"
# huggingface-cli caches under ~/.cache/huggingface; re-runs are no-ops once
# cached. DIFT loads it by repo id via from_pretrained, so pre-fetching here
# just warms the cache. Only the subfolders DIFT actually uses are pulled.
huggingface-cli download "$SD_REPO" \
    --include "unet/*" "vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "*.json"

echo ""
echo "Done. TAPIR checkpoint: $TAPIR_CKPT"
echo "     SD 2.1 weights cached for DIFT ($SD_REPO)."
