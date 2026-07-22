#!/usr/bin/env bash
# =============================================================================
# Set up the main `point-policy` environment: requirements.txt, dift, and
# tapnet submodules, plus the version pins needed on top.
#
# Either activate `conda env create -f conda_env.yaml` first, or (if not
# using conda) just run this directly -- it pip installs requirements.txt
# itself. Idempotent: re-running skips checkpoints already downloaded. Run
# from anywhere -- it cd's to its own location first.
#
# hamer/OmniHands are NOT installed here -- they need their own isolated
# venvs (see instructions/installation_and_data_collection.md) since their
# torch/numpy/mmcv pins conflict with this environment.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

git submodule update --init --recursive

echo "==> [1/4] requirements.txt"
pip install -r requirements.txt

echo "==> [2/4] dift"
cd dift
pip install xformers==0.0.29.post1
pip install accelerate  # required by DIFT's SDFeaturizer (enable_sequential_cpu_offload)
git checkout main
cd "$REPO_ROOT"

echo "==> [3/4] tapnet (TAPIR point tracking)"
pip install -e tapnet
bash download_tapir_models.sh

echo "==> [4/4] Version pins"
# torchvision/mediapipe: conda_env.yaml installs torchvision unpinned
# (pytorch::torchvision) and doesn't install mediapipe at all, so these are
# this repo's actual version-pinning step, not a fix-up.
#
# torch/torchvision are pinned to 2.10.0/0.25.0, NOT the older 2.5.0/0.20.0
# pair dift/xformers were originally built against -- dependencies/lerobot_v3
# requires torchvision>=0.21.0 (hence torch>=~2.6), and that constraint wins
# since lerobot shares this same environment. xformers==0.0.29.post1 (dift,
# below) doesn't support torch this new and will fail to load at runtime;
# dift_sd.py's SDFeaturizer catches that and falls back to
# enable_attention_slicing() (slower but functional) instead of crashing.
pip install torchvision==0.25.0
pip install mediapipe==0.10.11
# transformers/huggingface_hub/numpy are already pinned correctly in
# requirements.txt/conda_env.yaml, but dift/tapnet's own installs above
# transitively drag in different versions -- these --force-reinstalls just
# re-assert the versions this repo actually needs, so they must stay last.
pip install --force-reinstall transformers==4.45.2
pip install --force-reinstall huggingface_hub==0.36.2
pip install --force-reinstall numpy==2.1.3
pip install --force-reinstall torch==2.10.0

echo ""
echo "Done. point-policy environment ready (requirements.txt, dift, tapnet installed)."
