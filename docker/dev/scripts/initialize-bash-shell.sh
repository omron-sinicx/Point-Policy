#!/bin/bash

################################################################################

# Link the default shell 'sh' to Bash.
alias sh='/bin/bash'

################################################################################

# Configure the terminal.

# Disable flow control. If enabled, inputting 'ctrl+s' locks the terminal until inputting 'ctrl+q'.
stty -ixon

################################################################################

# Configure 'umask' for giving read/write/execute permission to group members.
umask 0002

################################################################################

# Activate the Python virtual environment so the shell prompt shows (venv).
source /opt/venv/bin/activate

################################################################################

# Define Bash functions to conveniently run common Point-Policy tasks.

function pp-install-packages () {
  # Install the submodule packages as editable installs.
  # Run this once after first starting the container if pip cannot find them.
  pushd /root/Point-Policy > /dev/null
  pip install -e "co-tracker/[all]"
  pip install -e Franka-Teach/
  pip install -e franka-env/
  popd > /dev/null
}

function pp-download-checkpoints () {
  # Download the CoTracker checkpoint required by the policy.
  mkdir -p /root/Point-Policy/co-tracker/checkpoints
  wget -q --show-progress \
    https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth \
    -O /root/Point-Policy/co-tracker/checkpoints/scaled_online.pth
  echo "Checkpoint saved to co-tracker/checkpoints/scaled_online.pth"
}

function pp-train () {
  pushd /root/Point-Policy > /dev/null
  python point_policy/train.py "$@"
  popd > /dev/null
}

function pp-eval () {
  pushd /root/Point-Policy > /dev/null
  python point_policy/eval.py "$@"
  popd > /dev/null
}

################################################################################

# Aliases.

alias pp='cd /root/Point-Policy'

################################################################################

# Move to the working directory.
cd /root/Point-Policy/
