#!/bin/bash

################################################################################

# Install the SSH deploy key of the GitLab repository.
# Set up SSH key manually for security.
# Github: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
# Gitlab: https://docs.gitlab.com/ee/user/ssh.html

################################################################################

# Modify the HTTPS URL of the Git remote origin to use SSH:
# Enable authentication by using '~/.ssh/id_rsa' instead of a GitLab account.
GIT_HTTPS_URL=`git config --get remote.origin.url`
GIT_SSH_URL=`echo ${GIT_HTTPS_URL} | sed -e 's/^https\{0,1\}:\/\//git@/' -e 's/com\//com:/'`
if [ "${GIT_HTTPS_URL}" != "${GIT_SSH_URL}" ]; then
  echo "Updated the Git remote origin from '${GIT_HTTPS_URL}' to '${GIT_SSH_URL}'."
  echo "Execute 'git config --get remote.origin.url' to manually confirm if needed."
  git remote set-url origin ${GIT_SSH_URL}
fi

################################################################################

# Download and initialize all the Git submodules recursively.
# https://git-scm.com/book/en/v2/Git-Tools-Submodules
git submodule update --init --recursive

################################################################################

# Setup the Bash shell environment with '.bashrc'.

# Force color prompt in terminal.
sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

# Set default Docker runtime to use in './docker/docker-compose.yml'.
if ! grep -q "export DOCKER_RUNTIME" ~/.bashrc; then
  if [ -e /proc/driver/nvidia/version ]; then
    DOCKER_RUNTIME=nvidia
  else
    DOCKER_RUNTIME=runc
  fi
  cat <<EOF >> ~/.bashrc

# Set default Docker runtime to use in '~/osx-ur/docker/docker-compose.yml':
# 'runc' (Docker default) or 'nvidia' (Nvidia Docker 2).
export DOCKER_RUNTIME=${DOCKER_RUNTIME}
EOF
fi

# Create .env file with host user and group IDs
cat <<EOF > .env
# Host user and group IDs for Docker container
HOST_UID=$(id -u)
HOST_GID=$(id -g)
DOCKER_RUNTIME=${DOCKER_RUNTIME}
EOF

echo "Created .env file with host user and group IDs"
