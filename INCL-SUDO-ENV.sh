#!/bin/bash

echo "=== Starting INCL-SUDO-ENV.sh ==="

################################################################################

# Pass 'sudo' privileges if previously granted in parent scripts.
if [ ! -z "$SUDO_USER" ]; then
  export USER=$SUDO_USER
  echo "Using sudo user: $USER"
fi

################################################################################

echo "=== Installing Docker Community Edition ==="

# Remove older versions of Docker if any.
echo "Removing older Docker versions..."
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do apt-get remove $pkg; done

echo "Updating package lists..."
apt-get update

# Gather required packages for Docker installation.
echo "Installing required packages for Docker..."
apt-get update && apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  software-properties-common

# Add the official Docker GPG key.
echo "Adding Docker GPG key..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo "Adding Docker repository to apt sources..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update

# Install Docker version 'DOCKER_VERSION'.
echo "Installing Docker packages..."
apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

# Test the Docker installation after making sure that the service is running.
echo "Testing Docker installation..."
docker version
docker run --rm hello-world

################################################################################

echo "=== Configuring Docker user permissions ==="

# Add the current user to the 'docker' group to run Docker without 'sudo'.
echo "Adding user to docker group..."
groupadd docker
usermod -aG docker $USER
echo "Added the current user '${USER}' to the 'docker' group."

# Configure the host system so that 'adduser' command adds future new users to the 'docker' group automatically.
echo "Configuring adduser to add new users to docker group..."
ADDUSER_CONFIG=/etc/adduser.conf
if [ ! -f ${ADDUSER_CONFIG} ]; then
  echo "Failed to add future new users to the 'docker' group because the system configuration file '${ADDUSER_CONFIG}' was not found."
else
  if ! grep -q "#EXTRA_GROUPS=\"dialout cdrom floppy audio video plugdev users\"" ${ADDUSER_CONFIG}; then
    echo "Failed to add future new users to the 'docker' group because 'EXTRA_GROUPS' in '${ADDUSER_CONFIG}' has already been customized."
  else
    sed -i 's/#EXTRA_GROUPS="dialout cdrom floppy audio video plugdev users"/EXTRA_GROUPS="dialout cdrom floppy audio video plugdev users docker"/' ${ADDUSER_CONFIG}
    sed -i 's/#ADD_EXTRA_GROUPS=1/ADD_EXTRA_GROUPS=1/' ${ADDUSER_CONFIG}
    echo "Modified '${ADDUSER_CONFIG}' to add all future new users to the 'docker' group upon creation."
  fi
fi

################################################################################

echo "=== Installing Nvidia Container Toolkit ==="

# Remove 'nvidia-docker' and all existing GPU containers.
echo "Adding Nvidia repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install 'nvidia-container-toolkit' and reload the Docker daemon configuration.
echo "Installing nvidia-container-toolkit..."
apt-get update && apt-get install -y \
  nvidia-container-toolkit

echo "Configuring Docker runtime for Nvidia..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Test the Nvidia Docker installation after making sure that the service is running and that Nvidia drivers are found.
echo "Testing Nvidia Docker installation..."
if [ -e /proc/driver/nvidia/version ]; then
  docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
fi

################################################################################

echo "=== Installing Terminator terminal ==="

# Install the latest version of Terminator from the Ubuntu repositories.
echo "Installing Terminator..."
apt-get update && apt-get install -y \
  terminator

# Prevent the Terminator installation to replace the default Ubuntu terminal.
echo "Setting default terminal..."
update-alternatives --set x-terminal-emulator /usr/bin/gnome-terminal.wrapper

echo "=== INCL-SUDO-ENV.sh completed successfully ==="
