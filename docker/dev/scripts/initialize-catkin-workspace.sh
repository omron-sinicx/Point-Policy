#!/bin/bash

################################################################################
# Display information in CI for version traceability.
if [[ "${CI}" == "true" ]]; then
  echo -e "\n"
  echo -e "######################## DEBUG DATA ########################"
  echo -e "Current commit: $(cd /root/HSR/ && git rev-parse --short HEAD)"
  echo -e "DOCKER_RUNTIME: $DOCKER_RUNTIME"
  echo -e "DOCKER_IMAGE_VERSION: $DOCKER_IMAGE_VERSION"
  echo -e "NVIDIA_CUDAGL_VERSION: $NVIDIA_CUDAGL_VERSION"
  echo -e "NVIDIA_CUDNN_VERSION: $NVIDIA_CUDNN_VERSION"
  echo -e "ROS_DESKTOP_VERSION: $ROS_DESKTOP_VERSION"
  echo -e "ROS_TMC_VERSION: $ROS_TMC_VERSION"
  echo -e "ROS_GAZEBO_VERSION: $ROS_GAZEBO_VERSION"
  echo -e "PYTHON_PIP_VERSION: $PYTHON_PIP_VERSION"
  echo -e "############################################################"
  echo -e "\n"
fi

################################################################################

# Set non-interactive frontend to avoid prompts
export DEBIAN_FRONTEND=noninteractive

# Pre-configure postfix to avoid interactive prompts
echo "postfix postfix/mailname string localhost" | debconf-set-selections
echo "postfix postfix/main_mailer_type string 'No configuration'" | debconf-set-selections

# Download package lists from Ubuntu repositories.
apt-get update && apt-get install -y python-is-python3

# Install system dependencies required by specific ROS packages.
# http://wiki.ros.org/rosdep
rosdep update

# Set ROS distribution explicitly and source the updated ROS environment.
source /opt/ros/$ROS_DISTRO/setup.bash

################################################################################

# Initialize the underlay workspace with vcstool and install dependencies
cd /root/osx-ur/underlay_ws/ && \
    # Add the main workspace and common paths as safe directories
    git config --global --add safe.directory /root/osx-ur && \
    git config --global --add safe.directory /root/osx-ur/underlay_ws && \
    # Pre-emptively add all expected repository paths from .rosinstall as safe
    if [ -f src/.rosinstall ]; then
        grep -E "^\s*local-name:\s*" src/.rosinstall | sed 's/.*local-name:\s*//' | while read local_name; do
            git config --global --add safe.directory "/root/osx-ur/underlay_ws/src/$local_name"
        done
    fi && \
    # Import repositories with vcstool
    vcs import src < src/.rosinstall --force && \
    # Add any additional git repositories found after import
    find src -name ".git" -type d 2>/dev/null | while read gitdir; do
        repo_path=$(dirname "$gitdir")
        git config --global --add safe.directory "/root/osx-ur/underlay_ws/$repo_path"
    done

rosdep install --from-paths src --ignore-src -r -y 

# Initialize, build and source the underlay workspace.
# Blacklist packages that we do not use but that are part of metapackages we need
cd /root/osx-ur/underlay_ws/ && catkin config -init --extend /opt/ros/one --skiplist robotiq_3f_gripper_articulated_gazebo robotiq_3f_gripper_articulated_gazebo_plugins robotiq_3f_rviz \
                          robotiq_3f_gripper_control robotiq_3f_gripper_rviz robotiq_3f_gripper_joint_state_publisher robotiq_3f_gripper_visualization \
                          ur3_e_moveit_config ur10_e_moveit_config robotiq_2f_gripper_action_server robotiq_ft_sensor \
                          --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev
catkin build -j 10 -s
source /root/osx-ur/underlay_ws/devel/setup.bash

# Install the dependencies of the main workspace
cd /root/osx-ur/catkin_ws/ && \
  rosdep install --from-paths src --ignore-src -r -y

# Source the main Catkin workspace.
cd /root/osx-ur/catkin_ws/ && \
  catkin config -init --extend /root/osx-ur/underlay_ws/devel \
                          --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5 --cmake-args -Wno-dev
catkin build -j 10 -s

# ################################################################################

# Fix permission issues
osx-fix-permission-issues

# Add sourcing lines to .bashrc only if they do not already exist
grep -qxF 'source /opt/ros/one/setup.bash' /root/.bashrc || echo -e '\nsource /opt/ros/one/setup.bash' >> /root/.bashrc
grep -qxF 'source /root/osx-ur/underlay_ws/devel/setup.bash' /root/.bashrc || echo -e '\nsource /root/osx-ur/underlay_ws/devel/setup.bash' >> /root/.bashrc
grep -qxF 'source /root/osx-ur/catkin_ws/devel/setup.bash' /root/.bashrc || echo -e '\nsource /root/osx-ur/catkin_ws/devel/setup.bash' >> /root/.bashrc
# Add alias for plotjuggler if it does not already exist
if ! grep -qxF 'alias plotjuggler="/opt/ros/one/lib/plotjuggler/plotjuggler"' /root/.bashrc; then
    echo 'alias plotjuggler="/opt/ros/one/lib/plotjuggler/plotjuggler"' >> /root/.bashrc
fi
