#!/bin/bash

################################################################################

# Download package lists from Ubuntu repositories.
apt-get update

# Install system dependencies required by specific ROS packages.
# http://wiki.ros.org/rosdep
rosdep update

# TODO: Does this have any effect on the current shell?
# Source the ROS environment.
source /opt/ros/$ROS_DISTRO/setup.bash

################################################################################

# Remove the Catkin workspace:
# Delete expected 'catkin build' artefacts.
cd /root/osx-ur/catkin_ws/ && catkin clean -y
cd /root/osx-ur/catkin_ws/ && rm -r CMakeLists.txt .catkin_tools/
# Delete unexpected 'catkin_make' artefacts.
cd /root/osx-ur/catkin_ws/ && rm src/CMakeLists.txt

################################################################################

# Remove the underlay workspace:
# Delete expected 'catkin build' artefacts.
cd /root/osx-ur/underlay_ws/ && catkin clean -y
cd /root/osx-ur/underlay_ws/ && rm -r CMakeLists.txt .catkin_tools/
# Delete unexpected 'catkin_make' artefacts.
cd /root/osx-ur/underlay_ws/ && rm src/CMakeLists.txt

################################################################################

bash /root/osx-ur/docker/dev/scripts/initialize-catkin-workspace.sh
