#!/bin/bash

################################################################################

# Support Intel 3D acceleration (when no Nvidia GPU drivers are found).

# Remove the configuration files related to the GL library provided by Nvidia and
# re-generate '/etc/ld.so.cache' by executing 'ldconfig'. This is required to
# avoid problems when the host system does not have the Nvidia GL library.
if [ "$DOCKER_RUNTIME" = "runc" ]; then
  rm -f /etc/ld.so.conf.d/nvidia.conf /etc/ld.so.conf.d/glvnd.conf
  ldconfig
fi

################################################################################

# Keep the Docker container running in the background.
# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null
