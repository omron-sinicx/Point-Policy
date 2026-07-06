#!/bin/bash

################################################################################

# Source the environment file if it exists
if [ -f "docker.env" ]; then
    source docker.env
fi

# Set the Docker container name from a project name (first argument).
# If no argument is given, use DOCKER_PROJECT_NAME from env file, or fall back to current user name
PROJECT=$1
if [ -z "${PROJECT}" ]; then
    if [ -n "${DOCKER_PROJECT_NAME}" ]; then
        PROJECT=${DOCKER_PROJECT_NAME}
    else
        PROJECT=${USER}
    fi
fi
CONTAINER="${PROJECT}-dev-1"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"

# Run the Docker container in the background.
# Any changes made to './docker/docker-compose.yml' will recreate and overwrite the container.
docker compose -p ${PROJECT} -f ./docker/docker-compose.yml up -d

################################################################################

# Display GUI through X Server by granting full access to any external client.
xhost +

################################################################################

# Enter the Docker container with a Bash shell (with or without a custom command).
case "$3" in
  ( "" )
  case "$2" in
    ( "" )
    docker exec -i -t ${CONTAINER} bash
    ;;
    ( * )
    docker exec -i -t ${CONTAINER} bash -i -c "~/Point-Policy/docker/dev/scripts/run-command-repeatedly.sh $2"
  esac
  ;;
  ( * )
  echo "Failed to enter the Docker container '${CONTAINER}': '$3' is not a valid argument value."
  ;;
esac
